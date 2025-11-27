import os
import time
import json
import traceback
from tqdm import tqdm

import opts
import models
from dataloader import DataLoader
import misc.eval_utils as eval_utils
import misc.utils as utils
from modules.loss_wrapper import LossWrapper
import paddle

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def finetune(opt):
    acc_steps = getattr(opt, 'acc_steps', 1)

    loader = DataLoader(opt)
    opt.vocab_size = loader.get_vocab_size()
    opt.seq_length = loader.get_seq_length()

    tb_summary_writer = tb and tb.SummaryWriter(os.path.join(opt.checkpoint_path, 'log_runs/exp'))

    infos = {}
    histories = {}
    val_result_history = {}
    if opt.start_from is not None:
        # Load infos and histories
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], \
                    "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
                val_result_history = histories.get('val_result_history', {})
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['loader_state_dict'] = None
        infos['vocab'] = loader.get_vocab()
    infos['opt'] = opt

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    # Always ignore saved DataLoader sampler state to avoid stale indices
    infos['loader_state_dict'] = None
    loader.load_state_dict(None)

    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    opt.vocab = loader.get_vocab()
    model = models.setup(opt)
    del opt.vocab
    dp_model = paddle.DataParallel(model)
    lw_model = LossWrapper(model, opt)
    dp_lw_model = paddle.DataParallel(lw_model)

    # Load pretrained weights selectively
    if opt.start_from is not None:
        ckpt_path = os.path.join(opt.start_from, 'model_best.pdparams')
        if os.path.isfile(ckpt_path):
            print(f"Loading pretrained weights from {ckpt_path} with selective loading...")
            pretrained_state = paddle.load(ckpt_path)
            model_state = model.state_dict()
            for key in model_state.keys():
                if key in pretrained_state and pretrained_state[key].shape == model_state[key].shape:
                    model_state[key] = pretrained_state[key]
                else:
                    print(f"Skipping loading parameter: {key}, shape mismatch or not found.")
            model.set_state_dict(model_state)
        else:
            print(f"No pretrained weights found at {ckpt_path}")

    optimizer = utils.build_optimizer(model.parameters(), opt)

    epoch_done = True
    dp_lw_model.train()

    epoch_loss_sum = 0.0
    epoch_loss_count = 0

    try:
        pbar = tqdm(initial=iteration, unit='iter', desc=f"Epoch {epoch}", ncols=100)

        while True:
            if epoch_done:
                # Calculate and print average loss of previous epoch before resetting
                if epoch_loss_count > 0:
                    avg_epoch_loss = epoch_loss_sum / epoch_loss_count
                    print(f"\nEpoch {epoch} completed. Average training loss: {avg_epoch_loss:.4f}\n")

                # Reset epoch loss trackers for next epoch
                epoch_loss_sum = 0.0
                epoch_loss_count = 0

                pbar.set_description(f"Epoch {epoch}")

                if epoch > opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                optimizer.set_lr(opt.current_lr)

                if epoch > opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob

                sc_flag = (opt.self_critical_after != -1 and epoch >= opt.self_critical_after)

                epoch_done = False

            data = loader.get_batch('train')
            if iteration % acc_steps == 0:
                optimizer.clear_grad()

            start = time.time()
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            if use_gpu:
                tmp = [_ if _ is None else _.cuda() for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'],
                                   paddle.arange(0, len(data['gts'])), sc_flag)

            loss = model_out['loss'].mean()
            loss_sp = loss / acc_steps

            loss_sp.backward()
            if (iteration + 1) % acc_steps == 0:
                optimizer.step()
            train_loss = loss.item()
            end = time.time()

            # Update epoch loss trackers
            if not sc_flag:
                epoch_loss_sum += train_loss
                epoch_loss_count += 1

            if not sc_flag:
                pbar.set_postfix(train_loss=f"{train_loss:.3f}")
            else:
                pbar.set_postfix(avg_reward=f"{model_out['reward'].mean().item():.3f}")

            pbar.update(1)

            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True
                pbar.set_description(f"Epoch {epoch}")

            if iteration % opt.losses_log_every == 0:
                add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                if sc_flag:
                    add_summary_value(tb_summary_writer, 'avg_reward', model_out['reward'].mean().item(), iteration)

                loss_history[iteration] = train_loss if not sc_flag else model_out['reward'].mean().item()
                lr_history[iteration] = opt.current_lr
                ss_prob_history[iteration] = model.ss_prob

            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

            if iteration % opt.save_checkpoint_every == 0:
                eval_kwargs = {'split': 'val', 'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, lw_model.crit, loader, eval_kwargs)

                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)

                if lang_stats is not None:
                    for k, v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                json_path = os.path.join(opt.checkpoint_path, 'val_results.json')
                with open(json_path, 'w') as f_json:
                    json.dump(val_result_history, f_json, indent=4)

                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = -val_loss

                best_flag = False
                best_val_score = infos.get('best_val_score', None)
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer, append=str(iteration))
                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

            if epoch >= opt.max_epochs != -1:
                break

        # Print average loss for last epoch on normal loop exit
        if epoch_loss_count > 0:
            avg_epoch_loss = epoch_loss_sum / epoch_loss_count
            print(f"\nEpoch {epoch} completed. Average training loss: {avg_epoch_loss:.4f}\n")

        pbar.close()

    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        print(traceback.format_exc())


if __name__ == "__main__":
    opt = opts.parse_opt()
    finetune(opt)
