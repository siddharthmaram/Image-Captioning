from dataclasses import dataclass

@dataclass
class CFG:
    IMG_SIZE: int = 256
    CROP_SIZE: int = 224

    MIN_WORD_FREQ: int = 1
    MAX_LEN: int = 200

    EMBED_DIM: int = 256
    HIDDEN_DIM: int = 512
    ATT_DIM: int = 512

    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LR: float = 5e-5
    CLIP_GRAD: float = 5.0

    BACKBONE: str = "resnet101"  # resnet50 or resnet101
    FREEZE_ENCODER: bool = False

    BEAM_SIZE: int = 3
    NUM_WORKERS: int = 2

    OUT_DIR: str = "sat_csv_biology"

cfg = CFG()