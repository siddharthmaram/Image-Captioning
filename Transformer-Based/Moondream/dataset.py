from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

class ImageCaptioningDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_dataset("imagefolder", data_dir="path/to/dataset", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],
            "qa": [
                {
                    "question": "Describe this image in a paragraph.",
                    "answer": sample["text"],
                }
            ]
        }

datasets = {
    "train": ImageCaptioningDataset("train"),
    "test": ImageCaptioningDataset("test"),
    "validation": ImageCaptioningDataset("validation")
}