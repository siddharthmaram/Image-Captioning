from dataclasses import dataclass

@dataclass
class CFG:
    DATASET_ROOT: str = "/kaggle/input/split-10k-dataset"  # <-- EDIT if needed
    USE_FOLDER_WITH_SPACE: str = "Split Dataset"              # as in your screenshot
    USE_DESCRIPTION: str = "description_b.csv"                # non gif/webp file

    IMG_SIZE: int = 256
    CROP_SIZE: int = 224

    MIN_WORD_FREQ: int = 1
    MAX_LEN: int = 200

    EMBED_DIM: int = 256
    HIDDEN_DIM: int = 512
    ATT_DIM: int = 512

    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LR: float = 5e-5
    CLIP_GRAD: float = 5.0

    BACKBONE: str = "resnet101"  # resnet50 or resnet101
    FREEZE_ENCODER: bool = False

    BEAM_SIZE: int = 3
    NUM_WORKERS: int = 2

    OUT_DIR: str = "sat_csv_biology"

cfg = CFG()