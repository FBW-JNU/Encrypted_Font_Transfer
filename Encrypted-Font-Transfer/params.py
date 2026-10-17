import os
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
FILES_DIR = PROJECT_ROOT / "files"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_IMAGES_DIR = PROJECT_ROOT / "saved_images"

###############################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME = False
DATASET = 'IAM'
if RESUME:
    EXP_NAME = "IAM-NoiseAddBinKey-FCHashNet-ClassLoss--AddThreeTimesKey-ModelKey-240515"
else:
    OPT = "JPEGCompTest"
    EXP_NAME = DATASET + "-" + OPT + '-' + time.strftime("%y%m%d", time.localtime());  # IAM-339-15-E3D3-LR0.00005-bs8


if DATASET == 'IAM':
    DATASET_PATHS = FILES_DIR / 'IAM-32.pickle'
    NUM_WRITERS = 339
if DATASET == 'CVL':
    DATASET_PATHS = FILES_DIR / 'CVL-32.pickle'
    NUM_WRITERS = 283
ENGLISH_WORDS_PATH = FILES_DIR / 'english_words.txt'

# Optional image-secret datasets. Binary-key experiments do not use these.
SECRET_PATH = os.environ.get("EFT_SECRET_PATH", "")
SECRET_VAL_PATH = os.environ.get("EFT_SECRET_VAL_PATH", "")

SECRET_TYPE = 8  # 0:image | >=1:binary(max 512)

###############################################

IMG_HEIGHT = 32
resolution = 16
batch_size = 8
NUM_EXAMPLES = 15
TN_HIDDEN_DIM = 512
TN_DROPOUT = 0.1
TN_NHEADS = 8
TN_DIM_FEEDFORWARD = 512
TN_ENC_LAYERS = 8
TN_DEC_LAYERS = 8
TN_EMB_LAYERS = 8
TN_PRE_LAYERS = 2
TN_EXT_LAYERS = 8

ALPHABET = 'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
VOCAB_SIZE = len(ALPHABET)
G_LR = 0.00005
Ext_LR = 0.000005
D_LR = 0.00005
W_LR = 0.00005
OCR_LR = 0.00005
EPOCHS = 100000
NUM_CRITIC_GOCR_TRAIN = 2
NUM_CRITIC_DOCR_TRAIN = 2
NUM_CRITIC_GWL_TRAIN = 2
NUM_CRITIC_DWL_TRAIN = 2
NUM_FID_FREQ = 100
LOSS_SCHEDULE = 10
EMB_THRES = 6


IS_SEQ = True
NUM_WORDS = 3
if not IS_SEQ: NUM_WORDS = NUM_EXAMPLES
IS_CYCLE = False
IS_KLD = False
ADD_NOISE = True
ALL_CHARS = False
SAVE_MODEL = 40
SAVE_IMAGE = 100
SAVE_MODEL_HISTORY = 400

def load_or_create_key(model_path, resume=False):
    """Load a local experiment key or create it during explicit startup."""
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    key_path = model_path / "key.pt"
    if resume:
        if not key_path.exists():
            raise FileNotFoundError(f"Cannot resume: missing {key_path}")
        return torch.load(key_path, map_location="cpu")
    if key_path.exists():
        raise FileExistsError(
            f"{key_path} already exists; set RESUME=True or choose a new EXP_NAME"
        )
    key = torch.round(torch.rand(TN_HIDDEN_DIM))
    torch.save(key, key_path)
    return key

def init_project():
    """Create runtime image directories without deleting existing output."""
    root = SAVED_IMAGES_DIR / EXP_NAME
    (root / "Real").mkdir(parents=True, exist_ok=True)
    (root / "Fake").mkdir(parents=True, exist_ok=True)
