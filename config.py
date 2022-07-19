import torch

# data preprocessing parameters
TEST_SPLIT = 0.20  # percentage of test samples from ALL samples 
VAL_SPLIT = 0.10  # percentage of validation samples from TRAINING samples
SEED_SPLIT = 42

# dataset parameters
IMG_FORMAT = "RGD"
CORNELL_PATH = r'C:\Users\vholo\PycharmProjects\exploring_rotated_object_detectors_final\dataset\cornell'
OCID_PATH = r'C:\Users\vholo\PycharmProjects\exploring_rotated_object_detectors_final\dataset\ocid'
DOTA_PATH = r'C:\Users\vholo\PycharmProjects\exploring_rotated_object_detectors_final\DOTA_dataset'

# training models parameters
MODELS_PATH = r'C:\Users\vholo\PycharmProjects\exploring_rotated_object_detectors_final\models'
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_DEVICE = "cpu"
EPOCHS = 5

# batch sizes
TRAIN_BS = 8
TEST_BS = 1
VAL_BS = 1
NUM_WORKERS = 4
