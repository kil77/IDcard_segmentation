import sys
import os

import argparse
import numpy as np

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

BATCH_SIZE = 4
BN_WEIGHT_DECAY = 0.9997
CKPT = 14000
DATA_DIRECTORY = './datasets/IDImage'
DATA_NAME = 'IDImage'
FREEZEN_BN = False
IGNORE_LABEL = 255
IMAGENET = './pretrained_models'
IMAGE_PATH = './datasets/IDImage/JPGImage'
INPUT_SIZE = 512
IS_TRAINING = True
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_GPUS = 1
NUM_LAYERS = 50
NUM_STEPS = 30000
NUM_TRAIN = 515
NUM_VAL = 125
OUTPUT_PATH = './datasets/IDImage/inference_out'
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = None
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots'
SPLIT_NAME = 'train'
WEIGHT_DECAY = 1e-4

parser = argparse.ArgumentParser(description="DeepLabV3")
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Number of images sent to the network in one step.")
parser.add_argument("--bn-weight-decay", type=float, default=BN_WEIGHT_DECAY,
                    help="Regularisation parameter for batch norm.")
parser.add_argument("--ckpt", type=int, default=CKPT, 
                    help="Checkpoint to restore.")
parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                    help="Path to the directory containing the PASCAL VOC dataset.")
parser.add_argument("--data-name", type=str, default=DATA_NAME,
                    help="Name of the dataset.")
parser.add_argument("--freeze-bn", type = bool, default= FREEZEN_BN,
                    help="Whether to freeze batch norm params.")
parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                    help="The index of the label to ignore during the training.")
parser.add_argument("--imagenet", type=str, default=IMAGENET,
                    help="Path to ImageNet pretrained weights.")
parser.add_argument("--image-path", type=str, default=IMAGE_PATH,
                    help="Path to JPGImages.")
parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                    help="height and width of images.")
parser.add_argument("--is-training", type=bool, default=IS_TRAINING,
                    help="the flag of whether is training now")
parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--momentum", type=float, default=MOMENTUM,
                    help="Momentum component of the optimiser.")
parser.add_argument("--not-restore-last", action="store_true",
                    help="Whether to not restore last (FC) layers.")
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                    help="Number of classes to predict (including background).")
parser.add_argument("--num-gpus", type=int, default=NUM_GPUS,
                    help="Number of GPUs to use.")
parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                    help="Number of layes in ResNet).")
parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                    help="Number of training steps.")
parser.add_argument("--output-path", type=str, default=OUTPUT_PATH,
                    help="the output path during val")
parser.add_argument("--power", type=float, default=POWER,
                    help="Decay parameter to compute the learning rate.")
parser.add_argument("--random-mirror", action="store_true",
                    help="Whether to randomly mirror the inputs during the training.")
parser.add_argument("--random-scale", action="store_true",
                    help="Whether to randomly scale the inputs during the training.")
parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                    help="Random seed to have reproducible results.")
parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                    help="Where restore model parameters from.")
parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                    help="How many images to save.")
parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                    help="Save summaries and checkpoint every often.")
parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                    help="Where to save snapshots of the model.")
parser.add_argument("--split-name", type=str, default=SPLIT_NAME,
                    help="Split name.")
parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                    help="Regularisation parameter for L2-loss.")

args = parser.parse_args()