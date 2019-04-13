import os

#################
# preprocess
################
DATA_ROOT = 'D:\\dataset'
PASCAL_DIR = os.path.join(DATA_ROOT, 'VOCdevkit')

ROOT_DIR = 'D:\\yolo'
OUTPUT_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'voc_train_2012.txt')
NUMPY_FILE = os.path.join(OUTPUT_DIR, 'train_2012.npz')
YEAR = 'VOC2012'
VOC_CLASS_FILE = os.path.join(OUTPUT_DIR, 'voc_classes.txt')
VOC_CLASS = open(VOC_CLASS_FILE).read().split('\n')

LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
WEIGHTS = 'YOLO_small.ckpt'


EPOCHS = 100
KEEP_PROB = 0.5
IMAGE_SIZE = 448
CELL_SIZE = 7
BOXES_PER_CELL = 2

ALPHA = 0.1
OBJECT_SCALE = 2
NOOBJECT_SCALE = 1
CLASS_SCALE = 1.0
COORD_SCALE = 5.0

LEARNING_RATE = 0.0001
DECAY_STEPS = 300
DECAY_RATE = 0.9
STAIRCASE = True

BATCH_SIZE = 32
MAX_STEP = 30000
SUMMARY_STEP = 100
SAVE_STEP = 50

THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
