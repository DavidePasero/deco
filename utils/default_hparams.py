from yacs.config import CfgNode as CN

# Set default hparams to construct new default config
# Make sure the defaults are same as in parser
hparams = CN()

# General settings
hparams.EXP_NAME = 'default'
hparams.PROJECT_NAME = 'default'
hparams.OUTPUT_DIR = 'deco_results/'
hparams.CONDOR_DIR = '/is/cluster/work/achatterjee/condor/rich/'
hparams.LOGDIR = ''

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 4
hparams.DATASET.NORMALIZE_IMAGES = True

# Optimizer hparams
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 5e-5
hparams.OPTIMIZER.NUM_UPDATE_LR = 10

# Training hparams
hparams.TRAINING = CN()
hparams.TRAINING.MODEL_TYPE = 'deco'
hparams.TRAINING.ENCODER = 'dinov2'
hparams.TRAINING.CLASSIFIER_TYPE = 'shared'
hparams.TRAINING.CONTEXT = True
hparams.TRAINING.NUM_EPOCHS = 50
hparams.TRAINING.SUMMARY_STEPS = 100
hparams.TRAINING.CHECKPOINT_EPOCHS = 5
hparams.TRAINING.NUM_EARLY_STOP = 10
hparams.TRAINING.DATASETS = ['rich']
hparams.TRAINING.DATASET_MIX_PDF = ['1.']
hparams.TRAINING.DATASET_ROOT_PATH = '/is/cluster/work/achatterjee/rich/npzs'
hparams.TRAINING.BEST_MODEL_PATH = '/is/cluster/work/achatterjee/weights/rich/exp/rich_exp.pth'
hparams.TRAINING.LOSS_WEIGHTS = 1.
hparams.TRAINING.PAL_LOSS_WEIGHTS = 1.
hparams.TRAINING.OBJECT_CLASSIFIER = True
hparams.TRAINING.TRAIN_BACKBONE = False
hparams.TRAINING.NUM_ENCODER = 1
hparams.TRAINING.USE_VLM = False

# Training hparams
hparams.VALIDATION = CN()
hparams.VALIDATION.SUMMARY_STEPS = 100
hparams.VALIDATION.DATASETS = ['rich']
hparams.VALIDATION.MAIN_DATASET = 'rich'
