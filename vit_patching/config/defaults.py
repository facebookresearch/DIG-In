# Written by Pengchuan Zhang, penzhan@microsoft.com
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image size during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# training data augmentation
_C.INPUT = CN()
_C.INPUT.NORMALIZE = True
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]
_C.INPUT.IMAGE_SIZE = 224 # 299 for inception_v3
_C.INPUT.CROP_PCT = 0.875 # 0.816 for inception_v3
_C.INPUT.INTERPOLATION = 2

_C.AMP = CN()
_C.AMP.ENABLED = False
_C.AMP.MEMORY_FORMAT = 'nchw'

# data augmentation
_C.AUG = CN()
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0/4.0, 4.0/3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.REPEATED_AUG = False
_C.AUG.TIMM_AUG = CN(new_allowed=True)
_C.AUG.TIMM_AUG.USE_TRANSFORM = False

_C.DATA = CN()
# choices=['toy_ill', 'toy_well', 'mnist', 'cifar', 'cifar100', 'imagenet', 'wikitext-2', 'celeba']
_C.DATA.TRAIN = ('cifar',)
_C.DATA.TEST = ('cifar',)
_C.DATA.NUM_CLASSES = 10
_C.DATA.TARGETMAP = ''
# path to datasets, default=os.getenv('PT_DATA_DIR', './datasets')
_C.DATA.PATH = "./datasets"
# path to other necessary data like checkpoints other than datasets.
# default=os.getenv('PT_DATA_DIR', 'data')
_C.DATA.DATA_DIR = "./data"

# choices=['mse', 'xentropy', 'bce'], msr for least regression or xentropy for classification
_C.LOSS = CN()
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.LOSS = 'xentropy'
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.NORMALIZE = True
_C.LOSS.FOCAL.ALPHA = 1.0
_C.LOSS.FOCAL.GAMMA = 0.5


# dataloader
_C.DATALOADER = CN()
# batch size
_C.DATALOADER.BSZ = 128
# samples are drawn with replacement if yes
_C.DATALOADER.RE = 'no'
# number of data loading workers
_C.DATALOADER.WORKERS = 0

# optimizer
_C.OPTIM = CN()
# optimizer, choices=['zero_bat', 'zero_seq', 'yaida_diag', 'yaida_seq',
#  'yaida_ratio', 'lxz', 'baydin',
#  'pflug_bat', 'pflug_seq', 'sgd', 'qhm', 'adam',
#  'statsnon', 'pflug_wilcox',
#  'sasa_xd_seq', 'sasa_xd', 'sgd_sls', 'salsa', 'ssls', 'salsa_new', 'slope'],
#  default='sgd')
_C.OPTIM.OPT = 'qhm'
# effective learning rate
_C.OPTIM.LR = 1.0
# effective momentum value
_C.OPTIM.MOM = 0.9
# nu value for qhm
_C.OPTIM.NU = 1.0
# weight decay lambda
_C.OPTIM.WD = 5e-4
# Number of Epochs
_C.OPTIM.EPOCHS = 150
# Warm up: epochs of qhm before switching to sasa/salsa
_C.OPTIM.WARMUP = 0
# Drop frequency and factor for all methods
_C.OPTIM.DROP_FREQ = 50
_C.OPTIM.DROP_FACTOR = 10.0
# use validation dataset to adapt learning rate
_C.OPTIM.VAL = 0
# Hypergradient adaption parameters
_C.OPTIM.INC_REG = 1.0
_C.OPTIM.DEC_REG = -1.0
_C.OPTIM.INC_COS = 0.1
_C.OPTIM.DEC_COS = -0.1
_C.OPTIM.INC_FACTOR = 2.0
_C.OPTIM.DEC_FACTOR = 4.0
_C.OPTIM.TEST_FREQ = 1000

# ADAM's default parameters
_C.OPTIM.ADAM = CN()
_C.OPTIM.ADAM.BETA1 = 0.9
_C.OPTIM.ADAM.BETA2 = 0.999
_C.OPTIM.ADAM.EPS = 1e-8

# SASA's default parameters
_C.OPTIM.SASA = CN()
# leaky bucket ratio
_C.OPTIM.SASA.LEAK_RATIO = 4
# significance level
_C.OPTIM.SASA.SIGNIF = 0.01
# Minimal sample size for statistical test
_C.OPTIM.SASA.N = 1000
# delta in equivalence test
_C.OPTIM.SASA.DELTA = 0.02
# method to calculate variance, choices=['iid', 'bm', 'olbm']
# reused for the mode of slope test, choices=['linear', 'log']
_C.OPTIM.SASA.MODE = 'olbm'
# log frequency (iterations) for statistics, 0 means logging at statistical tests
_C.OPTIM.SASA.LOGSTATS = 0
# number of statistical tests in one epoch
_C.OPTIM.SASA.TESTS_PER_EPOCH = 1
_C.OPTIM.SASA.TESTFREQ = 5005

# Line search
_C.OPTIM.LS = CN()
# smoothing factor
_C.OPTIM.LS.GAMMA = 0.01
# Sufficient decreasing constant
_C.OPTIM.LS.SDC = 0.1
# Increase factor
_C.OPTIM.LS.INC = 2.0
# Decrease factor
_C.OPTIM.LS.DEC = 0.5
# Maximal backtracking steps
_C.OPTIM.LS.MAX = 10
# Ignore the backtracking that reaches _C.OPTIM.LS.MAX
_C.OPTIM.LS.IGN = 0
# function call in evaluation mode for line search
_C.OPTIM.LS.EVAL = 1
# line search vector 'g'radient or 'd'irection to update
_C.OPTIM.LS.DIR = 'g'
# binary: whether or not to use cosine in line search
_C.OPTIM.LS.COS = 0
# stochastic moving average line search parameters
_C.OPTIM.LS.WARMUP = 1000
_C.OPTIM.LS.LRATIO = 2
_C.OPTIM.LS.RELRED = 0.01

# SALSA
_C.OPTIM.SALSA = CN()
# binary: automatic switch between SSLS ans SASA
_C.OPTIM.SALSA.AUTOSWITCH = 0

# Variance Reduction 
_C.OPTIM.VR = CN()
_C.OPTIM.VR.PERIOD = 100
_C.OPTIM.VR.BATCH = 100
_C.OPTIM.VR.l1WEIGHT = 0.0
_C.OPTIM.VR.l1WARMUP = 0

# LR scheduler
_C.SOLVER = CN()
_C.SOLVER.LR_POLICY = '' # multistep, cosine, linear
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_EPOCHS = 5.0
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.MIN_LR = 0.0 # MAX_LR is _C.OPTIM.LR
_C.SOLVER.DETECT_ANOMALY = False
_C.SOLVER.EPOCH_BASED_SCHEDULE = False
_C.SOLVER.USE_LARC = False

# models
_C.MODEL = CN()
# choices=model_names + my_model_names + seq_model_names,
#     help='model architecture: ' +
#          ' | '.join(model_names + my_model_names + seq_model_names) +
#          ' (default: resnet18)')
_C.MODEL.ARCH = 'resnet18'
# nonlinearity, choices=['celu', 'softplus', 'gelu']
_C.MODEL.NONLINEARITY = 'celu'
# relative path of checkpoint relative to DATA_DIR
_C.MODEL.MODEL_PATH = ""
# use pre-trained model from torchvision
_C.MODEL.PRETRAINED = False
_C.MODEL.FREEZE_CONV_BODY_AT = -1

_C.MODEL.RNN = CN()
# size of word embeddings
_C.MODEL.RNN.EMSIZE = 1500
# number of hidden units per layer
_C.MODEL.RNN.NHID = 1500
# number of layers
_C.MODEL.RNN.NLAYERS = 2
# sequence length when back-propogation through time
_C.MODEL.RNN.BPTT = 35
# dropout applied to layers (0 = no dropout)
_C.MODEL.RNN.DROPOUT = 0.65
# tie the word embedding and softmax weights
_C.MODEL.RNN.TIED = True
# gradient clipping
_C.MODEL.RNN.CLIP = 0.25
# whether we randomly shuttle the sequential data or not
_C.MODEL.RNN.SHUFFLE = 0
# set 1 to make the initial hidden state 0!
_C.MODEL.RNN.INIT0 = 0
# the number of heads in the encoder/decoder of the transformer model
_C.MODEL.RNN.NHEAD = 2

_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.DROP = 0.0
_C.MODEL.TRANSFORMER.DROP_PATH = 0.1
_C.MODEL.TRANSFORMER.NORM_EMBED = False

# ---------------------------------------------------------------------------- #
# Adversarial training options
# ---------------------------------------------------------------------------- #
_C.ATTACK = CN()
_C.ATTACK.ENABLED = False
_C.ATTACK.CONSTRAINT = '2'
_C.ATTACK.EPS = 3.
_C.ATTACK.ITERATIONS = 3
_C.ATTACK.STEP_SIZE = 1.5
_C.ATTACK.RANDOM_START = False
_C.ATTACK.RANDOM_RESTARTS = False
_C.ATTACK.USE_BEST = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# default=os.getenv('PT_OUTPUT_DIR', '/tmp')
_C.OUTPUT_DIR = "/tmp"
# default=os.getenv('PHILLY_LOG_DIRECTORY', None)
_C.BACKUP_LOG_DIR = ""
_C.LOG_FREQ = 10
# evaluate model on validation set
_C.EVALUATE = False
_C.OUTPUT_PERCLASS_ACC = False
# Only save the last checkpoint in the checkpointer
_C.ONLY_SAVE_LAST = 0

_C.DISTRIBUTED_BACKEND = "nccl"  # could be "nccl", "gloo" or "mpi"
# whether to use CPU to do gather of predictions. Note that this requires
# running with "gloo" (or "mpi") distributed backend
_C.GATHER_ON_CPU = False