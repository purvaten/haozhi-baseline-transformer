# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

_C.DATA_ROOT = './data/coco'
_C.DATASET_ABS = 'Phys'
_C.BASE = ''
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode()
_C.INPUT.PRELOAD_TO_MEMORY = False
_C.INPUT.IMAGE_MEAN = [0, 0, 0]
_C.INPUT.IMAGE_STD = [1.0, 1.0, 1.0]
_C.INPUT.PHYRE_USE_EMBEDDING = False
_C.INPUT.IMAGE_CHANNEL = 3
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.VAL_INTERVAL = 16000
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WARMUP_ITERS = -1
_C.SOLVER.LR_MILESTONES = [12000000, 18000000]
_C.SOLVER.MAX_ITERS = 20000000
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.SCHEDULER = 'step'

# ---------------------------------------------------------------------------- #
# Intuitive Physics models
# ---------------------------------------------------------------------------- #
_C.RIN = CfgNode()
_C.RIN.ARCH = ''
_C.RIN.BACKBONE = ''
_C.RIN.HORIZONTAL_FLIP = True
_C.RIN.VERTICAL_FLIP = True
_C.RIN.OVER_SAMPLE_COLLISION = False
_C.RIN.COLLISION_ENV_MULTIPLY = 1
_C.RIN.COLLISION_OBJ_MULTIPLY = 10
_C.RIN.COLLISION_CONSTRAINT = 4
# prediction setting
_C.RIN.INPUT_SIZE = 4
_C.RIN.PRED_SIZE_TRAIN = 20
_C.RIN.PRED_SIZE_TEST = 40
# input for mixed dataset
_C.RIN.IMAGE_EXT = '.jpg'
_C.RIN.INPUT_HEIGHT = 360
_C.RIN.INPUT_WIDTH = 640
# training setting
_C.RIN.NUM_OBJS = 3
_C.RIN.POSITION_LOSS_WEIGHT = 1
_C.RIN.MASK_LOSS_WEIGHT = 0.0
_C.RIN.MASK_SIZE = 14
# additional input
_C.RIN.IMAGE_FEATURE = True
# ROI POOLING
_C.RIN.ROI_POOL_SIZE = 1
_C.RIN.ROI_POOL_SPATIAL_SCALE = 0.25
_C.RIN.ROI_POOL_SAMPLE_R = 1
_C.RIN.COOR_FEATURE = False
_C.RIN.COOR_FEATURE_EMBEDDING = False
_C.RIN.COOR_FEATURE_SINUSOID = False
_C.RIN.COOR_FEATURE_SINUSOID_BASE = 100
_C.RIN.IN_CONDITION = False
_C.RIN.IN_CONDITION_R = 1.5
# parameter
_C.RIN.VE_FEAT_DIM = 32
_C.RIN.IN_FEAT_DIM = 64
# roi masking
_C.RIN.ROI_MASKING = False
_C.RIN.ROI_CROPPING = False
_C.RIN.USE_VIN_FEAT = False
_C.RIN.ROI_CROPPING_R = 4
_C.RIN.VAE = False
_C.RIN.VAE_KL_LOSS_WEIGHT = 0.001
# sequence cls
_C.RIN.SEQ_CLS_LOSS_WEIGHT = 0.0
# DISCOUNT
_C.RIN.DISCOUNT_TAU = 0.01
# EXPLORE ARCH
_C.RIN.N_EXTRA_ROI_F = 0
_C.RIN.N_EXTRA_PRED_F = 0
_C.RIN.N_EXTRA_SELFD_F = 0
_C.RIN.N_EXTRA_RELD_F = 0
_C.RIN.N_EXTRA_AFFECTOR_F = 0
_C.RIN.N_EXTRA_AGGREGATOR_F = 0
_C.RIN.EXTRA_F_KERNEL = 3
_C.RIN.EXTRA_F_PADDING = 1

# ---------------------------------------------------------------------------- #
# PHYRE Classifier Model
# ---------------------------------------------------------------------------- #
_C.PCLS = CfgNode()
_C.PCLS.ARCH = 'resnet18'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = './outputs/default'
