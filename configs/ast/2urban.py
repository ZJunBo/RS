from configs.ToURBAN import SOURCE_DATA_CONFIG,TARGET_DATA_CONFIG, EVAL_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET
MODEL = 'ResNet'

IGNORE_LABEL = -1
MOMENTUM = 0.9
NUM_CLASSES = 7
SNAPSHOT_DIR = './log/ast/urban'

#---------------------------adversariy training-----------------------
#Hyper Paramters
AT_WEIGHT_DECAY = 0.0005
AT_LEARNING_RATE = 5e-3
AT_LEARNING_RATE_D = 1e-4
# AT_NUM_STEPS = 15000
AT_traing_steps = 1000  # Use damping instead of early stopping，10000
AT_ITER_SIZE=1
AT_PREHEAT_STEPS = 0 #int(NUM_STEPS / 20)
AT_POWER = 0.9
AT_LAMBDA_SEG = 0.1
AT_LAMBDA_ADV_TARGET1 = 0.001
AT_EVAL_EVERY = 1000 # 2000
#--------------------------self-training--------------------
# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 1e-2
NUM_STEPS_MAX = 14001  # for learning rate poly,原始16000
# NUM_STEPS_STOP = 10001  # Use damping instead of early stopping
# FIRST_STAGE_STEP = 1000  # for first stage
PREHEAT_STEPS = int(NUM_STEPS_MAX / 20)  # for warm-up
POWER = 0.9  # lr poly power
EVAL_EVERY = 500
GENERATE_PSEDO_EVERY = 500
MULTI_LAYER = True
IGNORE_BG = True
PSEUDO_SELECT = True


TARGET_SET = TARGET_SET
SOURCE_DATA_CONFIG = SOURCE_DATA_CONFIG
TARGET_DATA_CONFIG = TARGET_DATA_CONFIG
EVAL_DATA_CONFIG = EVAL_DATA_CONFIG
TEST_DATA_CONFIG = TEST_DATA_CONFIG
