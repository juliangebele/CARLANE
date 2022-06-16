# DATA
data_root = './MuLane/data/'
source_train = './MuLane/splits/source_train.txt'
target_train = './MuLane/splits/target_train.txt'
source_val = './MuLane/splits/source_val.txt'
target_val = './MuLane/splits/target_val.txt'
target_test = './MuLane/splits/target_test.txt'

# TRAIN
epoch = 30
batch_size = 16
optimizer = 'Adam'    # ['SGD','Adam']
learning_rate = 1e-6
learning_rate_disc = 1e-3
weight_decay = 2.5e-5
momentum = 0.9
slope = 0.2

scheduler = 'cos'     # ['multi', 'cos']
# steps = [50,75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
num_lanes = 4
cls_num_per_lane = 56
use_aux = False
num_workers = 4

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = './ufld_adda_log/'

# FINETUNE or RESUME MODEL PATH
pretrained = './pretrained_models/mulane/ep149.pth'
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None
