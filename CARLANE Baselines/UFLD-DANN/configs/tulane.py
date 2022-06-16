# DATA
data_root = 'D:/TuLane/data/'
source_train = 'D:/TuLane/splits/source_train.txt'
target_train = 'D:/TuLane/splits/target_train.txt'
source_val = 'D:/TuLane/splits/source_val.txt'
target_val = 'D:/TuLane/splits/target_val.txt'
target_test = 'D:/TuLane/splits/target_test.txt'

# TRAIN
epoch = 30
batch_size = 4
optimizer = 'Adam'    # ['SGD','Adam']
learning_rate = 1e-5
learning_rate_disc = 1e-3
weight_decay = 1e-4
momentum = 0.9

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
use_aux = True
num_workers = 4

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0
domain_loss_w = 1.0

# EXP
note = ''

log_path = 'D:/ufld_dann_log/'

# FINETUNE or RESUME MODEL PATH
finetune = None  # './pretrained_models/tulane/ep149.pth'
resume = None

# TEST
test_model = None
test_work_dir = None
