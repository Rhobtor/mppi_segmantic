import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

C.rgbx_dir = os.path.dirname(
    os.path.abspath(__file__)
)

# Repository root: one level above CMX/, V2_0/, or V2_2/
C.repo_dir = osp.dirname(C.rgbx_dir)
C.root_dir = C.repo_dir
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'SAFERNEet'
C.dataset_path = osp.abspath(
    os.environ.get(
        "RGBT_DATASET_ROOT",
        osp.join(C.repo_dir, "dataset")
    )
)
C.rgb_root_folder = osp.join(C.dataset_path, 'rgb')
C.rgb_format = '.png'
C.gt_root_folder = osp.join(C.dataset_path, 'masks')
C.gt_format = '.png'
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'tir')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.train_source = osp.join(C.dataset_path, 'splits', 'train.txt')
C.eval_source = osp.join(C.dataset_path, 'splits', 'test.txt')
C.is_test = False
C.num_train_imgs = 426
C.num_eval_imgs = 184
C.num_classes = 12
C.class_names =  ['First-Responder', 'Civilian', 'Vegetation', 'Road', 'Dirt-Road', 
                    'Building', 'Undergrowth', 'Civilian-Car', 'Responder-Vehicle', 
                    'Debris', 'Command-Post', 'Wall']

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 480
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'mit_b5' # Remember change the path below.
C.pretrained_model = osp.abspath(
    os.environ.get(
        "RGBT_PRETRAINED",
        osp.join(
            C.repo_dir,
            "pretrained",
            "segformer",
            "mit_b5.pth"
        )
    )
)
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 16
C.nepochs = 300
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.75, 1, 1.25] # [1]
C.eval_flip = True #False # True # 
C.eval_crop_size = [480, 480] # [height weight] #640

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.rgbx_dir))

C.output_root = osp.abspath(
    os.environ.get(
        "RGBT_OUTPUT_ROOT",
        osp.join(C.repo_dir, "outputs")
    )
)

C.log_dir = osp.join(C.output_root, "V2_2")
C.tb_dir = osp.join(C.log_dir, "tensorboard")
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.join(C.log_dir, "checkpoints")

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
