import argparse
from pathlib import Path
from utils.util import cfg,cfg_from_yaml_file



def parse_config():
    #grad,get_Jacb,,Resize,stn
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='specify the config.yaml for training')

    #parser.add_argument('--model', default='PDEnetv4', help='选择模型 VxmNet,PDEnet,PDEnetv2,PDEnet_out5')
    parser.add_argument('--epochs', default=500, help='训练轮数')
    parser.add_argument('--valEpoch', default=10, help='')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='liver_SbySnet3', help='save to project/name')
    parser.add_argument('--tensorboard', type=bool, default=True, help='strat tensorboard')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg


if __name__ == '__main__':
    opt, cfg = parse_config()