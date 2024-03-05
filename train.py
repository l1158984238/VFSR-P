import argparse
from pathlib import Path
from utils.util import cfg,cfg_from_yaml_file,build_savePath,seed_torch
from utils.Dataloaders import createTrainDataset,createValDataset
import os
from torch.utils.data import DataLoader


def parse_config():
    #grad,get_Jacb,,Resize,stn
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='config.yaml', help='specify the config.yaml for training')

    #parser.add_argument('--model', default='PDEnetv4', help='选择模型 VxmNet,PDEnet,PDEnetv2,PDEnet_out5')
    parser.add_argument('--epochs', default=500, help='train epochs')
    parser.add_argument('--valEpoch', default=10, help='')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='test1', help='save to project/name')
    parser.add_argument('--tensorboard', type=bool, default=True, help='strat tensorboard')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg


if __name__ == '__main__':
    opt, cfg = parse_config()
    opt.Root_savepath, opt.model_dir, opt.reg_dir = \
        build_savePath(os.path.join(os.getcwd(), opt.project), opt.name)
    seed_torch(818)

    print("***********- ***********- READ DATA and processing-*************")
    # if 'lits' in cfg['data']['data_root']:
    #     train_dataset = createLiverDataset(cfg['data']['data_root'],'lits_train.h5',mode='train')
    #     val_dataset=createLiverDataset(cfg['data']['data_root'],'lits_val.h5',mode='val')
    train_dataset = createTrainDataset(cfg['data']['data_root'], 'train.txt')
    val_dataset = createValDataset(cfg['data']['data_root'], 'pairs_val.csv')
    print('training dataset long: {}'.format(len(train_dataset)))
    print('valid dataset long: {}'.format(len(val_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4,
                                  pin_memory=True)
    test_dataloader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=4,
                                 pin_memory=True)
    print("***********- ***********loading model ***********- ***********")