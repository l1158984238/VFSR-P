import yaml
from easydict import EasyDict
from pathlib import Path
import os
import shutil
import torch
import random
import numpy as np

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda .manual_seed_all(seed)# if you are using multi-GPU
    torch. backends.cudnn.benchmark= False
    torch.backends.cudnn.deterministic = True

def build_savePath(savepath,name,model_dir='models',reg_dir='registration',):
    #Generate the file storage location produced during model training
    mkdir(savepath)
    exSavePath = os.path.join(savepath, name)
    flag=os.path.exists(exSavePath)

    while flag:
        exSavePath=os.path.join(savepath,name)
        modelSavePath=os.path.join(exSavePath,model_dir)
        if not os.path.exists(modelSavePath):
            flag = False
            continue
        if len(os.listdir(modelSavePath))!=0:
            subname=name.split('_')
            name=subname[0]+'_'+str(int(subname[1])+1)
        else:
            flag=False

    modelSavePath = os.path.join(exSavePath, model_dir)
    registrationSavePath = os.path.join(exSavePath, reg_dir)
    # warpedSavePath = os.path.join(exSavePath, warped_dir)

    mkdir(exSavePath)
    mkdir(modelSavePath)
    mkdir(registrationSavePath)
    # mkdir(warpedSavePath)

    suffix='.yaml'
    files = os.listdir(exSavePath)
    for k in range(len(files)):
        files[k]=os.path.splitext(files[k])[1]
    flag=False if suffix in files else True
    if flag:
        shutil.copyfile('./config.yaml',os.path.join(exSavePath,'config.yaml'))

    return exSavePath,modelSavePath,registrationSavePath

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0