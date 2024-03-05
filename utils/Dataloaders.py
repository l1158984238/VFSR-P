import torch.utils.data as Data
import os
import torch
import random
import numpy as np
import nibabel as nib
import scipy
import pandas as pd

from utils.util import read_file_list



def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices

def resize(array, factor,resize_shape,batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if factor=='auto':
            newsize=np.array(resize_shape,float)
            oldsize=np.array(array.shape[:-1])
            factor=newsize/oldsize
        if not batch_axis:
            if len(factor)==1:
                dim_factors = [factor for _ in array.shape[:-1]] + [1]
            else:
                assert len(list(factor))==len(array.shape[:-1])
                dim_factors = list(factor) + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)

def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False,
    resize_shape=None
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        img = nib.load(filename)
        vol = img.get_fdata().squeeze()
        affine = img.affine

    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)
    if add_feat_axis:
        vol = vol[..., np.newaxis]
    if resize_factor != 1:
        vol = resize(vol, resize_factor,resize_shape=resize_shape)
    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol

class createTrainDataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_root, imgTxT, img_suffix='aligned_norm.nii.gz'):
        'Initialization'
        super(createTrainDataset, self).__init__()
        img_path = os.path.join(data_root, imgTxT)
        self.imgs_path = read_file_list(img_path, prefix=data_root+'/',suffix='/'+img_suffix)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_path)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A = load_volfile(self.imgs_path[index])
        img_B = load_volfile(random.choice(self.imgs_path))
        img_A = img_A[np.newaxis, ...]
        img_B = img_B[np.newaxis, ...]
        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()

class createValDataset(Data.Dataset):
    def __init__(self, data_path, pairs,reg_flag=False,prefix='/OASIS_OAS1_0',suffix='_MR1'):
        'Initialization'
        super(createValDataset, self).__init__()
        self.name=pd.read_csv(os.path.join(data_path,pairs)).values
        self.img_list =os.listdir(data_path)
        self.img_list = [name for name in self.img_list if 'OASIS' in name]
        self.name = [[data_path+prefix+str(name[0])+suffix,data_path+prefix+str(name[1])+suffix] for name in self.name]
        self.reg_flag=reg_flag

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.name)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A = load_volfile(self.name[index][0]+'aligned_norm.nii.gz')
        img_A = img_A[np.newaxis, ...]
        label_A = load_volfile(self.name[index][0]+'aligned_seg35.nii.gz')
        label_A = label_A[np.newaxis, ...]

        img_B = load_volfile(self.name[index][1]+'aligned_norm.nii.gz')
        img_B = img_B[np.newaxis, ...]
        label_B = load_volfile(self.name[index][1]+'aligned_seg35.nii.gz')
        label_B = label_B[np.newaxis, ...]
        # if self.reg_flag==False:
        #     return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), \
        #             torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()
        # else:
        name_A=self.name[index][0].split('/')[-2]
        name_B = self.name[index][1].split('/')[-2]
        return  [torch.from_numpy(img_A).float(),torch.from_numpy(label_A).float(),name_A],\
                 [torch.from_numpy(img_B).float(),torch.from_numpy(label_B).float(),name_B]