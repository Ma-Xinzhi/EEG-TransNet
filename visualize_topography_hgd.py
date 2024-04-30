"""
Class activation topography (CAT) for EEG model visualization, combining class activity map and topography
Code: Class activation map (CAM) and then CAT

refer to high-star repo on github: 
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam

Salute every open-source researcher and developer!
"""


import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from model.TransNet import TransNet

import matplotlib.pyplot as plt
from torch.backends import cudnn
from utils import GradCAM
import yaml
from data.dataset import eegDataset

subId = 6

model_path = 'output/process_hgd'
specific_folder = '2024-02-23--21-52'
# specific_folder = '2024-02-23--16-19'

target_category = 3 # set the class (class activation mapping)

# ! A crucial step for adaptation on Transformer
# reshape_transform  b 64 1 32 -> b 32 1 64
def reshape_transform(tensor):
    result = rearrange(tensor, 'b w h e -> b e h w')
    # result = rearrange(tensor, 'b w h e -> b e h w')
    return result

def main(config):
    dataset_path = config['data_path']    

    modelParam = os.path.join(model_path, 'sub'+str(subId), specific_folder, 'model.pth')

    data_path = os.path.join(dataset_path, str(subId) + '_data.npy')
    label_path = os.path.join(dataset_path, str(subId) + '_label.npy')

    data = np.load(data_path)
    labels = np.load(label_path).squeeze()

    target_index = np.where(labels == target_category)
    target_data = data[target_index]
    print(target_data.shape)

    netArgs = config['network_args']
    model = eval(config['network'])(**netArgs)
    model.load_state_dict(torch.load(modelParam, map_location='cpu'))
    # print(model.state_dict())
    # print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    target_layers = [model.conv_encoder]  # set the target layer 
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)

    # TODO: Class Activation Topography (proposed in the paper)
    import mne
    from matplotlib import mlab as mlab

    # biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    biosemi_montage = mne.channels.make_standard_montage('standard_1005')

    index = [27, 29, 31, 33, 39, 43, 49, 51, 53, 55, 28, 30, 32, 38, 40, 42, 44, 50, 52, 54, 108, 109, 
    112, 113, 118, 119, 122, 123, 128, 129, 132, 133, 138, 139, 142, 143, 110, 111, 120, 121, 130, 131, 
    140, 141]  # for hgd
    biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
    biosemi_montage.dig = [biosemi_montage.dig[i+3] for i in index]
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250.0, ch_types='eeg')

    all_cam = []
    # this loop is used to obtain the cam of each trial/sample
    for i in range(len(target_data)):
        test = torch.as_tensor(target_data[i:i+1], dtype=torch.float32)
        test = torch.autograd.Variable(test, requires_grad=True)

        grayscale_cam = cam(input_tensor=test, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        all_cam.append(grayscale_cam)

    # the mean of all target data
    mean_target_data = np.mean(target_data, axis=0) # N*C*T -> C*T
    norm_mean_target_data = (mean_target_data - np.mean(mean_target_data)) / np.std(mean_target_data)
    vis_target_data = np.mean(norm_mean_target_data, axis=1) # C*T -> C

    # the mean of all cam
    mean_all_cam = np.mean(all_cam, axis=0) # N*1*T -> 1*T

    # apply cam on the input data
    hyb_all = mean_target_data * mean_all_cam
    hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
    vis_hyb_all = np.mean(hyb_all, axis=1)

    evoked = mne.EvokedArray(mean_target_data, info)
    evoked.set_montage(biosemi_montage)

    fig, [ax1, ax2] = plt.subplots(nrows=2)

    plt.subplot(211)
    im, cn2 = mne.viz.plot_topomap(vis_target_data, evoked.info, show=False, axes=ax1, res=1200)

    plt.subplot(212)
    im, cn3 = mne.viz.plot_topomap(vis_hyb_all, evoked.info, show=False, axes=ax2, res=1200)

    # manually fiddle the position of colorbar
    # ax_x_start = 0.95
    # ax_x_width = 0.04
    # ax_y_start = 0.1
    # ax_y_height = 0.9
    # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im)
    clb.set_ticks([-1.0,0,1.0])
    # clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar

    plt.savefig('./feet_topology_hgd.png', bbox_inches='tight', dpi=1000)
    # plt.show()

    plt.close()


if __name__ == '__main__':
    configFile = os.path.join(model_path, 'sub'+str(subId), specific_folder, 'config.yaml')
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)