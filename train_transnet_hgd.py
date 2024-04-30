import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from visdom import Visdom
from model.TransNet import TransNet
from model.baseModel import baseModel
import time
import os
import yaml
from data.data_utils import *
from data.dataset import eegDataset
from utils import *
torch.set_num_threads(15)
def setRandom(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dictToYaml(filePath, dictToWrite):
    with open(filePath, 'w', encoding='utf-8') as f:
        yaml.dump(dictToWrite, f, allow_unicode=True)
    f.close()

def main(config):
    data_path = config['data_path']
    out_folder = config['out_folder']
    random_folder = str(time.strftime('%Y-%m-%d--%H-%M', time.localtime()))
    
    lr = config['lr']

    for subId in range(1,15):
        data_file = str(subId) + '_data.npy'
        label_file = str(subId) + '_label.npy'

        out_path = os.path.join(out_folder, config['network'], 'sub'+str(subId), random_folder)
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        print("Results will be saved in folder: " + out_path)

        dictToYaml(os.path.join(out_path, 'config.yaml'), config)

        setRandom(config['random_seed'])

        train_data, train_labels = load_HGD_data(os.path.join(data_path, 'train'), data_file, label_file)
        test_data, test_labels = load_HGD_data(os.path.join(data_path, 'test'), data_file, label_file)

        train_dataset = eegDataset(train_data, train_labels)
        test_dataset = eegDataset(test_data, test_labels)

        net_args = config['network_args']
        net = eval(config['network'])(**net_args)
        print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        model = baseModel(net, config, optimizer, loss_func, result_savepath=out_path)

        model.train_test(train_dataset, test_dataset)

if __name__ == '__main__':
    configFile = 'config/hgd_transnet.yaml'
    file = open(configFile, 'r', encoding='utf-8')
    config = yaml.full_load(file)
    file.close()
    main(config)

