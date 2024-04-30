import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import os
import copy
import itertools
from mpl_toolkits.axes_grid1 import host_subplot
from datetime import datetime
import random

class baseModel():
    def __init__(self, net, config, optimizer, loss_func, scheduler=None, result_savepath=None):        
        self.batchsize = config['batch_size']
        self.epochs = config['epochs']
        self.preferred_device = config['preferred_device']

        # for data augmentation
        self.num_classes = config['num_classes']
        self.num_segs = config['num_segs']

        self.device = None
        self.set_device(config['nGPU'])
        self.net = net.to(self.device)

        # for training
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler

        # save result
        self.result_savepath = result_savepath
        self.log_write = None
        if self.result_savepath is not None:
            self.log_write = open(os.path.join(self.result_savepath, 'log_result.txt'), 'w')

    def set_device(self, nGPU):
        if self.preferred_device == 'gpu':
            self.device = torch.device('cuda:'+str(nGPU) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print("Code will be running on device ", self.device)

    # data: num_trials * channels * sample points
    def data_augmentation(self, data, label):
        aug_data = []
        aug_label = []

        N, C, T = data.shape
        seg_size = T // self.num_segs
        aug_data_size = self.batchsize // self.num_classes
        
        for cls in range(self.num_classes):
            cls_idx = np.where(label == cls)
            cls_data = data[cls_idx]
            data_size = cls_data.shape[0]
            if data_size == 0 or data_size == 1:
                continue
            temp_aug_data = np.zeros((aug_data_size, C, T))
            for i in range(aug_data_size):
                rand_idx = np.random.randint(0, data_size, self.num_segs)
                for j in range(self.num_segs):
                    temp_aug_data[i, :, j*seg_size:(j+1)*seg_size] = cls_data[rand_idx[j], :, j*seg_size:(j+1)*seg_size]
            aug_data.append(temp_aug_data)
            aug_label.extend([cls]*aug_data_size)

        aug_data = np.concatenate(aug_data, axis=0)
        aug_label = np.array(aug_label)

        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data)
        aug_label = torch.from_numpy(aug_label)

        return aug_data, aug_label

    def train_test(self, train_dataset, test_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batchsize, num_workers=4)

        best_acc = 0
        avg_acc = 0
        best_model = None

        for epoch in range(self.epochs):
            # train
            self.net.train()
            train_loss = 0
            train_predicted = []
            train_actual = []
            with torch.enable_grad():
                for train_data, train_label in train_dataloader:
                    # data augmentation
                    aug_data, aug_label = self.data_augmentation(train_data, train_label)

                    train_data = torch.cat((train_data, aug_data), axis=0)
                    train_label = torch.cat((train_label, aug_label), axis=0)

                    train_data = train_data.type(torch.FloatTensor).to(self.device)
                    train_label = train_label.type(torch.LongTensor).to(self.device)

                    train_output = self.net(train_data)

                    running_train_loss = self.loss_func(train_output, train_label)
                    self.optimizer.zero_grad()
                    running_train_loss.backward()
                    self.optimizer.step()

                    train_loss += running_train_loss.item()

                    train_predicted.extend(torch.max(train_output, 1)[1].cpu().tolist())
                    train_actual.extend(train_label.cpu().tolist())

            train_loss /= len(train_dataloader)

            if self.scheduler is not None:
                    self.scheduler.step(train_loss)

            # test
            self.net.eval()
            test_loss = 0
            test_predicted = []
            test_actual = []
            with torch.no_grad():
                for test_data, test_label in test_dataloader:
                    test_data = test_data.type(torch.FloatTensor).to(self.device)
                    test_label = test_label.type(torch.LongTensor).to(self.device)

                    test_output = self.net(test_data)

                    running_test_loss  = self.loss_func(test_output, test_label)

                    test_predicted.extend(torch.max(test_output, 1)[1].cpu().tolist())
                    test_actual.extend(test_label.cpu().tolist())
                    test_loss += running_test_loss 

            test_loss /= len(test_dataloader)

            train_acc = accuracy_score(train_actual, train_predicted)
            test_acc = accuracy_score(test_actual, test_predicted)
            test_kappa = cohen_kappa_score(test_actual, test_predicted)

            avg_acc += test_acc

            if test_acc > best_acc:
                best_acc = test_acc
                best_kappa = test_kappa
                best_model = copy.deepcopy(self.net.state_dict())

            print('Epoch [%d] | Train Loss: %.6f  Train Accuracy: %.6f | Test Loss: %.6f  Test Accuracy: %.6f | lr: %.6f' 
                      %(epoch+1, train_loss, train_acc, test_loss, test_acc, self.optimizer.param_groups[0]['lr']))
            if self.log_write and epoch % 50 == 0:
                self.log_write.write(f'Epoch [{epoch+1}] | Train Loss: {train_loss:.6f}  Train Accuracy: {train_acc:.6f} | Test Loss: {test_loss:.6f} Test Accuracy: {test_acc:.6f} Test Kappa: {test_kappa:.6f} \n')

        avg_acc /= self.epochs
        print('The average accuracy is: ', avg_acc)
        print('The best accuracy is: ', best_acc)
        if self.log_write:
            self.log_write.write(f'The average accuracy is: {avg_acc:.6f}\n')
            self.log_write.write(f'The best accuracy is: {best_acc:.6f}\n')
            self.log_write.write(f'The best kappa is: {best_kappa:.6f}\n')
            self.log_write.close()

        torch.save(best_model, os.path.join(self.result_savepath, 'model.pth'))        