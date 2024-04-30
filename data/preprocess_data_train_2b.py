import mne
from mne import event
from mne import epochs
from mne import label
from mne.io.fiff import raw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import time

factor_new = 1e-3
init_block_size = 1000

data_path = 'E:/Dataset/BCI_Competition_IV/Datasets2b'
all_data_files = [['B0'+str(i)+'0'+str(j)+'T.gdf' for j in range(1,4)] for i in range(1,10)]

label_path = 'E:/Dataset/BCI_Competition_IV/True_Labels/Datasets2b'
all_label_files = [['B0'+str(i)+'0'+str(j)+'T.mat' for j in range(1,4)] for i in range(1,10)]

save_path = 'dataset/bci_iv_2b/raw'

if not os.path.exists(save_path):
    os.makedirs(save_path)

event_description = {'769':"CueLeft", '770':"CueRight"}

for sub in range(1, 10):
    print(f'Processing Subject {sub}'.format(sub))
    data_files = all_data_files[sub-1]
    label_files = all_label_files[sub-1]
    sub_data = []
    sub_labels = []
    for i in range(len(data_files)):
        raw_data = mne.io.read_raw_gdf(os.path.join(data_path, data_files[i]), preload=True, verbose=False)
    
        raw_events, all_event_id = mne.events_from_annotations(raw_data)

        raw_data = mne.io.RawArray(raw_data.get_data()*1e6, raw_data.info)

        # 滤波4-38Hz
        # start = time.time()
        # raw_data.filter(4, 38, fir_design='firwin')
        # end = time.time()
        # print("Cost: ", end-start)

        raw_data.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']

        picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')

        tmin, tmax = 0, 4

        event_id = dict()

        for event in all_event_id:
            if event in event_description:
                event_id[event_description[event]] = all_event_id[event]

        raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

        true_labels = raw_epochs.events[:, -1] - event_id['CueLeft'] + 1
        data = raw_epochs.get_data() # [n_epochs, n_channels, n_times]
        data = data[:, :, :-1]
        print(data.shape)
        print(true_labels.shape)

        sub_data.append(data)
        sub_labels.extend(true_labels)
    
    sub_data = np.concatenate(sub_data, axis=0)
    sub_labels = np.array(sub_labels)

    print('Data shape', sub_data.shape)
    print('Label shape', sub_labels.shape)

    np.save(os.path.join(save_path, 'B0'+str(sub)+'T_data.npy'), sub_data)
    np.save(os.path.join(save_path, 'B0'+str(sub)+'T_label.npy'), sub_labels) 

