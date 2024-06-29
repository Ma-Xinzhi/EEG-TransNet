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

data_path = 'E:/Dataset/BCI_Competition_IV/Datasets2a'
data_files = ['A0'+str(i)+'T.gdf' for i in range(1,10)]

save_path = 'dataset/bci_iv_2a'

event_description = {'769':"CueLeft", '770':"CueRight", '771':"CueFoot", '772':"CueTongue"}

for file in data_files:
    raw_data = mne.io.read_raw_gdf(os.path.join(data_path, file), preload=True, verbose=False)
    
    # print(raw_data)

    raw_events, all_event_id = mne.events_from_annotations(raw_data)
    # print(raw_events)

    raw_data = mne.io.RawArray(raw_data.get_data()*1e6, raw_data.info)


    # filtered_data = exponential_running_standardize(raw_data.get_data().T, factor_new=factor_new, init_block_size=init_block_size)

    # raw_data = mne.io.RawArray(filtered_data.T, raw_data.info)

    # raw_data, raw_events = resample(raw_data, raw_events, 128)

    raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

    picks = mne.pick_types(raw_data.info, eeg=True, exclude='bads')

    tmin, tmax = 0, 4

    # left_hand = 769, right_hand = 770, foot = 771, tongue = 772
    # event_id = dict({'769':7, '770':8, '771':9, '772':10})
    event_id = dict()

    for event in all_event_id:
        if event in event_description:
            event_id[event_description[event]] = all_event_id[event]

    raw_epochs = mne.Epochs(raw_data, raw_events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)

    # print(test_epochs)

    true_labels = raw_epochs.events[:, -1] - event_id['CueLeft'] + 1
    data = raw_epochs.get_data() # [n_epochs, n_channels, n_times]
    data = data[:, :, :-1]
    # print(data.shape)

    np.save(os.path.join(save_path, file[:-4]+'_data.npy'), data)
    np.save(os.path.join(save_path, file[:-4]+'_label.npy'), true_labels)

