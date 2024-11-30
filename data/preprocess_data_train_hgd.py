from typing import OrderedDict
import numpy as np
import mne
import os
from hgd_dataset import BBCIDataset
from hgd_data_utils import create_data_label_from_raw_mne, resample_raw_mne
from hgd_signal_utils import mne_apply, exponential_running_standardize, highpass_cnt

data_path = 'raw_data/train'
save_path = 'process_data/train'

time_window = [0, 4000]

for sub_id in range(1,15):
    print('Processing sub ', sub_id)
    data_file = os.path.join(data_path, str(sub_id)+'.mat')

    loader = BBCIDataset(data_file)
    print('Loading data...')

    raw_mne = loader.load()

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2]),
                              ('Rest', [3]), ('Feet', [4])])

    data, labels = create_data_label_from_raw_mne(raw_mne, marker_def, time_window)
    
    clean_trial_mask = np.max(np.abs(data), axis=(1, 2)) < 800

    print("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(data),
        np.mean(clean_trial_mask) * 100))
    
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    raw_mne = raw_mne.pick_channels(C_sensors)

    print('Resampling...')
    raw_mne = resample_raw_mne(raw_mne, 250.0)
    print('Highpassing...')
    raw_mne = mne_apply(lambda a: highpass_cnt(a, 0, raw_mne.info['sfreq'], filt_order=3, axis=1), raw_mne)
    print('Standardizing...')
    raw_mne = mne_apply(lambda a: exponential_running_standardize(a.T, factor_new=1e-3, 
    init_block_size=1000, eps=1e-4).T, raw_mne)


    train_data, train_label = create_data_label_from_raw_mne(raw_mne, marker_def, time_window)
    train_data = train_data[clean_trial_mask]
    train_label = train_label[clean_trial_mask]
    print('Data shape', train_data.shape)
    print('Label shape', train_label.shape)
    
    np.save(os.path.join(save_path, str(sub_id)+'_data.npy'), train_data)
    np.save(os.path.join(save_path, str(sub_id)+'_label.npy'), train_label)

