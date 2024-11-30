import logging

import pandas as pd
import numpy as np
import scipy
import scipy.signal
import mne
from copy import deepcopy
import resampy
from mne.io.base import concatenate_raws

log = logging.getLogger(__name__)


def exponential_running_standardize(
    data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    """
    Perform exponential running standardization. 
    
    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    
    Then, compute exponential running variance :math:`v_t` at time `t` as 
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.
    
    
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization. 
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (
            data[0:init_block_size] - init_mean
        ) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized


def exponential_running_demean(data, factor_new=0.001, init_block_size=None):
    """
    Perform exponential running demeanining. 

    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Deman the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t)`.


    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Demean data before to this index with regular demeaning. 
        
    Returns
    -------
    demeaned: 2darray (time, channels)
        Demeaned data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    demeaned = np.array(demeaned)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        demeaned[0:init_block_size] = data[0:init_block_size] - init_mean
    return demeaned


def highpass_cnt(data, low_cut_hz, fs, filt_order=3, axis=0):
    """
     Highpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    highpassed_data: 2d-array
        Data after applying highpass filter.
    """
    if (low_cut_hz is None) or (low_cut_hz == 0):
        log.info("Not doing any highpass, since low 0 or None")
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, low_cut_hz / (fs / 2.0), btype="highpass"
    )
    assert filter_is_stable(a)
    data_highpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_highpassed


def lowpass_cnt(data, high_cut_hz, fs, filt_order=3, axis=0):
    """
     Lowpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    high_cut_hz: float
    fs: float
    filt_order: int

    Returns
    -------
    lowpassed_data: 2d-array
        Data after applying lowpass filter.
    """
    if (high_cut_hz is None) or (high_cut_hz == fs / 2.0):
        log.info(
            "Not doing any lowpass, since high cut hz is None or nyquist freq."
        )
        return data.copy()
    b, a = scipy.signal.butter(
        filt_order, high_cut_hz / (fs / 2.0), btype="lowpass"
    )
    assert filter_is_stable(a)
    data_lowpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_lowpassed


def bandpass_cnt(
    data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0, filtfilt=False
):
    """
     Bandpass signal applying **causal** butterworth filter of given order.

    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    filtfilt: bool
        Whether to use filtfilt instead of lfilter

    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
        high_cut_hz == None or high_cut_hz == fs / 2.0
    ):
        log.info(
            "Not doing any bandpass, since low 0 or None and "
            "high None or nyquist frequency"
        )
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(
            data, high_cut_hz, fs, filt_order=filt_order, axis=axis
        )
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        log.info(
            "Using highpass filter since high cut hz is None or nyquist freq"
        )
        return highpass_cnt(
            data, low_cut_hz, fs, filt_order=filt_order, axis=axis
        )

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype="bandpass")
    assert filter_is_stable(a), "Filter should be stable..."
    if filtfilt:
        data_bandpassed = scipy.signal.filtfilt(b, a, data, axis=axis)
    else:
        data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed


def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.
    
    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.

    Returns
    -------
    is_stable: bool
        Filter is stable or not.  
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.
    
    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a))
    )
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)


def concatenate_raws_with_events(raws):
    """
    Concatenates `mne.io.RawArray` objects, respects `info['events']` attributes
    and concatenates them correctly. Also does not modify `raws[0]` inplace
    as the :func:`concatenate_raws` function of MNE does.
    
    Parameters
    ----------
    raws: list of `mne.io.RawArray`

    Returns
    -------
    concatenated_raw: `mne.io.RawArray`
    """
    # prevent in-place modification of raws[0]
    raws[0] = deepcopy(raws[0])
    event_lists = [r.info["events"] for r in raws]
    new_raw, new_events = concatenate_raws(raws, events_list=event_lists)
    new_raw.info["events"] = new_events
    return new_raw


def resample_cnt(cnt, new_fs):
    """
    Resample continuous recording using `resampy`.

    Parameters
    ----------
    cnt: `mne.io.RawArray`
    new_fs: float
        New sampling rate.

    Returns
    -------
    resampled: `mne.io.RawArray`
        Resampled object.

    """
    if new_fs == cnt.info["sfreq"]:
        log.info(
            "Just copying data, no resampling, since new sampling rate same."
        )
        return deepcopy(cnt)
    log.warning("This is not causal, uses future data....")
    log.info(
        "Resampling from {:f} to {:f} Hz.".format(cnt.info["sfreq"], new_fs)
    )

    data = cnt.get_data().T

    new_data = resampy.resample(
        data, cnt.info["sfreq"], new_fs, axis=0, filter="kaiser_fast"
    ).T
    old_fs = cnt.info["sfreq"]
    new_info = deepcopy(cnt.info)
    new_info["sfreq"] = new_fs
    events = new_info["events"]
    event_samples_old = cnt.info["events"][:, 0]
    event_samples = event_samples_old * new_fs / float(old_fs)
    events[:, 0] = event_samples
    return mne.io.RawArray(new_data, new_info)


def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.
    
    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.

    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.

    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)


def common_average_reference_cnt(cnt,):
    """
    Common average reference, subtract average over electrodes at each timestep.

    Parameters
    ----------
    cnt: `mne.io.RawArray`

    Returns
    -------
    car_cnt: cnt: `mne.io.RawArray`
        Same data after common average reference.
    """

    return mne_apply(lambda a: a - np.mean(a, axis=0, keepdim=True), cnt)