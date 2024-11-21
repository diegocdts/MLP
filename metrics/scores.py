import numpy as np
from skimage import metrics
from sklearn.metrics import mean_squared_error


def mse(output, target):
    return mean_squared_error(target, output)

def snr2(_input, output, target):

    diff_clean = output - target
    mse_pred_clean = np.mean(np.power(diff_clean, 2))

    diff_noisy = _input - target
    mse_noisy_clean = np.mean(np.power(diff_noisy, 2))

    _snr2 = 1 - (np.sqrt(mse_pred_clean) / np.sqrt(mse_noisy_clean))

    return _snr2

def psnr(output, target, data_range):
    psnr = metrics.peak_signal_noise_ratio(target, output, data_range=data_range)
    return psnr