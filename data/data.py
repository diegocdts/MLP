import glob
import os.path

import numpy as np
import segyio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from metrics.scores import snr2, psnr


class TraceDataset(Dataset):

    def __init__(self, inputs, targets = None):
        self.inputs = inputs
        self.targets = None if targets is None else targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.inputs[idx]
        else:
            return self.inputs[idx], self.targets[idx]


def input_target_names(input_path, target_path, start, end):
    if os.path.exists(input_path) and os.path.exists(target_path):
        if os.path.isfile(input_path) and os.path.isfile(target_path):
            return [input_path], [target_path]
    input_files = sorted(glob.glob(f'{input_path}*'))[start:end]
    all_target_files = sorted(glob.glob(f'{target_path}*'))
    if len(input_files) == 1 and len(all_target_files) == 1:
        return input_files, all_target_files

    target_files = []
    for input_file in input_files:
        end = -5 if input_file.endswith('segy') else -4
        suffix = input_file[input_file.index('inline'):end]
        for target_file in all_target_files:
            if suffix in target_file:
                target_files.append(target_file)
                break
    return input_files, target_files


def load_data(x_names, y_names, n_receivers):
    x_data, y_data = [], []

    for idx in range(len(x_names)):
        x_data.extend(load_file(x_names[idx]))
        y_data.extend(load_file(y_names[idx]))

    x_data, y_data = np.array(x_data), np.array(y_data)
    x_mean, x_std = x_data.mean(), x_data.std()

    x_data = normalize_slices(x_data, n_receivers)
    y_data = normalize_slices(y_data, n_receivers)
    return x_data, y_data, x_mean, x_std


def load_file(file_path):
    if 'segy' in file_path or 'sgy' in file_path:
        f1 = segyio.open(file_path, ignore_geometry=True)
        data = segyio.collect(f1.trace[:])
    else:
        data = np.load(file_path)
    if len(data.shape) == 3:
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    return data.real.astype(np.float32)


def normalize_slices(data, n_receivers):
    normalized_data = []
    n_shots = int(data.shape[0] / n_receivers)
    for idx in range(n_shots):
        start = idx * n_receivers
        end = start + n_receivers
        shot = zscore(data[start:end])
        normalized_data.extend(shot)
    return np.array(normalized_data)


def zscore(data, mean = None, std = None):
    if mean is None and std is None:
        return (data - data.mean()) / (3 * data.std())
    else:
        return (data - mean) / (3 * std)


def plot_image(data):
    plt.imshow(data.T, aspect='auto', cmap='seismic', origin='upper', vmin=-10e+4, vmax=10e+4)
    plt.colorbar()
    plt.show()


def plot_compare(blended, deblended, predicted=None, _snr2=None, _psnr=None, comparison_path=None):
    columns = 4 if predicted is not None else 2

    fig, axs = plt.subplots(1, columns, figsize=(12, 6), constrained_layout=True)
    vmin = blended.mean() - blended.std()
    vmax = blended.mean() + blended.std()

    im1 = axs[0].imshow(blended.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
    axs[0].set_title('Blended')

    im2 = axs[1].imshow(deblended.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
    axs[1].set_title('Deblended')

    im4 = None
    if predicted is not None:
        vmin = predicted.mean() - predicted.std()
        vmax = predicted.mean() + predicted.std()
        im3 = axs[2].imshow(predicted.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
        axs[2].set_title('Prediction')

        axs[2].text(0.05, 0.05, f'SNR2: {round(float(_snr2), 4)}\nPSNR: {round(_psnr, 4)}', color='black', ha='left', va='bottom',
                    transform=axs[2].transAxes, fontsize=10)

        diff = predicted - deblended
        im4 = axs[3].imshow(diff.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
        axs[3].set_title('Difference')

    cbar = fig.colorbar(im2 if predicted is None else im4, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

    if comparison_path:
        plt.savefig(comparison_path)
    else:
        plt.show()
    plt.close(fig)


def compare(blended_path, deblended_path, predicted_path, prediction_dir, n_receivers):
    blended = zscore(load_file(blended_path))
    deblended = zscore(load_file(deblended_path))
    predicted = load_file(f'{predicted_path}.npy')

    shape = blended.shape
    print(shape)

    _snr2_shots, _psnr_shots = [], []
    _snr2_receivers, _psnr_receivers = [], []

    n_shots = int(shape[0] / n_receivers)

    for i in range(n_shots):
        start = i * n_receivers
        end = start + n_receivers

        _snr2, _psnr = metrics(blended[start:end], deblended[start:end], predicted[start:end])
        _snr2_shots.append(_snr2)
        _psnr_shots.append(_psnr)

        comparison_path = f'{prediction_dir}/SHOT_{i}.png'
        plot_compare(blended[start:end], deblended[start:end], predicted[start:end], _snr2, _psnr, comparison_path)

    blended2 = blended.reshape(n_shots, n_receivers, shape[1])
    deblended2 = deblended.reshape(n_shots, n_receivers, shape[1])
    predicted2 = predicted.reshape(n_shots, n_receivers, shape[1])
    for i in range(n_receivers):
        _snr2, _psnr = metrics(blended2[:,i,:], deblended2[:,i,:], predicted2[:,i,:])
        _snr2_receivers.append(_snr2)
        _psnr_receivers.append(_psnr)

        comparison_path = f'{prediction_dir}/RECEIVER_{i}.png'
        plot_compare(blended2[:,i,:], deblended2[:,i,:], predicted2[:,i,:], _snr2, _psnr, comparison_path)

    metrics_shots_path = f'{prediction_dir}/METRICS_SHOTS.csv'
    metrics_receivers_path = f'{prediction_dir}/METRICS_RECEIVERS.csv'
    write_metrics(_snr2_shots, _psnr_shots, metrics_shots_path)
    write_metrics(_snr2_receivers, _psnr_receivers, metrics_receivers_path)


def metrics(blended, deblended, predicted):
    vmin = predicted.mean() - predicted.std()
    vmax = predicted.mean() + predicted.std()
    _snr2 = snr2(blended, predicted, deblended)
    _psnr = psnr(predicted, deblended, data_range=vmax - vmin)
    return _snr2, _psnr

def write_metrics(_snr2, _psnr, path):
    mean_snr2 = sum(_snr2)/len(_snr2)
    mean_psnr = sum(_psnr)/len(_psnr)
    with open(path, 'w') as file:
        file.write(f'MEAN_SNR2: {mean_snr2}')
        file.write(f'MEAN_PSNR: {mean_psnr}')
        file.write('SNR2 PSNR\n')
        for idx in range(len(_snr2)):
            file.write(f'{_snr2[idx]} {_psnr[idx]}\n')
