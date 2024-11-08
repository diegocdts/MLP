import glob
import numpy as np
import segyio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


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


def plot_compare(blended, deblended, predicted = None, comparison_path = None):
    columns = 3 if predicted is not None else 2

    plt.subplot(1, columns, 1)
    vmin = blended.mean() - blended.std()
    vmax = blended.mean() + blended.std()
    plt.imshow(blended.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
    plt.title('Blended')

    plt.subplot(1, columns, 2)
    plt.imshow(deblended.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
    plt.title('Deblended')

    if predicted is not None:
        vmin = predicted.mean() - predicted.std()
        vmax = predicted.mean() + predicted.std()
        plt.subplot(1, 3, 3)
        plt.imshow(predicted.T, aspect='auto', cmap='seismic', origin='upper', vmin=vmin, vmax=vmax)
        plt.title('Prediction')
    plt.colorbar()
    plt.savefig(comparison_path)
    plt.close()


def compare(blended_path, deblended_path, predicted_path, prediction_dir, n_receivers):
    blended = zscore(load_file(blended_path))
    deblended = zscore(load_file(deblended_path))
    predicted = load_file(f'{predicted_path}.npy')
    shape = blended.shape
    print(shape)
    n_images = int(shape[0] / n_receivers)
    for i in range(n_images):
        start = i * n_receivers
        end = start + n_receivers
        comparison_path = f'{prediction_dir}/SHOT_{i}.png'
        plot_compare(blended[start:end], deblended[start:end], predicted[start:end], comparison_path)
