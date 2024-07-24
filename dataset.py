import os
import csv
import numpy as np
from fastdtw import fastdtw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from einops import repeat


from interpolate import natural_cubic_spline_coeffs


files = {
    'PEMS03': ['PEMS03/pems03.npz', 'PEMS03/distance.csv'],
    'PEMS04': ['PEMS04/PEMS04.npz', 'PEMS04/distance.csv'],
    'PEMS07': ['PEMS07/pems07.npy', 'PEMS07/distance.csv'],
    'PEMS08': ['PEMS08/pems08.npy', 'PEMS08/distance.csv'],
    'PEMSBAY': ['PEMSBAY/pems_bay.npy', 'PEMSBAY/distance.csv'],
    'PeMSD7M': ['PeMSD7M/PeMSD7M.npy', 'PeMSD7M/distance.csv'],
    'PeMSD7L': ['PeMSD7L/PeMSD7L.npy', 'PeMSD7L/distance.csv'],
    'syn': ['syn/syndata.npy', 'syn/distance.csv']
}

def read_data(args):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        sp_matrix: array, spatial adjacency matrix
    """
    filename = args.filename
    file = files[filename]
    filepath = "./data/"
    data = np.load(filepath + file[0])['data']
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    # use directed graph
    # if not os.path.exists(f'data/{filename}_adj.npy'):
    with open(filepath + file[1], 'r') as fp:
        dist_matrix = np.zeros((num_node, num_node)) 
        next(fp)
        file = csv.reader(fp)
        for line in file:
            start = int(line[0])
            end = int(line[1])
            dist_matrix[start][end] = 1
            # dist_matrix[start][end] = float(line[2])
        np.save(f'data/{filename}/{filename}_adj.npy', dist_matrix)

    adj = np.load(f'data/{filename}/{filename}_adj.npy')

    return torch.from_numpy(data).to(torch.float32), mean_value, std_value, adj 


def get_normalized_adj_tensor(A):
    A2 = torch.mm(A, A)
    A3 = torch.mm(A, A2)
    A = A + A2 + A3
    row_sum = torch.sum(A, axis=1) + 1e-9
    col_sum = torch.sum(A, axis=0) + 1e-9
    row_sum_sqrt_inv = 1 / torch.sqrt(row_sum)
    col_sum_sqrt_inv = 1 / torch.sqrt(col_sum)
    A_wave = torch.einsum('i, ij, j->ij', row_sum_sqrt_inv, A, col_sum_sqrt_inv)
    return A_wave

def get_normalized_adj(A):
    A2 = np.dot(A, A)
    A3 = np.dot(A, A2)
    A = A + A2 + A3
    row_sum = np.sum(A, axis=1) + 1e-9
    col_sum = np.sum(A, axis=0) + 1e-9
    row_sum_sqrt_inv = 1 / np.sqrt(row_sum)
    col_sum_sqrt_inv = 1 / np.sqrt(col_sum)
    A_wave = np.einsum('i, ij, j->ij', row_sum_sqrt_inv, A, col_sum_sqrt_inv)
    return A_wave
    # A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    # return A_reg
    # return torch.from_numpy(A_reg.astype(np.float32))


def generate_dataset(data, args):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length
    total_length = data.shape[0]

    train_data = data[: int(total_length * train_ratio)]
    valid_data = data[int(total_length * train_ratio): int(total_length * (train_ratio + valid_ratio))]
    test_data = data[int(total_length * (train_ratio + valid_ratio)): ]

    def add_window(data):
        max_index = data.shape[0] 
        index = 0 
        X, Y = [], []
        index = 0
        while index + his_length + pred_length < max_index:
            X.append(data[index: index + his_length])
            Y.append(data[index + his_length: index + pred_length + his_length][:, :, 0])
            index += 1
        X = torch.stack(X, dim=0).permute(0, 2, 1, 3)
        aug_time = repeat(torch.linspace(0, 11, 12), 'h -> a b h c', a=X.shape[0], b=X.shape[1], c=1)
        X = torch.cat([X, aug_time], dim=3)

        Y = torch.stack(Y, dim=0).permute(0, 2, 1)
        return X, Y

    x_train, y_train = add_window(train_data)
    x_valid, y_valid = add_window(valid_data)
    x_test, y_test = add_window(test_data)

    times = torch.linspace(0, 11, 12)
    train_coeffs = natural_cubic_spline_coeffs(times, x_train)
    valid_coeffs = natural_cubic_spline_coeffs(times, x_valid)
    test_coeffs = natural_cubic_spline_coeffs(times, x_test)

    def get_loader(X, Y, batch_size, shuffle):
        data = torch.utils.data.TensorDataset(*X, Y)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    train_loader = get_loader(train_coeffs, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = get_loader(valid_coeffs, y_valid, batch_size=batch_size, shuffle=False)
    test_loader = get_loader(test_coeffs, y_test, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
