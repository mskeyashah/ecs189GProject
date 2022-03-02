'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import numpy as np
import torch
import pandas as pd


def process_features(features):
    row_sum_diag = np.sum(features, axis=1)
    row_sum_diag_inv = [1 / x for x in row_sum_diag]
    row_sum_inv = np.diag(row_sum_diag_inv)
    return np.dot(row_sum_inv, features)


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    ylabel = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_data(self):
        df = pd.read_csv(self.dataset_source_folder_path + 'node', sep='\t', header=None).sort_index()
        df = df.sort_values(by=df.columns[0])
        df.reset_index(drop=True)
        idx_train = []
        idx_test = []
        for lab in self.ylabel:
            testing = df[df[len(df.columns) -1] == lab]
            idxlist = testing.sample(n=20).index.tolist()
            testing = testing.drop(idxlist)
            idx_train = idx_train + idxlist
            idx_test = idx_test + testing.sample(n=200, replace=True).index.tolist()

        ally = pd.get_dummies(df[df.columns[len(df.columns) - 1]])
        allx = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)

        link = pd.read_csv(self.dataset_source_folder_path+'link', sep='\t', header=None, names=["to", "from"])
        link.set_index("from", inplace=True)
        link = link.sort_index()
        indexes = link.index
        adj = np.zeros((len(allx), len(allx)))

        for i in range(len(link)):
            adj[allx.index[allx[0] == indexes[i]].tolist()[0]][
                allx.index[allx[0] == link["to"].iloc[i]].tolist()[0]] = 1

        allx.drop(allx.columns[0], axis=1, inplace=True)
        allx = allx.reset_index(drop=True)

        # preprocess test indices and combine all data
        features = np.vstack([allx])
        labels = np.vstack([ally])

        train_mask = sample_mask(idx_train, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])
        zeros = np.zeros(labels.shape)
        y_train = zeros.copy()
        y_test = zeros.copy()
        y_train[train_mask, :] = labels[train_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        features = torch.from_numpy(process_features(features))
        y_train, y_test, train_mask, test_mask = \
            torch.from_numpy(y_train), torch.from_numpy(y_test), \
            torch.from_numpy(train_mask), torch.from_numpy(test_mask)

        return adj, features, y_train, y_test, train_mask, test_mask