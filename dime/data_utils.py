import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset
import pickle


class UKBDataset(Dataset):
    def __init__(self, outcome, split='train'):
        super().__init__()

        outcomes_displayname = {'chronic obstructive pulmonary disease report': "COPD",
                                'asthma report': "asthma",
                                'all cause dementia report': "dementia",
                                'end stage renal disease report': "end_stage_renal_disease",
                                'myocardial infarction': "myocardial_infarction",
                                'all cause parkinsonism report': "parkinsonism",
                                'stroke': "stroke"}

        X = pd.read_csv(f"/projects/leelab3/nbbwang/UKB/small_dataset/%s/standardized_X_{split}.csv"%(outcomes_displayname[outcome]))
        Y = pd.read_csv(f"/projects/leelab3/nbbwang/UKB/small_dataset/%s/y_{split}.csv"%(outcomes_displayname[outcome]))
        self.col_names = X.columns
        self.X = np.array(X).astype('float32')
        self.feature_mapper = pickle.load(open('/projects/leelab3/nbbwang/UKB/small_dataset/feature_names_dictionary_DateToAge.pkl', 'rb'))

        # print(self.X)
        self.Y = np.array(Y).astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # print(self.Y[index].shape)
        return self.X[index], self.Y[index][0]


class DenseDatasetSelected(Dataset):
    def __init__(self, data_dir, feature_list=None, cols_to_drop=None):
        super().__init__()
        
        # Load data
        self.data_dir = os.path.expanduser(data_dir)
        data = pd.read_csv(self.data_dir)

        if cols_to_drop is not None:
            data = data.drop(columns=cols_to_drop)
        
        # Set features, x, y
        if feature_list is not None:
            self.features = feature_list
        else:
            self.features = [f for f in data.columns if f not in ['Outcome']]
        self.X = np.array(data.drop(['Outcome'], axis=1)[self.features]).astype('float32')
        self.Y = np.array(data['Outcome']).astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    

class DenseDatasetSelected(Dataset):
    def __init__(self, data_dir, feature_list=None, cols_to_drop=None):
        # Load data.
        self.data_dir = os.path.expanduser(data_dir)
        data = pd.read_csv(self.data_dir)

        print(data.columns)

        if cols_to_drop is not None:
            data = data.drop(columns=cols_to_drop)

        # Set features, inputs and outputs.
        if feature_list is not None:
            self.features = feature_list
        else:
            self.features = [f for f in data.columns if f not in ['Outcome']]
        self.X = np.array(data.drop(['Outcome'], axis=1)[self.features]).astype('float32')
        self.Y = np.array(data['Outcome']).astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class ROSMAPDataset(Dataset):
    def __init__(self, data_dir, split='train', cols_to_drop=None, use_apoe=False):
        # Load data.
        self.data_dir = os.path.expanduser(data_dir)
        if use_apoe:
            data = pd.read_csv(data_dir + f"/standardized_X_{split}.csv")
        else:
            data = pd.read_csv(data_dir + f"/standardized_X_{split}_no_apoe.csv")

        if use_apoe:
            apoe_cols = ['apoe_genotype__22.0', 'apoe_genotype__23.0', 'apoe_genotype__24.0',
                         'apoe_genotype__33.0', 'apoe_genotype__34.0']
            for c in apoe_cols:
                if c not in data.columns:
                    data[c] = 0
            data['apoe4_1copy'] = data['apoe_genotype__24.0'] + data['apoe_genotype__34.0']
            data.rename(columns={'apoe_genotype__44.0': 'apoe4_2copies'}, inplace=True)
            data.drop(labels=apoe_cols, axis=1, inplace=True)

        data['mci'] = data['dcfdx__2.0'] + data['dcfdx__3.0']
        data.drop(labels=['dcfdx__2.0', 'dcfdx__3.0'], axis=1, inplace=True)
        labels = pd.read_csv(data_dir + f"/y_{split}.csv")['onset_label_time_binary'].values.tolist()

        if cols_to_drop is not None:
            data = data.drop(columns=cols_to_drop)
        self.X = np.array(data).astype('float32')
        self.Y = np.array(labels).astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        print(self.Y[index])
        return self.X[index], self.Y[index]


class HistopathologyDownsampledDataset(Dataset):
    def __init__(self, data_dir, df, image_transforms):
        self.image_id_list = list(df['Image Name'])
        self.transforms = image_transforms
        self.labels = list((df['Majority Vote Label'] == 'SSA').astype('int'))
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.data_dir + self.image_id_list[idx])
        image = self.transforms(image)
        return image, self.labels[idx]


class HistopathologyDownsampledEdgeDataset(Dataset):
    def __init__(self, data_dir, df, image_transforms):
        self.image_id_list = list(df['Image Name'])
        self.transforms = image_transforms
        self.labels = list((df['Majority Vote Label'] == 'SSA').astype('int'))
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.data_dir + self.image_id_list[idx])
        orig_image_transformed = self.transforms(image)
        im = Image.fromarray(cv2.bilateralFilter(np.array(image), 5, 75, 75))
        gray_scale = transforms.Grayscale(1)
        sketch_image = gray_scale(im)
        edges = cv2.Canny(np.array(sketch_image), 150, 250)
        out = np.stack([edges, edges, edges], axis=-1)
        out = transforms.ToPILImage()(out)
        sketch_image = self.transforms(out)
        return orig_image_transformed, sketch_image, self.labels[idx]


def get_group_matrix(features, feature_groups):
    # Add singleton groups.
    complete_groups = {}
    groups_union = []
    for key in feature_groups:
        groups_union += feature_groups[key]
        complete_groups[key] = feature_groups[key]
    for feature in features:
        if feature not in groups_union:
            complete_groups[feature] = [feature]

    # Create groups matrix.
    group_matrix = np.zeros((len(complete_groups), len(features)))
    for i, key in enumerate(complete_groups):
        inds = [features.index(feature) for feature in complete_groups[key]]
        group_matrix[i, inds] = 1

    return complete_groups, group_matrix


def get_groups_dict_mask(feature_groups, num_feature):
    group_start = list(feature_groups.keys())
    feature_groups_dict = {}
    num_group = 0
    i = 0
    while i < num_feature:
        feature_groups_dict[num_group] = []
        if i in group_start:
            for j in range(feature_groups[i]):
                feature_groups_dict[num_group].append(i+j)
            num_group += 1
            i += feature_groups[i]
        else:
            feature_groups_dict[num_group].append(i)
            num_group += 1
            i += 1
    feature_groups_mask = np.zeros((num_feature, len(feature_groups_dict)))
    for i in range(len(feature_groups_dict)):
        for j in feature_groups_dict[i]:
            feature_groups_mask[j, i] = 1
    return feature_groups_dict, feature_groups_mask


def get_xy(dataset):
    x, y = zip(*list(dataset))
    if isinstance(x[0], np.ndarray):
        return np.array(x), np.array(y)
    elif isinstance(x[0], torch.Tensor):
        if isinstance(y[0], (int, float)):
            return torch.stack(x), torch.tensor(y)
        else:
            return torch.stack(x), torch.stack(y)
    else:
        raise ValueError(f'not sure how to concatenate data type: {type(x[0])}')


def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0):
    # Shuffle sample indices.
    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test + n_val):]

    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)
    return train_dataset, val_dataset, test_dataset
