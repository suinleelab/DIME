# Ablation where the sketch is integrated into the predictor only for a pre-trained and otherwise frozen version of DIME. 
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import AUROC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from fastai.vision.all import untar_data, URLs
import pandas as pd
import sys
sys.path.append('../')
from data_utils import MaskLayerGaussian, MaskLayer2d, HistopathologyDownsampledDataset, HistopathologyDownsampledEdgeDataset
sys.path.append('../../')
from dime.masking_pretrainer import MaskingPretrainer
from dime.greedy_models import GreedyCMIEstimator
from dime.sketch_supervision_predictor import SketchSupervisionPredictor
from utils import accuracy, auc, normalize
from dime.vit import PredictorViT, ValueNetworViT, PredictorSemiSupervisedVit, ValueNetworkSemiSupervisedVit
from dime.resnet_imagenet import resnet18, resnet34, resnet50, Predictor, ValueNetwork, ResNet18Backbone
# from dime.vit import vit_tiny_patch16_224
import timm

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mask_type', type=str, 
                                default='zero', 
                                choices=['gaussian', 'zero'], 
                                help="Type of mask to apply: either Gaussain blur (gaussian) or zero-out (zero)")
parser.add_argument('--mask_width', type=int, 
                                default=14, 
                                choices=[7, 14], 
                                help="Mask width to use in the mask layer")
parser.add_argument('--lr', type=float, 
                                default=1e-5, 
                                help="Learning rate used train the network")

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    mask_type = args.mask_type
    image_size = 224
    mask_width = args.mask_width
    lr = args.lr
    if lr == 1e-3:
        min_lr = 1e-6
    else:
        min_lr = 1e-8

    if mask_type == 'gaussian':
        mask_layer = MaskLayerGaussian(append=False, mask_width=mask_width, patch_size=image_size/mask_width, sigma=1)
    else:
        mask_layer = MaskLayer2d(append=False, mask_width=mask_width, patch_size=image_size/mask_width)
        
    device = torch.device('cuda', args.gpu)

    norm_constants = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    pretrained_model_name = 'vit_small_patch16_224'

    # Setup for data loading.
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants),
    ])

    data_dir = '/projects/<labname>/<username>/hist_data/mhist/'

    # Get train and test datasets
    df = pd.read_csv(data_dir + 'annotations.csv')
    train_dataset = HistopathologyDownsampledEdgeDataset(data_dir + 'images/', df.loc[df['Partition'] == 'train'], transforms_train)
    test_dataset = HistopathologyDownsampledEdgeDataset(data_dir + 'images/', df.loc[df['Partition'] == 'test'], transforms_test)
    test_dataset_len = len(test_dataset)

    # Split test dataset into val
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(test_dataset_len, size=int(test_dataset_len*0.5), replace=False))
    test_inds = np.setdiff1d(np.arange(test_dataset_len), val_inds)

    val_dataset = torch.utils.data.Subset(test_dataset, val_inds)
    test_dataset = torch.utils.data.Subset(test_dataset, test_inds)

    # Prepare dataloaders.
    mbsize = 16
    train_dataloader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, pin_memory=True,
                            drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=mbsize, pin_memory=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=mbsize, pin_memory=True, drop_last=True, num_workers=4)
    
    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # value_network = ValueNetwork(backbone, block_layer_stride=block_layer_stride)
    backbone = timm.create_model(pretrained_model_name, pretrained=True)
    predictor =  PredictorViT(backbone, num_classes=2)
    value_network = ValueNetworViT(backbone, mask_width=mask_width)
    predictor_with_sketch =  PredictorSemiSupervisedVit(backbone, backbone, num_classes=2)

    trained_predictor_name = f"{pretrained_model_name}_predictor_sketch_in_predictor_lr_{str(lr)}{mask_type}_mask_width_{mask_width}.pth"
    if os.path.exists(f"results/{trained_predictor_name}"):
        # Load pretrained predictor
        print("Loading Pretrained Predictor")
        print("-"*8)
        predictor.load_state_dict(torch.load(f"results/{trained_predictor_name}"))
    else:
        # Pretrain predictor
        print("Pretraining Predictor")
        pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
        pretrain.fit(train_dataset,
                     val_dataset,
                     mbsize=mbsize,
                     lr=lr,
                     min_lr=min_lr,
                     nepochs=50,
                     loss_fn=nn.CrossEntropyLoss(),
                     val_loss_fn=AUROC(task='multiclass', num_classes=2),
                     val_loss_mode='max',
                     patience=5,
                     verbose=True,
                     trained_predictor_name=trained_predictor_name)

    run_description = f"max_features_100_{pretrained_model_name}_lr_{str(lr)}_sketch_in_predictor_use_entropy_{mask_type}_mask_width_{mask_width}"

    greedy_cmi_estimator = GreedyCMIEstimator(value_network, predictor, mask_layer).to(device)
    greedy_cmi_estimator.fit(train_dataloader, 
                            val_dataloader,
                            lr=lr,
                            min_lr=min_lr,
                            nepochs=50,
                            max_features=100,
                            eps=0.05,
                            loss_fn=nn.CrossEntropyLoss(reduction='none'),
                            val_loss_fn=auc,
                            tensorboard_file_name_suffix=run_description,
                            eps_decay=True,
                            eps_decay_rate=0.2,
                            patience=3,
                            feature_costs=None,
                            use_entropy=True)
    
    predictor.load_state_dict(torch.load(f"results/predictor_trained_{run_description}.pth"))
    value_network.load_state_dict(torch.load(f"results/value_network_trained_{run_description}.pth"))

    sketch_supervision_predictor = SketchSupervisionPredictor(value_network, predictor, predictor_with_sketch, mask_layer).to(device)
    sketch_supervision_predictor.fit(train_dataloader, 
                            val_dataloader,
                            lr=lr,
                            min_lr=min_lr,
                            nepochs=250,
                            max_features=100,
                            eps=0.05,
                            loss_fn=nn.CrossEntropyLoss(reduction='none'),
                            val_loss_fn=auc,
                            tensorboard_file_name_suffix=run_description,
                            eps_decay=True,
                            eps_decay_rate=0.2,
                            patience=3,
                            feature_costs=None,
                            use_entropy=True)
    

    
