# PyTorch lightning version

import torch
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import AUROC
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dime.data_utils import HistopathologyDownsampledEdgeDataset
from dime.utils import MaskLayer2d
from dime import MaskingPretrainerPrior
from dime import CMIEstimatorPrior
from dime.vit import PredictorViTPrior, ValueNetworkViTPrior
import timm

vit_model_options = ['vit_small_patch16_224', 'vit_tiny_patch16_224']

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
parser.add_argument('--pretrained_model_name', type=str,
                    default='vit_small_patch16_224',
                    choices=vit_model_options,
                    help="Name of the pretrained model to use")

if __name__ == '__main__':
    auc_metric = AUROC(task='multiclass', num_classes=2)

    # Parse args
    args = parser.parse_args()
    mask_type = args.mask_type
    image_size = 224
    mask_width = args.mask_width
    pretrained_model_name = args.pretrained_model_name
    lr = args.lr
    if lr == 1e-3:
        min_lr = 1e-6
    else:
        min_lr = 1e-8

    mask_layer = MaskLayer2d(append=False, mask_width=mask_width, patch_size=image_size/mask_width)
        
    device = torch.device('cuda', args.gpu)
    norm_constants = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Setup for data loading.
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants)
    ])

    data_dir = '/projects/<labname>/<username>/hist_data/MHIST/'

    # Get train and test datasets
    df = pd.read_csv(data_dir + 'annotations.csv')
    train_dataset = HistopathologyDownsampledEdgeDataset(data_dir + 'images/', df.loc[df['Partition'] == 'train'],
                                                         transforms_train)
    test_dataset = HistopathologyDownsampledEdgeDataset(data_dir + 'images/', df.loc[df['Partition'] == 'test'],
                                                        transforms_test)
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
    val_dataloader = DataLoader(val_dataset, batch_size=mbsize, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=mbsize, pin_memory=True, num_workers=4)

    # Make results directory.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Incorporate prior information using sketch (Canny edge image)
    backbone1 = timm.create_model(pretrained_model_name, pretrained=True)
    backbone2 = timm.create_model(pretrained_model_name, pretrained=True)

    predictor = PredictorViTPrior(backbone1, backbone2, num_classes=2)
    value_network = ValueNetworkViTPrior(backbone1, backbone2)

    # pretrain = MaskingPretrainerPrior(
    #         predictor,
    #         mask_layer,
    #         lr=1e-5,
    #         loss_fn=nn.CrossEntropyLoss(),
    #         val_loss_fn=auc_metric)
    
    # trainer = Trainer(
    #     accelerator='gpu',
    #     devices=[args.gpu],
    #     max_epochs=200,
    #     num_sanity_val_steps=0
    # )
    # trainer.fit(pretrain, train_dataloader, val_dataloader)

    run_description = f"max_features_60_{pretrained_model_name}_lr_{str(lr)}_prior_info_individual_backbone_use_entropy"
    logger = TensorBoardLogger("logs", name=f"{run_description}")
    checkpoint_callback = best_hard_callback = ModelCheckpoint(
                save_top_k=1,
                monitor='Perf Val/Mean',
                mode='max',
                filename='best_val_perfomance_model',
                verbose=False
            )

    # Jointly train predictor and value networks
    greedy_cmi_estimator = CMIEstimatorPrior(value_network,
                                             predictor,
                                             mask_layer,
                                             lr=lr,
                                             min_lr=min_lr,
                                             max_features=60,
                                             eps=0.05,
                                             loss_fn=nn.CrossEntropyLoss(reduction='none'),
                                             val_loss_fn=AUROC(task='multiclass', num_classes=2),
                                             eps_decay=0.2,
                                             eps_steps=10,
                                             patience=3,
                                             feature_costs=None)

    trainer = Trainer(
                accelerator='gpu',
                devices=[args.gpu],
                max_epochs=250,
                precision=16,
                logger=logger,
                num_sanity_val_steps=0,
                callbacks=[checkpoint_callback]
            )

    trainer.fit(greedy_cmi_estimator, train_dataloader, val_dataloader)