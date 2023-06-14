# Traing full input
import sys
from torchvision import transforms
import argparse
sys.path.append('../')
from data_utils import MaskLayerGaussian, MaskLayer2d, HistopathologyDownsampledEdgeDataset
sys.path.append('../../')
import timm
from utils import accuracy, auc, normalize
from torchvision.datasets import ImageFolder
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dime.resnet_imagenet import resnet18

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--backbone', type=str, 
                                default='vit', 
                                choices=['vit', 'resnet'], 
                                help="Backbone used to train the network")


if __name__ == "__main__":
    run_description = f"vit_no_mask"
    args = parser.parse_args()
    writer = SummaryWriter(filename_suffix=run_description)
    norm_constants = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    image_size = 224
    device = torch.device('cuda', args.gpu)
    dataset_path = "/projects/<labname>/<username>/ImageNet100"

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

    # Get Datasets
    train_dataset_train_transforms = ImageFolder(dataset_path+'/train.X1', transforms_train)
    train_dataset_all_len = len(train_dataset_train_transforms)

    # Get train and val indices
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(train_dataset_all_len, size=int(train_dataset_all_len*0.1), replace=False))
    train_inds = np.setdiff1d(np.arange(train_dataset_all_len), val_inds)

    train_dataset = torch.utils.data.Subset(train_dataset_train_transforms, train_inds)

    train_dataset_test_transforms = ImageFolder(dataset_path+'/train.X1', transforms_test)
    val_dataset = torch.utils.data.Subset(train_dataset_test_transforms, val_inds)

    test_dataset = ImageFolder(dataset_path+'/val.X', transforms_test)

    # Prepare dataloaders.
    mbsize = 32
    train_dataloader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, pin_memory=True,
                            drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=mbsize, pin_memory=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=mbsize, pin_memory=True, drop_last=True, num_workers=4)

    d_in = image_size * image_size
    d_out = 100


    device = torch.device('cuda:1')
    if args.backbone == 'vit':
        model = timm.create_model("vit_small_patch16_224", pretrained=True)
        model.head = torch.nn.Linear(model.embed_dim, 100)
    else:
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.expansion * 512, 100)

    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.2, patience=5,
            min_lr=1e-8, verbose=True)

    for epoch in range(250):
        model.train()
        train_batch_loss = 0
        val_batch_loss = 0
        val_pred_list = []
        val_y_list = []

        for i, batch in enumerate(tqdm(train_dataloader)):
            if len(batch) == 2:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            train_loss = criterion(pred, y)
            train_batch_loss += train_loss.item()
            train_loss.backward()
            opt.step()
            model.zero_grad()
        
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader)):
                if len(batch) == 2:
                    x, y = batch
                x = x.to(device)
                y = y.to(device)
            
                pred = model(x)
                val_loss = criterion(pred, y)
                val_batch_loss += val_loss.item()
                val_pred_list.append(pred.cpu())
                val_y_list.append(y.cpu())

            scheduler.step(val_batch_loss/len(val_dataloader))
        
        writer.add_scalar("Loss/Train", train_batch_loss/len(train_dataloader), epoch)
        writer.add_scalar("Loss/Val", val_batch_loss/len(val_dataloader), epoch)
        writer.add_scalar("Performance/Val", accuracy(torch.cat(val_y_list), torch.cat(val_pred_list)), epoch)

        print(f"Epoch: {epoch}, Train Loss: {train_batch_loss/len(train_dataloader)}, Val Loss: {val_batch_loss/len(val_dataloader)}, Val Performance: {accuracy(torch.cat(val_y_list), torch.cat(val_pred_list))}")