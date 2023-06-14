import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
import sys
sys.path.append('../../')
from dime.greedy_models import MaskingPretrainer, GreedyCMIEstimator
from utils import MaskLayer, accuracy, generate_2d_gaussion_cost, generate_pixel_based_cost, selection_with_lamda, selection_without_lamda
from torch.utils.data import DataLoader
import os.path
from os import path

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Load train dataset, split into train/val
    mnist_dataset = MNIST('/tmp/mnist/', download=True, train=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]))
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(len(mnist_dataset), size=10000, replace=False))
    train_inds = np.setdiff1d(np.arange(len(mnist_dataset)), val_inds)
    train_dataset = torch.utils.data.Subset(mnist_dataset, train_inds)
    val_dataset = torch.utils.data.Subset(mnist_dataset, val_inds)
    d_in = 784
    d_out = 10
    
    # Load test dataset
    test_dataset = MNIST('/tmp/mnist/', download=True, train=False,
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]))
    device = torch.device('cuda', args.gpu)
    print(device)

    # Set up architecture
    hidden = 512
    dropout = 0.3

    # Outcome Predictor
    predictor = nn.Sequential(
        nn.Linear(d_in * 2, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out))

    # CMI Predictor
    value_network = nn.Sequential(
        nn.Linear(d_in * 2, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_in),
        nn.Softplus())

    # Tie weights
    value_network[0] = predictor[0]
    value_network[3] = predictor[3]
    mask_layer = MaskLayer(append=True)
    
    if path.exists("results/pretrained_predictor.pth"):
        # Load pretrained predictor
        print("Loading Pretrained Predictor")
        print("-"*8)
        predictor.load_state_dict(torch.load("results/pretrained_predictor.pth"))
    else:
        # Pretrain predictor
        print("Pretraining Predictor")
        print("-"*8)
        pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
        pretrain.fit(train_dataset,
                    val_dataset,
                    mbsize=128,
                    lr=1e-3,
                    nepochs=200,
                    loss_fn=nn.CrossEntropyLoss(),
                    verbose=True)

    greedy_cmi_estimator = GreedyCMIEstimator(value_network, predictor, mask_layer).to(device)

    # Set up data loaders.
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True,
        drop_last=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, pin_memory=True,
        drop_last=True, num_workers=4)
        
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, pin_memory=True,
        drop_last=True, num_workers=4)

    # Train CMI estimator
    print("Training CMI estimator")
    print("-"*8)
    run_description = "max_features_35_eps_0.05_with_decay_rate_0.2_use_entropy_with_uniform_cost"

    # Jointly train value network and predictor
    greedy_cmi_estimator.fit(train_dataloader, 
                            val_dataloader,
                            lr=1e-3,
                            nepochs=200,
                            max_features=35,
                            eps=0.05,
                            loss_fn=nn.CrossEntropyLoss(reduction='none'),
                            val_loss_fn=accuracy,
                            tensorboard_file_name_suffix=run_description,
                            eps_decay=True,
                            eps_decay_rate=0.2,
                            patience=3,
                            feature_costs=None,
                            use_entropy=True)

    # Save model
    greedy_cmi_estimator.cpu()
    torch.save(greedy_cmi_estimator, f'results/greedy_cmi_estimator_trained_{run_description}.pt')