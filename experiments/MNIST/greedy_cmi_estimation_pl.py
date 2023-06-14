import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
sys.path.append('../../')
from dime.masking_pretrainer import MaskingPretrainer
from dime.greedy_model_pl import GreedyCMIEstimatorPL
from utils import MaskLayer, accuracy, generate_2d_gaussion_cost, generate_pixel_based_cost, selection_with_lamda, selection_without_lamda
from torch.utils.data import DataLoader
import os.path
from os import path

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=5)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    num_trials = args.num_trials

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

    for trial in range(num_trials):
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
            nn.Sigmoid())

        # Tie weights
        value_network[0] = predictor[0]
        value_network[3] = predictor[3]
        mask_layer = MaskLayer(append=True, mask_size=d_in)
        
        if path.exists(f"results/pretrained_predictor_trial_save_best_loss_{trial}.pth"):
            # Load pretrained predictor
            print("Loading Pretrained Predictor")
            print("-"*8)
            predictor.load_state_dict(torch.load(f"results/pretrained_predictor_trial_{trial}.pth"))
        else:
            # Pretrain predictor
            print("Pretraining Predictor")
            print("-"*8)
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            pretrain.fit(train_dataset,
                        val_dataset,
                        mbsize=128,
                        lr=1e-3,
                        nepochs=100,
                        val_loss_fn=accuracy,
                        val_loss_mode='max',
                        loss_fn=nn.CrossEntropyLoss(),
                        verbose=True)
        
        run_description = f"max_features_50_eps_0.05_with_decay_rate_0.2_save_best_loss_with_entropy_fix_trial_{trial}"
        logger = TensorBoardLogger("logs", name=f"{run_description}")
        checkpoint_callback = ModelCheckpoint(
                    save_top_k=1,
                    monitor='Performance_Val',
                    mode='max',
                    filename='best_val_perfomance_model',
                    verbose=False
                )
        
        checkpoint_callback_loss = ModelCheckpoint(
                    save_top_k=1,
                    monitor='Predictor Loss Val',
                    mode='min',
                    filename='best_val_perfomance_model',
                    verbose=False
                )

       

        greedy_cmi_estimator = GreedyCMIEstimatorPL(value_network, predictor, mask_layer,
                                        lr=1e-3,
                                        max_features=50,
                                        eps=0.05,
                                        loss_fn=nn.CrossEntropyLoss(reduction='none'),
                                        val_loss_fn=accuracy,
                                        eps_decay=True,
                                        eps_decay_rate=0.2,
                                        patience=5,
                                        feature_costs=None,
                                        use_entropy=True)

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

        # Jointly train value network and predictor
        trainer = Trainer(
                    accelerator='gpu',
                    devices=[args.gpu],
                    max_epochs=200,
                    precision=16,
                    logger=logger,
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback_loss]
                )

        trainer.fit(greedy_cmi_estimator, train_dataloader, val_dataloader)