import torch
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dime import MaskingPretrainerPL, CMIEstimator, MaskLayer
from torch.utils.data import DataLoader

# Set up command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_trials', type=int, default=5)

if __name__ == '__main__':
    acc_metric = Accuracy(task='multiclass', num_classes=10)

    # Parse args.
    args = parser.parse_args()
    num_trials = args.num_trials

    # Load train dataset, split into train/val.
    mnist_dataset = MNIST('/tmp/mnist/', download=True, train=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]))
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(len(mnist_dataset), size=10000, replace=False))
    train_inds = np.setdiff1d(np.arange(len(mnist_dataset)), val_inds)
    train_dataset = torch.utils.data.Subset(mnist_dataset, train_inds)
    val_dataset = torch.utils.data.Subset(mnist_dataset, val_inds)
    d_in = 784
    d_out = 10

    # Load test dataset.
    test_dataset = MNIST('/tmp/mnist/', download=True, train=False,
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]))
    device = torch.device('cuda', args.gpu)

    # Set up data loaders.
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True,
        drop_last=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    # Set up architecture.
    hidden = 512
    dropout = 0.3

    for trial in range(num_trials):
        # For predicting response variable.
        predictor = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out))

        # For predicting CMI.
        value_network = nn.Sequential(
            nn.Linear(d_in * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_in))

        # For masking unobserved features.
        mask_layer = MaskLayer(mask_size=d_in, append=True)

        # Pretrain predictor.
        print('Pretraining predictor')
        print('-'*8)

        pretrain = MaskingPretrainerPL(
            predictor,
            mask_layer,
            lr=1e-3,
            loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=acc_metric)
        trainer = Trainer(
            accelerator='gpu',
            devices=[args.gpu],
            max_epochs=200,
            num_sanity_val_steps=0
        )
        trainer.fit(pretrain, train_dataloader, val_dataloader)

        # Joint training.
        print('Training CMI estimator')
        print('-'*8)

        greedy_cmi_estimator = CMIEstimator(
            value_network,
            predictor,
            mask_layer,
            lr=1e-3,
            max_features=50,
            eps=0.05,
            loss_fn=nn.CrossEntropyLoss(reduction='none'),
            val_loss_fn=acc_metric,
            eps_steps=2,
            patience=5
        )
        run_description = f'max_features_50_eps_0.05_with_decay_rate_0.2_save_best_loss_with_entropy_fix_trial_{trial}'
        logger = TensorBoardLogger('logs', name=f'{run_description}')
        checkpoint_callback_loss = ModelCheckpoint(
            save_top_k=1,
            monitor='Loss Val/Mean',
            mode='min',
            filename='best_val_loss_model',
            verbose=False
        )
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
