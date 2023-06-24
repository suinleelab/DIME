import torch
import feature_groups
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import AUROC
from torch.utils.data import DataLoader
import pandas as pd
from dime.data_utils import ROSMAPDataset, get_group_matrix
from dime.utils import MaskLayerGrouped
from dime import MaskingPretrainer
from dime import CMIEstimator
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_apoe', default=False, action="store_true")
parser.add_argument('--use_feature_costs', default=False, action="store_true")
parser.add_argument('--num_trials', type=int, default=5)

rosmap_feature_names = feature_groups.rosmap_feature_names
rosmap_feature_groups = feature_groups.rosmap_feature_groups

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    num_trials = args.num_trials
    # cols_to_drop = [str(x) for x in range(len(fib_feature_names)) if str(x) not in ['98', '104', '107', '108', '109', '110']]
    cols_to_drop = []
    if cols_to_drop is not None:
        rosmap_feature_names = [item for item in rosmap_feature_names if str(rosmap_feature_names.index(item)) 
                                not in cols_to_drop]
        
    # Load dataset
    train_dataset = ROSMAPDataset('./data', split='train', cols_to_drop=cols_to_drop, use_apoe=args.use_apoe)
    d_in = train_dataset.X.shape[1]
    d_out = len(np.unique(train_dataset.Y))

    val_dataset = ROSMAPDataset('./data', split='val', cols_to_drop=cols_to_drop, use_apoe=args.use_apoe)
    test_dataset = ROSMAPDataset('./data', split='test', cols_to_drop=cols_to_drop, use_apoe=args.use_apoe)

    if not args.use_apoe:
        rosmap_feature_names = [f for f in rosmap_feature_names if f not in ['apoe4_1copy','apoe4_2copies']]

    if args.use_feature_costs:
        df = pd.read_csv("./data/rosmap_feature_costs.csv", header=None)
        if args.use_apoe:
            feature_costs = df[1].tolist()
        else:
            feature_costs = df[~df[0].isin(['apoe4_1copy', 'apoe4_2copies'])][1].tolist()

    # Get features and groups
    feature_groups_dict, feature_groups_mask = get_group_matrix(rosmap_feature_names, rosmap_feature_groups)
    num_groups = len(feature_groups_mask) 
    print("Num groups=", num_groups)
    print("Num features=", d_in)
    
    print(f'Train samples = {len(train_dataset)}, val samples = {len(val_dataset)}, test samples = {len(test_dataset)}')

    # Setup
    max_features = 30
    device = torch.device('cuda', args.gpu)

    # Set up architecture
    hidden = 128
    dropout = 0.3

    print(f"args.use_apoe={args.use_apoe}")

    for trial_num in range(num_trials):
        # Predictor
        predictor = nn.Sequential(
            nn.Linear(d_in + num_groups, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out))

        # CMI Predictor
        value_network = nn.Sequential(
            nn.Linear(d_in + num_groups, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_groups),
            nn.Sigmoid())

        mask_layer = MaskLayerGrouped(append=True, group_matrix=torch.tensor(feature_groups_mask))

        # Set up data loaders.
        train_dataloader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, pin_memory=True,
            drop_last=True, num_workers=4)
            
        val_dataloader = DataLoader(
            val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
            
        test_dataloader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
        
        pretrain = MaskingPretrainer(
            predictor,
            mask_layer,
            lr=1e-3,
            loss_fn=nn.CrossEntropyLoss(),
            val_loss_fn=AUROC(task='multiclass', num_classes=2))
        
        trainer = Trainer(
            accelerator='gpu',
            devices=[args.gpu],
            max_epochs=200,
            num_sanity_val_steps=0
        )
        trainer.fit(pretrain, train_dataloader, val_dataloader)

        # Train CMI estimator
        print("Training CMI estimator")
        print("-"*8)
        run_description = f"max_feature_15_apoe_{args.use_apoe}_ep_0.05_decay_0.2_{args.use_feature_costs}_trial_{trial_num}"
        logger = TensorBoardLogger("logs", name=f"{run_description}")

        checkpoint_callback_perf = ModelCheckpoint(
                    save_top_k=1,
                    monitor='Perf Val/Mean',
                    mode='max',
                    filename='best_val_perfomance_model',
                    verbose=False
                )
        
        checkpoint_callback_loss = ModelCheckpoint(
                    save_top_k=1,
                    monitor='Loss Val/Mean',
                    mode='min',
                    filename='best_val_loss_model',
                    verbose=False
                )

        greedy_cmi_estimator = CMIEstimator(value_network,
                                            predictor,
                                            mask_layer,
                                            lr=1e-3,
                                            min_lr=1e-6,
                                            max_features=15,
                                            eps=0.05,
                                            loss_fn=nn.CrossEntropyLoss(reduction='none'),
                                            val_loss_fn=AUROC(task='multiclass', num_classes=2),
                                            eps_decay=0.2,
                                            eps_steps=10,
                                            patience=5,
                                            feature_costs=None)

        trainer = Trainer(
                    accelerator='gpu',
                    devices=[args.gpu],
                    max_epochs=250,
                    precision=16,
                    logger=logger,
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback_loss],
                    log_every_n_steps=10
                )

        trainer.fit(greedy_cmi_estimator, train_dataloader, val_dataloader)