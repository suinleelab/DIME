import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchmetrics import AUROC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from os import path
import sys
import pandas as pd
import feature_groups
sys.path.append('../')
from data_utils import DenseDatasetSelected, get_group_matrix, get_xy, MaskLayerGrouped, data_split
sys.path.append('../../')
from models.masking_pretrainer import MaskingPretrainer#, GreedyCMIEstimator
from models.greedy_model_pl import GreedyCMIEstimatorPL
from utils import accuracy, auc, normalize
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_feature_costs', default=False, action="store_true")
parser.add_argument('--num_trials', type=int, default=5)

if __name__ == '__main__':
    intub_feature_names = feature_groups.intub_feature_names
    intub_feature_groups = feature_groups.intub_feature_groups

    # Parse args
    args = parser.parse_args()
    # cols_to_drop = [str(x) for x in range(len(intub_feature_names)) if str(x) not in ['98', '104', '107', '108', '109', '110']]
    cols_to_drop = []
    if cols_to_drop is not None:
        intub_feature_names = [item for item in intub_feature_names if str(intub_feature_names.index(item)) not in cols_to_drop]

    # Load dataset
    dataset = DenseDatasetSelected('data/intub.csv', cols_to_drop=cols_to_drop)
    d_in = dataset.X.shape[1]  # 121
    d_out = len(np.unique(dataset.Y))  # 2

    # Get features and groups
    feature_groups_dict, feature_groups_mask = get_group_matrix(intub_feature_names, intub_feature_groups)
    num_groups = len(feature_groups_mask)  # 45
    print("Num groups=", num_groups)
    print("Num features=", d_in)

    print(feature_groups_dict.keys())

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(0))

    daataset_dict = dict(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    f = open('./data/dataset.pkl', "wb", pickle.HIGHEST_PROTOCOL)
    pickle.dump(daataset_dict, f)
    
    # train_dataset.X
    print(f'Train samples = {len(train_dataset)}, val samples = {len(val_dataset)}, test samples = {len(test_dataset)}')

    # Find mean/variance for normalizing
    x, y = get_xy(train_dataset)
    mean = np.mean(x, axis=0)
    std = np.std(y, axis=0)

    # Normalize via the original dataset
    dataset.X = dataset.X - mean

    # Setup
    device = torch.device('cuda', args.gpu)

    # Set up architecture
    hidden = 128
    dropout = 0.3

    for trial in range(args.num_trials):
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

        # Tie weights
        value_network[0] = predictor[0]
        value_network[3] = predictor[3]

        use_feature_costs = False
        if args.use_feature_costs:
            use_feature_costs = True
        
        mask_layer = MaskLayerGrouped(append=True, group_matrix=torch.tensor(feature_groups_mask))
        trained_predictor_name = f"pretrained_predictor_pl_feature_cost_{use_feature_costs}_trial_{trial}.pth"
        if path.exists(f"results/{trained_predictor_name}"):
            # Load pretrained predictor
            print("Loading Pretrained Predictor")
            print("-"*8)
            predictor.load_state_dict(torch.load(f"results/{trained_predictor_name}"))
        else:
            # Pretrain predictor
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            pretrain.fit(train_dataset,
                        val_dataset,
                        mbsize=128,
                        lr=1e-3,
                        nepochs=100,
                        loss_fn=nn.CrossEntropyLoss(),
                        val_loss_fn=auc,
                        val_loss_mode='max',
                        patience=5,
                        verbose=True,
                        trained_predictor_name=trained_predictor_name)

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

        feature_costs = None

        if args.use_feature_costs:
            feature_cost_df = pd.read_csv("data/feature_list_intub-nw.csv")
            feature_costs = [feature_cost_df[feature_cost_df['Feature Name'] == feature]['Cost (Hours)'].item() for feature in list(feature_groups_dict.keys())]


        print(feature_costs)

        # Jointly train value network and predictor
        print("Training CMI estimator")
        print("-"*8)
        run_description = f"max_features_40_eps_0.0_with_decay_rate_0.2_use_entropy_feature_cost_{use_feature_costs}_trial_{trial}"
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
                                        min_lr=1e-6,
                                        max_features=40,
                                        eps=0.0,
                                        loss_fn=nn.CrossEntropyLoss(reduction='none'),
                                        val_loss_fn=auc,
                                        eps_decay=True,
                                        eps_decay_rate=0.2,
                                        patience=5,
                                        feature_costs=feature_costs,
                                        use_entropy=True)

        trainer = Trainer(
                    accelerator='gpu',
                    devices=[args.gpu],
                    max_epochs=100,
                    precision=16,
                    logger=logger,
                    num_sanity_val_steps=0,
                    callbacks=[checkpoint_callback],
                    log_every_n_steps=10
                )

        trainer.fit(greedy_cmi_estimator, train_dataloader, val_dataloader)
    
