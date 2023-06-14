import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
import pandas as pd
from os import path
import feature_groups
import sys
sys.path.append('../')
from data_utils import ROSMAPDataset, get_group_matrix, get_xy, MaskLayerGrouped, data_split, get_mlp_network
sys.path.append('../../')
from dime.masking_pretrainer import MaskingPretrainer
from baseline_models.base_model import BaseModel
from utils import accuracy, auc, normalize, StaticMaskLayer1d, MaskLayer, ConcreteMask, get_confidence
from torchvision import transforms
from torchmetrics import Accuracy, AUROC
from torchvision.datasets import MNIST
from baselines import eddi, pvae, iterative, dfs, cae
import torch.optim as optim
from tqdm import tqdm

#from baselines import EDDI, PVAE

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method', type=str, default='cae',
                    choices=['cae', 'eddi', 'dfs', 'fully_supervised'])
parser.add_argument('--use_apoe', default=False, action="store_true")
parser.add_argument('--use_feature_costs', default=False, action="store_true")
parser.add_argument('--num_trials', type=int, default=5)

rosmap_feature_names = feature_groups.rosmap_feature_names
rosmap_feature_groups = feature_groups.rosmap_feature_groups

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    auc_metric = AUROC(task='multiclass', num_classes=2)
    num_trials = args.num_trials


    # cols_to_drop = [str(x) for x in range(len(fib_feature_names)) if str(x) not in ['98', '104', '107', '108', '109', '110']]
    cols_to_drop = []
    if cols_to_drop is not None:
        rosmap_feature_names = [item for item in rosmap_feature_names if str(rosmap_feature_names.index(item)) not in cols_to_drop]
    
    
    device = torch.device('cuda', args.gpu)

    # Load dataset
    train_dataset = ROSMAPDataset('./data', split='train', cols_to_drop=cols_to_drop, use_apoe=args.use_apoe)
    d_in = train_dataset.X.shape[1]  
    d_out = len(np.unique(train_dataset.Y))

    val_dataset = ROSMAPDataset('./data', split='val', cols_to_drop=cols_to_drop, use_apoe=args.use_apoe)
    test_dataset = ROSMAPDataset('./data', split='test', cols_to_drop=cols_to_drop, use_apoe=args.use_apoe)

    if not args.use_apoe:
        rosmap_feature_names = [f for f in rosmap_feature_names if f not in ['apoe4_1copy','apoe4_2copies']]
    
    feature_groups_dict, feature_groups_mask = get_group_matrix(rosmap_feature_names, rosmap_feature_groups)
    feature_group_indices = {i : key for i, key in enumerate(feature_groups_dict.keys())}
    feat_to_ind = {key: i for i, key in enumerate(rosmap_feature_names)}

    num_groups = len(feature_groups_mask)  # 45
    print("Num groups=", num_groups)
    print("Num features=", d_in)
    
    # train_dataset.X
    print(f'Train samples = {len(train_dataset)}, val samples = {len(val_dataset)}, test samples = {len(test_dataset)}')

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

    num_features = list(range(1, 12, 1))
    mask_layer = MaskLayerGrouped(append=True, group_matrix=torch.tensor(feature_groups_mask))

    feature_costs = None
    if args.use_feature_costs:
        df = pd.read_csv("./data/rosmap_feature_costs.csv", header=None)
        if args.use_apoe:
            feature_costs = df[1].tolist()
        else:
            feature_costs = df[~df[0].isin(['apoe4_1copy','apoe4_2copies'])][1].tolist()

    for trial in range(5, 7):

        results_dict = {
            'acc': {},
            'features': {}
        }

        if args.method == 'cae':
            num_restarts = 5
            for num in num_features:
                # Train model with differentiable feature selection.
                model = get_mlp_network(d_in, d_out)
                selector_layer = ConcreteMask(num_groups, num, torch.tensor(feature_groups_mask))
                diff_selector = cae.DifferentiableSelector(model, selector_layer).to(device)
                diff_selector.fit(
                    train_dataloader,
                    val_dataloader,
                    lr=1e-3,
                    nepochs=250,
                    loss_fn=nn.CrossEntropyLoss(),
                    patience=5,
                    verbose=True)

                # Extract top featuresd.
                logits = selector_layer.logits.cpu().data.numpy()
                selected_groups = np.sort(logits.argmax(axis=1))
                if len(np.unique(selected_groups)) != num:
                    print(f'{len(np.unique(selected_groups))} selected instead of {num}, appending extras')
                    num_extras = num - len(np.unique(selected_groups))
                    remaining_groups = np.setdiff1d(np.arange(num_groups), selected_groups)
                    selected_groups = np.sort(np.concatenate([np.unique(selected_groups), remaining_groups[:num_extras]]))
                
                # selected_groups = [0]
                print(f"selected_groups={selected_groups}")
                selected_features = []
                for i in range(num):
                    selected_features += map(lambda x: feat_to_ind[x], feature_groups_dict[feature_group_indices[selected_groups[i]]])
                print(selected_features)

                # Prepare module to mask all but top features
                inds = torch.tensor(np.isin(np.arange(d_in), selected_features), device=device)
                mask_layer = StaticMaskLayer1d(inds)

                best_loss = np.inf
                for _ in range(num_restarts):
                    # Train model.
                    model = nn.Sequential(mask_layer, get_mlp_network(len(selected_features), d_out))
                    basemodel = BaseModel(model).to(device)
                    basemodel.fit(
                        train_dataloader,
                        val_dataloader,
                        lr=1e-3,
                        nepochs=250,
                        loss_fn=nn.CrossEntropyLoss(),
                        verbose=True)

                    # Check if best.
                    val_loss = basemodel.evaluate(val_dataloader, nn.CrossEntropyLoss())
                    if val_loss < best_loss:
                        best_model = basemodel
                        best_loss = val_loss

                # Evaluate using best model.
                acc = best_model.evaluate(test_dataloader, auc_metric)
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            print(results_dict)
            with open(f'results/rosmap_{args.method}_trial_{trial}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)

        if args.method == 'eddi':
            # Train PVAE.
            bottleneck = 16
            hidden = 128
            dropout = 0.3
            encoder = get_mlp_network(d_in + num_groups, bottleneck * 2)
            decoder = get_mlp_network(bottleneck, d_in)  

            pv = pvae.PVAE(encoder, decoder, mask_layer, 128, 'gaussian').to(device)
            pv.fit(
                train_dataloader,
                val_dataloader,
                lr=1e-3,
                nepochs=100,
                verbose=False)
            
            # Train masked predictor.
            model = get_mlp_network(d_in + num_groups, d_out)
            sampler = None
            # if trial == 0:
            sampler = iterative.UniformSampler(get_xy(train_dataset)[0])  # TODO don't actually need sampler
            iterative_selector = iterative.IterativeSelector(model, mask_layer, sampler).to(device)
            iterative_selector.fit(
                train_dataloader,
                val_dataloader,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)
            
            # Set up EDDI feature selection object.
            eddi_selector = eddi.EDDI(pv, model, mask_layer, feature_costs=feature_costs).to(device)
            
            # Evaluate.
            metrics_dict, cost_dict = eddi_selector.evaluate_multiple(test_dataloader, num_features, auc, verbose=False)
            for num in num_features:
                acc = metrics_dict[num]
                results_dict['acc'][num] = acc
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            print(results_dict)
            print(cost_dict)
            with open(f'results/rosmap_{args.method}_trial_{trial+1}_feature_costs_{args.use_feature_costs}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)
            
            with open(f'results/rosmap_costs_{args.method}_trial_{trial+1}_feature_costs_{args.use_feature_costs}.pkl', 'wb') as f:
                pickle.dump(cost_dict, f)
        
        if args.method == 'dfs':
            max_features = 15

            # Prepare networks.
            predictor = get_mlp_network(d_in + num_groups, d_out)
            selector = get_mlp_network(d_in + num_groups, num_groups)

            # Pretrain predictor
            pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            pretrain.fit(
                train_dataset,
                val_dataset,
                128,
                lr=1e-3,
                nepochs=100,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)

            # Train selector and predictor jointly.
            gdfs = dfs.GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
            gdfs.fit(
                train_dataloader,
                val_dataloader,
                lr=1e-3,
                nepochs=100,
                max_features=max_features,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)

            # Evaluate.
            for num in num_features:
                acc = gdfs.evaluate(test_dataloader, num, auc_metric)
                results_dict['acc'][num] = acc
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            with open(f'results/rosmap_{args.method}_trial_{trial}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)
                
            # Save model
            gdfs.cpu()
            torch.save(gdfs, f'results/rosmap_{args.method}_trial_{trial}.pt')
        
        # Train with full input
        if args.method == 'fully_supervised':
            model  = get_mlp_network(d_in, d_out).to(device)
            opt = optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=0.2, patience=5,
                    min_lr=1e-5, verbose=True)
            
            num_bad_epochs = 0
            early_stopping_epochs = 6

            for epoch in range(100):
                model.train()
                train_batch_loss = 0
                val_batch_loss = 0
                val_pred_list = []
                val_y_list = []

                for i, (x, y) in enumerate(tqdm(train_dataloader)):
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
                    for i, (x, y) in enumerate(tqdm(val_dataloader)):
                        x = x.to(device)
                        y = y.to(device)
                    
                        pred = model(x)
                        val_loss = criterion(pred, y)
                        val_batch_loss += val_loss.item()
                        val_pred_list.append(pred.cpu())
                        val_y_list.append(y.cpu())

                    scheduler.step(val_batch_loss/len(val_dataloader))
                    val_loss = val_batch_loss/len(val_dataloader)
                    # Check if best model.
                    if val_loss == scheduler.best:
                        # best_model = deepcopy(model)
                        num_bad_epochs = 0
                    else:
                        num_bad_epochs += 1
                        
                    # Early stopping.
                    if num_bad_epochs > early_stopping_epochs:
                        print(f'Stopping early at epoch {epoch+1}')
                        break

                print(f"Epoch: {epoch}, Train Loss: {train_batch_loss/len(train_dataloader)}, Val Loss: {val_batch_loss/len(val_dataloader)}, Val Performance: {auc(torch.cat(val_pred_list), torch.cat(val_y_list))}")
            

            print("Evaluating on test set")
            
            model.eval()
            confidence_list = []
            test_pred_list = []
            test_y_list = []
            for i, (x, y) in enumerate(tqdm(test_dataloader)):
                x = x.to(device)
                y = y.to(device)
            
                pred = model(x)
                test_pred_list.append(pred.cpu())
                test_y_list.append(y.cpu())

                confidence_list.append(get_confidence(pred.cpu()))
            
            print(f"Test Performance:{auc(torch.cat(test_pred_list), torch.cat(test_y_list))}")
            with open('confidence.npy', 'wb') as f:
                np.save(f, np.array(torch.cat(confidence_list).detach().numpy()))
