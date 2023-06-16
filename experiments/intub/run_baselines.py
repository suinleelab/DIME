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
from dime.data_utils import DenseDatasetSelected, get_group_matrix, get_xy, MaskLayerGrouped, data_split, get_mlp_network
from dime.masking_pretrainer import MaskingPretrainer
from dime.utils import accuracy, auc, normalize, StaticMaskLayer1d, MaskLayer, ConcreteMask, get_confidence
import sys
sys.path.append('../')
from baselines import  eddi, pvae, iterative, dfs, cae
sys.path.append('../../')
from baseline_models.base_model import BaseModel
import torch.optim as optim
from tqdm import tqdm

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method', type=str, default='cae',
                    choices=['cae', 'eddi', 'dfs', 'fully_supervised'])
parser.add_argument('--use_feature_costs', default=False, action="store_true")
parser.add_argument('--num_trials', type=int, default=5)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    device = torch.device('cuda', args.gpu)
    intub_feature_names = feature_groups.intub_feature_names
    intub_feature_groups = feature_groups.intub_feature_groups

    num_trials = args.num_trials

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
    feature_group_indices = {i : key for i, key in enumerate(feature_groups_dict.keys())}
    feat_to_ind = {key: i for i, key in enumerate(intub_feature_names)}

    num_groups = len(feature_groups_mask)  # 45
    print("Num groups=", num_groups)
    print("Num features=", d_in)

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(0))
    daataset_dict = dict(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    f = open('./data/dataset.pkl', "wb", pickle.HIGHEST_PROTOCOL)
    pickle.dump(daataset_dict, f)
    
    print(f'Train samples = {len(train_dataset)}, val samples = {len(val_dataset)}, test samples = {len(test_dataset)}')

    # Find mean/variance for normalizing
    x, y = get_xy(train_dataset)
    mean = np.mean(x, axis=0)
    std = np.clip(np.std(x, axis=0), 1e-3, None)

    # Normalize via the original dataset
    if args.method == 'eddi':
        dataset.X = (dataset.X - mean)/std
    else:
        dataset.X = dataset.X - mean

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

    mask_layer = MaskLayerGrouped(append=True, group_matrix=torch.tensor(feature_groups_mask))
    num_features = [1, 3, 5, 10, 15, 20, 25]
    use_feature_costs = False
    feature_costs = None
    if args.use_feature_costs:
        feature_cost_df = pd.read_csv("data/feature_list_intub-nw.csv")
        feature_costs = [feature_cost_df[feature_cost_df['Feature Name'] == feature]['Cost (Hours)'].item() for feature in list(feature_groups_dict.keys())]
        use_feature_costs = True

    for trial in range(num_trials):

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

                # Extract top features
                logits = selector_layer.logits.cpu().data.numpy()
                selected_groups = np.sort(logits.argmax(axis=1))
                if len(np.unique(selected_groups)) != num:
                    print(f'{len(np.unique(selected_groups))} selected instead of {num}, appending extras')
                    num_extras = num - len(np.unique(selected_groups))
                    remaining_groups = np.setdiff1d(np.arange(num_groups), selected_groups)
                    selected_groups = np.sort(np.concatenate([np.unique(selected_groups), remaining_groups[:num_extras]]))
                
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
                acc = best_model.evaluate(test_dataloader, auc)
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            print(results_dict)
            with open(f'results/intub_{args.method}_trial_{trial}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)

        if args.method == 'eddi':
            # Train PVAE.
            bottleneck = 16
            hidden = 128
            dropout = 0.3
            encoder = get_mlp_network(d_in + num_groups, bottleneck * 2)
            decoder = get_mlp_network(bottleneck, d_in)  

            pv = pvae.PVAE(encoder, decoder, mask_layer, 20, 'gaussian').to(device)
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
            with open(f'results/intub_{args.method}_trial_{trial}_feature_costs_{use_feature_costs}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)
            
            with open(f'results/intub_costs_{args.method}_trial_{trial}_feature_costs_{use_feature_costs}.pkl', 'wb') as f:
                pickle.dump(cost_dict, f)
        
        if args.method == 'dfs':
            max_features = 35

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
                val_loss_fn=auc,
                val_loss_mode='max',
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
                acc = gdfs.evaluate(test_dataloader, num, auc)
                results_dict['acc'][num] = acc
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            with open(f'results/intub_{args.method}_trial_{trial}.pkl', 'wb') as f:
                pickle.dump(results_dict, f)
                
            # Save model
            gdfs.cpu()
            torch.save(gdfs, f'results/intub_{args.method}_trial_{trial}.pt')
        
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
                
                # writer.add_scalar("Loss/Train", train_batch_loss/len(train_dataloader), epoch)
                # writer.add_scalar("Loss/Val", val_batch_loss/len(val_dataloader), epoch)
                # writer.add_scalar("Performance/Val", auc(torch.cat(val_y_list), torch.cat(val_pred_list)), epoch)

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
