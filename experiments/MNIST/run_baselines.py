import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from dime import MaskingPretrainer
from dime.utils import StaticMaskLayer1d, MaskLayer, ConcreteMask, get_confidence, get_mlp_network
from dime.data_utils import get_xy
from torchvision import transforms
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from pytorch_lightning import Trainer
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.append('../')
from baselines import eddi, pvae, iterative, dfs, cae
sys.path.append('../../')
from baseline_models.base_model import BaseModel
import time

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method', type=str, default='cae',
                    choices=['cae', 'eddi', 'dfs', 'fully_supervised'])
parser.add_argument('--num_trials', type=int, default=5)


if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    acc_metric = Accuracy(task='multiclass', num_classes=10)
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
    print(f"test dataset length={test_dataset.__len__()}")
    device = torch.device('cuda', args.gpu)

    # Set up data loaders.
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True,
        drop_last=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
    
    # test_dataset = torch.utils.data.Subset(test_dataset, range(0, 256))
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

    num_features = [3, 5, 10, 15, 20, 25]
    max_features = 35
    mbsize = 128
    
    for trial in range(num_trials):

        results_dict = {
            'acc': {},
            'features': {}
        }

        if args.method == 'cae':
            num_restarts = 3

            for num in num_features:
                # Train model with differentiable feature selection.
                model = get_mlp_network(d_in, d_out)
                selector_layer = ConcreteMask(d_in, num)
                diff_selector = cae.DifferentiableSelector(model, selector_layer).to(device)
                diff_selector.fit(
                    train_dataloader,
                    val_dataloader,
                    lr=1e-3,
                    nepochs=250,
                    loss_fn=nn.CrossEntropyLoss(),
                    patience=5,
                    verbose=True)

                # Extract top features.
                logits = selector_layer.logits.cpu().data.numpy()
                selected_features = np.sort(logits.argmax(axis=1))
                if len(np.unique(selected_features)) != num:
                    print(f'{len(np.unique(selected_features))} selected instead of {num}, appending extras')
                    num_extras = num - len(np.unique(selected_features))
                    remaining_features = np.setdiff1d(np.arange(d_in), selected_features)
                    selected_features = np.sort(np.concatenate([np.unique(selected_features), remaining_features[:num_extras]]))

                # Prepare module to mask all but top features
                inds = torch.tensor(np.isin(np.arange(d_in), selected_features), device=device)
                mask_layer = StaticMaskLayer1d(inds)

                best_loss = np.inf
                for _ in range(num_restarts):
                    # Train model.
                    model = nn.Sequential(mask_layer, get_mlp_network(num, d_out))
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
                acc = best_model.evaluate(test_dataloader, acc_metric)
                results_dict['acc'][num] = acc
                results_dict['features'][num] = selected_features
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            print(results_dict)

        if args.method == 'eddi':
            start_time = time.time()
            # Train PVAE.
            mask_layer = MaskLayer(append=True, mask_size=d_in)

            bottleneck = 16
            # encoder = get_mlp_network(d_in + num_groups, bottleneck * 2)
            # decoder = get_mlp_network(bottleneck, d_in)
            hidden = 128
            dropout = 0.3
            encoder = get_mlp_network(d_in * 2, bottleneck * 2)
            decoder = get_mlp_network(bottleneck, d_in)  

            pv = pvae.PVAE(encoder, decoder, mask_layer, 20, 'gaussian').to(device)
            pv.fit(
                train_dataloader,
                val_dataloader,
                lr=1e-3,
                nepochs=50,
                verbose=False)
            
            # Train masked predictor.
            model = get_mlp_network(d_in * 2, d_out)
            sampler = iterative.UniformSampler(get_xy(train_dataset)[0])  # TODO don't actually need sampler
            iterative_selector = iterative.IterativeSelector(model, mask_layer, sampler).to(device)
            iterative_selector.fit(
                train_dataloader,
                val_dataloader,
                lr=1e-3,
                nepochs=50,
                loss_fn=nn.CrossEntropyLoss(),
                patience=5,
                verbose=True)
            
            # Set up EDDI feature selection object.
            eddi_selector = eddi.EDDI(pv, model, mask_layer).to(device)
            
            training_time = time.time() - start_time
            print(f"Training time {args.method}= {training_time}")

            with open(f"training_time_{args.method}.txt", 'a') as f:
                f.write(f"Training time {args.method}= {training_time}\n")
                
            # Evaluate.
            metrics_dict, cost_dict = eddi_selector.evaluate_multiple(test_dataloader, num_features, acc_metric, verbose=False)
            for num in num_features:
                acc = metrics_dict[num]
                results_dict['acc'][num] = acc
                print(f'Num = {num}, Acc = {100*acc:.2f}')
            
            print(results_dict)

        
        if args.method == 'dfs':
            start_time = time.time()
            # Prepare networks.
            predictor = get_mlp_network(d_in * 2, d_out)
            selector = get_mlp_network(d_in * 2, d_in)

            # Pretrain predictor
            mask_layer = MaskLayer(append=True, mask_size=d_in)
            # pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
            # pretrain.fit(
            #     train_dataset,
            #     val_dataset,
            #     mbsize,
            #     lr=1e-3,
            #     nepochs=100,
            #     loss_fn=nn.CrossEntropyLoss(),
            #     patience=5,
            #     verbose=True)

            pretrain = MaskingPretrainer(
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
            
            training_time = time.time() - start_time
            print(f"Training time {args.method}= {training_time}")

            with open(f"training_time_{args.method}.txt", 'a') as f:
                f.write(f"Training time {args.method}= {training_time}\n")

            # Evaluate.
            for num in num_features:
                acc = gdfs.evaluate(test_dataloader, num, acc_metric)
                results_dict['acc'][num] = acc
                print(f'Num = {num}, Acc = {100*acc:.2f}')
                
            # Save model
            gdfs.cpu()
            torch.save(gdfs, f'results/mnist_{args.method}_trial_{trial}.pt')

        # Train with full input
        if args.method == 'fully_supervised':
            model = get_mlp_network(d_in, d_out).to(device)
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
    
                print(f"Epoch: {epoch}, Train Loss: {train_batch_loss/len(train_dataloader)}, Val Loss: {val_batch_loss/len(val_dataloader)}, Val Performance: {acc_metric(torch.cat(val_pred_list), torch.cat(val_y_list))}")
            
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
            
            print(f"Test Performance:{acc_metric(torch.cat(test_pred_list), torch.cat(test_y_list))}")
            with open('confidence.npy', 'wb') as f:
                np.save(f, np.array(torch.cat(confidence_list).detach().numpy()))

        with open(f'results/mnist_{args.method}_trial_{trial}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)

        