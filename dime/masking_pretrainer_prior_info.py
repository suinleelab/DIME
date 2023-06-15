import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torchvision import transforms
from torch.utils.data import DataLoader
from copy import deepcopy
from dime.utils import generate_uniform_mask, restore_parameters, get_entropy, get_confidence, selection_without_lamda, accuracy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

class MaskingPretrainerPriorInfo(nn.Module):
    '''Pretrain model with missing features.'''

    def __init__(self, model, mask_layer):
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer
        
    def fit(self,
            train,
            val,
            mbsize,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            early_stopping_epochs=None,
            verbose=True,
            trained_predictor_name='predictor.pth'):
        '''
        Train model.
        
        Args:
          train:
          val:
          mbsize:
          lr:
          nepochs:
          loss_fn:
          val_loss_fn:
          val_loss_mode:
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          verbose:
        '''
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')

        # Set up data loaders.
        train_loader = DataLoader(
            train, batch_size=mbsize, shuffle=True, pin_memory=True,
            drop_last=True, num_workers=4)
        val_loader = DataLoader(
            val, batch_size=mbsize, shuffle=False, pin_memory=True,
            drop_last=True, num_workers=4)
        
        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)
        
        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val))
            assert len(x.shape) == 1
            mask_size = len(x)

        # For tracking best model and early stopping.
        best_model = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
      
        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()

            for batch in tqdm(train_loader):
                x, x_sketch, y = batch
                x_sketch = x_sketch.to(device)
                # Move to device.
                x = x.to(device)
                
                y = y.to(device)
                
                # Generate missingness.
                m = generate_uniform_mask(len(x), mask_size).to(device)
                # m = torch.ones(len(x), mask_size).to(device)
                # Calculate loss.
                x_masked = mask_layer(x, m)
                pred = model(x_masked, x_sketch)

                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                
            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for batch in val_loader:
                    
                    x, x_sketch, y = batch
                    x_sketch = x_sketch.to(device)

                    # Move to device.
                    x = x.to(device)
                    
                    # Generate missingness.
                    # TODO this should be precomputed and shared across epochs
                    m = generate_uniform_mask(len(x), mask_size).to(device)
                    # m = torch.ones(len(x), mask_size).to(device)

                    # Calculate prediction.
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked, x_sketch.detach())

                    pred_list.append(pred.cpu())
                    label_list.append(y.cpu())
                    
                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = loss_fn(pred, y).item()
                
            
            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val loss = {val_loss:.4f}')
                print(f'Val Performance = {val_loss_fn(pred, y)}')
            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                
            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f'Stopping early at epoch {epoch+1}')
                break

        # Copy parameters from best model.
        restore_parameters(model, best_model)

        # save best model
        torch.save(model.state_dict(), f"results/{trained_predictor_name}")
        
    # TODO unclear whether this should use masking or all inputs
    def evaluate(self, dataset, metric, batch_size):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          dataset:
          metric:
          batch_size:
        '''
        # Setup.
        self.model.eval()
        device = next(self.model.parameters()).device
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
            drop_last=False, num_workers=4)

        # Determine mask size.
        if hasattr(self.mask_layer, 'mask_size') and (self.mask_layer.mask_size is not None):
            mask_size = self.mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(dataset))
            assert len(x.shape) == 1
            mask_size = len(x)

        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)
                mask = torch.ones(len(x), mask_size, device=device)

                # Calculate loss.
                pred = self.forward(x, mask)
                pred_list.append(pred.cpu())
                label_list.append(y.cpu())
                
            # Calculate metric(s).
            y = torch.cat(label_list, 0)
            pred = torch.cat(pred_list, 0)
            if isinstance(metric, (tuple, list)):
                score = [m(pred, y).item() for m in metric]
            elif isinstance(metric, dict):
                score = {name: m(pred, y).item() for name, m in metric.items()}
            else:
                score = metric(pred, y).item()
                
        return score
    
    def forward(self, x, mask):
        '''
        Generate model prediction.
        
        Args:
          x:
          mask:
        '''
        x_masked = self.mask_layer(x, mask)
        return self.model(x_masked)