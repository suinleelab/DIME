import torch
import torch.nn as nn
import torch.optim as optim
from dime.utils import restore_parameters
from copy import deepcopy

class BaseModel(nn.Module):
    '''
    Base model, no missing features.
    
    Args:
      model:
    '''

    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            early_stopping_epochs=None,
            verbose=True):
        '''
        Train model.
        
        Args:
          train_loader:
          val_loader:
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
        
        # Set up optimizer and lr scheduler.
        model = self.model
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)

        # For tracking best model and early stopping.
        best_model = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
            
        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)

                # Calculate loss.
                pred = model(x)
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

                for x, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    
                    # Calculate prediction.
                    pred = model(x)
                    pred_list.append(pred.cpu())
                    label_list.append(y.cpu())
                    
                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()
            
            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val loss = {val_loss:.4f}\n')
                
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
        
    def evaluate(self, loader, metric):
        '''
        Evaluate mean performance across a dataset.
        
        Args:
          loader:
          metric:
        '''
        # Setup.
        self.model.eval()
        device = next(self.model.parameters()).device

        # For calculating mean loss.
        pred_list = []
        label_list = []

        with torch.no_grad():
            for x, y in loader:
                # Move to GPU.
                x = x.to(device)

                # Calculate loss.
                pred = self.model(x)
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
    
    def forward(self, x):
        '''
        Generate model prediction.
        
        Args:
          x:
        '''
        return self.model(x)