import torch
import torch.optim as optim
import pytorch_lightning as pl
from dime.utils import generate_uniform_mask


class MaskingPretrainerPrior(pl.LightningModule):
    '''
    Pretrain model with missing features.

    TODO list args
    '''

    def __init__(self,
                 model,
                 mask_layer,
                 lr,
                 loss_fn,
                 val_loss_fn,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None):
        super().__init__()

        # Save network modules.
        self.model = model
        self.mask_layer = mask_layer
        self.mask_size = self.mask_layer.mask_size

        # Save optimization hyperparameters.
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs

        # Save loss functions.
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

    def on_fit_start(self):
        self.num_bad_epochs = 0

    def training_step(self, batch, batch_idx):
        # Setup for minibatch.
        x, prior, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)

        # Calculate predictions and loss.
        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked, prior)
        return self.loss_fn(pred, y)

    def train_epoch_end(self, outputs):
        # Log loss in progress bar.
        loss = torch.stack(outputs).mean()
        self.log('Loss Train', loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # Setup for minibatch.
        x, prior, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)

        # Calculate predictions.
        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked, prior)
        return pred, y

    def validation_epoch_end(self, outputs):
        pred_list, y_list = zip(*outputs)
        pred = torch.cat(pred_list)
        y = torch.cat(y_list)
        loss = self.loss_fn(pred, y)
        val_loss = self.val_loss_fn(pred, y)

        # Log to progress bar.
        self.log('Loss Val', loss, prog_bar=True, logger=True)
        self.log('Perf Val', val_loss, prog_bar=True, logger=True)

        # Perform manual early stopping. Note that this is called before lr scheduler step.
        sch = self.lr_schedulers()
        if loss < sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.early_stopping_epochs:
            # Early stopping.
            self.trainer.should_stop = True

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience,
            min_lr=self.min_lr, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'Loss Val'
        }
