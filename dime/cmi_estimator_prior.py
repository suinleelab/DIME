import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from dime.utils import get_entropy, get_confidence, ind_to_onehot


class CMIEstimatorPrior(pl.LightningModule):
    '''
    Greedy CMI estimation module that incorporates prior information.

    Args:
      value_network: network for estimating each feature's CMI.
      predictor: network for predicting response variable.
      mask_layer: layer for masking unobserved features.
      lr: learning rate.
      max_features: maximum number of features to select.
      eps: exploration rate.
      loss_fn: loss function for training predictor.
      val_loss_fn: loss function for validation performance.
      factor: factor by which to reduce learning rate on plateau.
      patience: number of epochs to wait before reducing learning rate.
      min_lr: minimum learning rate for scheduler.
      early_stopping_epochs: number of epochs to wait before stopping training.
      eps_decay: decay rate for exploration rate.
      eps_steps: number of exploration decay steps.
      feature_costs: cost of each feature.
      cmi_scaling: scaling method for CMI estimates ('none', 'positive', 'bounded').
        Recommended value is 'bounded'.
    '''

    def __init__(self,
                 value_network,
                 predictor,
                 mask_layer,
                 lr,
                 max_features,
                 eps,
                 loss_fn,
                 val_loss_fn,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None,
                 eps_decay=0.2,
                 eps_steps=1,
                 feature_costs=None,
                 cmi_scaling='bounded'):
        super().__init__()

        # Save network modules.
        self.value_network = value_network
        self.predictor = predictor
        self.mask_layer = mask_layer

        # Save optimization hyperparameters.
        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs
        self.factor = factor

        # Save feature hyperparameters.
        self.max_features = max_features
        self.mask_size = self.mask_layer.mask_size
        self.set_feature_costs(feature_costs)

        # Save loss functions.
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

        # Save CMI estimation hyperparameters.
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_steps = eps_steps
        if cmi_scaling not in ('none', 'positive', 'bounded'):
            raise ValueError('cmi_scaling must be one of "none", "positive", or "bounded"')
        self.cmi_scaling = cmi_scaling

        # Lightning module setup.
        self.automatic_optimization = False
        # self.save_hyperparameters()

    def set_feature_costs(self, feature_costs=None):
        '''Set feature cost values. Default is uniform cost.'''
        if feature_costs is None:
            feature_costs = torch.ones(self.mask_size)
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs)
        self.register_buffer('feature_costs', feature_costs)

    def set_stopping_criterion(self, budget=None, lam=None, confidence=None):
        '''Set parameters for stopping criterion.'''
        if sum([budget is None, lam is None, confidence is None]) != 2:
            raise ValueError('Must specify exactly one of budget, lam, and confidence')
        if budget is not None:
            self.budget = budget
            self.mode = 'budget'
        elif lam is not None:
            self.lam = lam
            self.mode = 'penalized'
        elif confidence is not None:
            self.confidence = confidence
            self.mode = confidence

    def on_fit_start(self):
        self.num_epsilon_steps = 0
        self.num_bad_epochs = 0

    def training_step(self, batch, batch_idx):
        # Prepare optimizer.
        opt = self.optimizers()
        opt.zero_grad()

        # Setup for minibatch.
        x, prior, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        pred_loss_total = 0

        # Predictor loss with no features.
        x_masked = self.mask_layer(x, mask)
        pred_without_next_feature = self.predictor(x_masked, prior)
        loss_without_next_feature = self.loss_fn(pred_without_next_feature, y)
        pred_loss = loss_without_next_feature.mean()
        pred_loss_total += pred_loss.detach()
        self.manual_backward(pred_loss / (self.max_features + 1))
        pred_without_next_feature = pred_without_next_feature.detach()
        loss_without_next_feature = loss_without_next_feature.detach()

        for _ in range(self.max_features):
            # Estimate CMI using value network.
            x_masked = self.mask_layer(x, mask)
            if self.cmi_scaling == 'bounded':
                entropy = get_entropy(pred_without_next_feature).unsqueeze(1)
                pred_cmi = self.value_network(x_masked, prior).sigmoid() * entropy
            elif self.cmi_scaling == 'positive':
                pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked, prior))
            else:
                pred_cmi = self.value_network(x_masked, prior)

            # Select next feature.
            best = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            random = torch.tensor(np.random.choice(self.mask_size, size=len(x)), device=x.device)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask, ind_to_onehot(actions, self.mask_size))

            # Predictor loss.
            x_masked = self.mask_layer(x, mask)
            pred_with_next_feature = self.predictor(x_masked, prior)
            loss_with_next_feature = self.loss_fn(pred_with_next_feature, y)

            # Value network loss.
            delta = loss_without_next_feature - loss_with_next_feature.detach()
            value_network_loss = nn.functional.mse_loss(pred_cmi[torch.arange(len(x)), actions], delta)

            # Calculate gradients.
            total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature)
            self.manual_backward(total_loss / (self.max_features + 1))

            # Updates.
            value_network_loss_total += torch.mean(value_network_loss)
            pred_loss_total += torch.mean(loss_with_next_feature)
            loss_without_next_feature = loss_with_next_feature.detach()
            pred_without_next_feature = pred_with_next_feature.detach()

        # Take optimizer step.
        opt.step()
        return {
            'value_network_loss': value_network_loss_total / self.max_features,
            'predictor_loss': pred_loss_total / (self.max_features + 1)}

    def training_epoch_end(self, outputs):
        # Get mean train losses.
        pred_loss = torch.stack([out['predictor_loss'] for out in outputs]).mean()
        value_network_loss = torch.stack([out['value_network_loss'] for out in outputs]).mean()

        # Log in progress bar.
        self.log('Loss Train/Mean', pred_loss, prog_bar=True, logger=False)
        self.log('Value Loss Train/Mean', value_network_loss, prog_bar=True, logger=False)

        # Log in tensorboard.
        self.logger.experiment.add_scalar('Loss Train/Mean', pred_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Value Loss Train/Mean', value_network_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        # Setup.
        x, prior, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked, prior)
        pred_list = [pred]

        for _ in range(self.max_features):
            # Estimate CMI using value network.
            x_masked = self.mask_layer(x, mask)
            if self.cmi_scaling == 'bounded':
                entropy = get_entropy(pred).unsqueeze(1)
                pred_cmi = self.value_network(x_masked, prior).sigmoid() * entropy
            elif self.cmi_scaling == 'positive':
                pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked, prior))
            else:
                pred_cmi = self.value_network(x_masked, prior)

            # Select next feature, ensure no repeats.
            pred_cmi -= 1e6 * mask
            best_feature_index = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            mask = torch.max(mask, ind_to_onehot(best_feature_index, self.mask_size))

            # Make prediction.
            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked, prior)
            pred_list.append(pred)

        # return pred, y
        return pred_list, y

    def validation_epoch_end(self, outputs):
        pred_list, y_list = zip(*outputs)
        y = torch.cat(y_list)
        preds_cat = [torch.cat(preds) for preds in zip(*pred_list)]
        pred_loss_ = [self.loss_fn(preds, y).mean() for preds in preds_cat]
        val_loss_ = [self.val_loss_fn(preds, y) for preds in preds_cat]
        pred_loss_mean = torch.stack(pred_loss_).mean()
        val_loss_mean = torch.stack(val_loss_).mean()
        pred_loss_final = pred_loss_[-1]
        val_loss_final = val_loss_[-1]

        # Log in progress bar.
        self.log('Loss Val/Mean', pred_loss_mean, prog_bar=True, logger=False)
        self.log('Perf Val/Mean', val_loss_mean, prog_bar=True, logger=False)
        self.log('Loss Val/Final', pred_loss_final, prog_bar=True, logger=False)
        self.log('Perf Val/Final', val_loss_final, prog_bar=True, logger=False)
        self.log('Eps Value', self.eps, prog_bar=True, logger=False)

        # Log in tensorboard.
        self.logger.experiment.add_scalar('Loss Val/Mean', pred_loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar('Perf Val/Mean', val_loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar('Loss Val/Final', pred_loss_final, self.current_epoch)
        self.logger.experiment.add_scalar('Perf Val/Final', val_loss_final, self.current_epoch)
        self.logger.experiment.add_scalar('Eps Value', self.eps, self.current_epoch)

        # Take lr scheduler step using loss.
        sch = self.lr_schedulers()
        sch.step(pred_loss_mean)

        # Check for loss improvement.
        if pred_loss_mean == sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.early_stopping_epochs:
            # Decay epsilon.
            self.eps *= self.eps_decay
            self.num_bad_epochs = 0
            self.num_epsilon_steps += 1
            print(f'Decaying eps to {self.eps:.5f}, step = {self.num_epsilon_steps}')

            # Early stopping.
            if self.num_epsilon_steps >= self.eps_steps:
                self.trainer.should_stop = True

            # Reset optimizer learning rate. Could fully reset optimizer and scheduler, but this is simpler.
            for g in self.optimizers().param_groups:
                g['lr'] = self.lr

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience,
            min_lr=self.min_lr, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }

    def on_predict_start(self):
        if not hasattr(self, 'mode'):
            print('Must specify stopping criterion. Recommended usage is via `inference` function')

    def predict_step(self, batch, batch_idx):
        # Setup.
        if len(batch) == 3:
            x, prior, y = batch
        else:
            x, prior = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        accept_sample = torch.ones(len(x), dtype=bool, device=x.device)

        for step in range(self.mask_size):
            # Estimate CMI using value network.
            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked, prior)
            if self.cmi_scaling == 'bounded':
                entropy = get_entropy(pred).unsqueeze(1)
                pred_cmi = self.value_network(x_masked, prior).sigmoid() * entropy
            elif self.cmi_scaling == 'positive':
                pred_cmi = torch.nn.functional.softplus(self.value_network(x_masked, prior))
            else:
                pred_cmi = self.value_network(x_masked, prior)

            # Determine best feature.
            pred_cmi -= 1e6 * mask
            best_feature_index = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            selection = ind_to_onehot(best_feature_index, self.mask_size)

            # Stopping criteria.
            if self.mode == 'penalized':
                # Check for sufficiently large CMI.
                accept_sample = torch.max(pred_cmi / self.feature_costs, dim=1).values > self.lam

            elif self.mode == 'budget':
                # Check for remaining budget.
                features_selected = torch.max(mask, selection)
                accept_sample = torch.sum(features_selected * self.feature_costs, dim=1) <= self.budget

            elif self.mode == 'confidence':
                # Check for sufficient confidence.
                confidences = get_confidence(pred)
                accept_sample = confidences < self.confidence

            # Ensure positive CMI.
            accept_sample = torch.bitwise_and(accept_sample, pred_cmi.max(dim=1).values > 0)

            # Stop if no samples were accepted.
            if sum(accept_sample).item() == 0:
                break

            # Update mask for accepted samples.
            mask[accept_sample] = torch.max(mask[accept_sample], selection[accept_sample])

        # Save final predictions and masks.
        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked, prior)
        if len(batch) == 3:
            return mask.cpu(), pred.cpu(), y.cpu()
        else:
            return mask.cpu(), pred.cpu()

    # Note: Lightning doesn't support using this function to post-process predictions.
    # def on_predict_epoch_end(self, outputs):
    #     pass

    def format_predictions(self, outputs):
        '''Format predictions output by trainer.predict().'''
        if len(outputs[0]) == 3:
            mask_list, pred_list, y_list = zip(*outputs)
            mask = torch.cat(mask_list)
            pred = torch.cat(pred_list)
            y = torch.cat(y_list)
            return {
                'mask': mask,
                'pred': pred,
                'y': y
            }
        else:
            mask_list, pred_list = zip(*outputs)
            mask = torch.cat(mask_list)
            pred = torch.cat(pred_list)
            return {
                'mask': mask,
                'pred': pred
            }

    def inference(self, trainer, data_loader, feature_costs=None, budget=None, lam=None, confidence=None):
        '''
        Make predictions on a dataset using the trained model.

        Args:
          trainer: Lightning trainer object.
          data_loader: PyTorch DataLoader object.
          feature_costs: cost of each feature. Defaults to values used during training.
          budget: maximum cost of selected features, for budget-constrained stopping criterion.
          lam: penalty for selecting features with low CMI, for penalized stopping criterion.
          confidence: minimum confidence of predictions, for confidence-constrained stopping criterion.

        Returns:
            Dictionary of predictions, masks, and labels if provided by dataloader.

        '''
        original_feature_costs = self.feature_costs.cpu()
        self.set_feature_costs(feature_costs)
        self.set_stopping_criterion(budget, lam, confidence)

        # Generate and format predictions.
        outputs = trainer.predict(self, data_loader)
        outputs = self.format_predictions(outputs)

        # Restore original feature costs.
        self.set_feature_costs(original_feature_costs)
        del self.mode

        return outputs
