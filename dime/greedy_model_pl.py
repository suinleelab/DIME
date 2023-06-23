import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from dime.utils import get_entropy, get_confidence, ind_to_onehot


class GreedyCMIEstimatorPL(pl.LightningModule):
    '''
    Greedy CMI estimation module.

    # TODO list args.
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
                 use_entropy=True):
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
        self.use_entropy = use_entropy

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
            raise ValueError('Must specify exactly one of budget, lam, and confidence.')
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
        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        pred_loss_total = 0

        # Predictor loss with no features.
        x_masked = self.mask_layer(x, mask)
        pred_without_next_feature = self.predictor(x_masked)
        loss_without_next_feature = self.loss_fn(pred_without_next_feature, y)
        self.manual_backward(loss_without_next_feature.mean() / (self.max_features + 1))
        pred_without_next_feature = pred_without_next_feature.detach()
        loss_without_next_feature = loss_without_next_feature.detach()

        for _ in range(self.max_features):
            # Estimate CMI using value network.
            x_masked = self.mask_layer(x, mask)
            # TODO remove unnecessary detach statement.
            if self.use_entropy:
                entropy = get_entropy(pred_without_next_feature.detach()).unsqueeze(1)
                # TODO why is sigmoid appended to the network? Activations should be applied here.
                # TODO use_entropy should be an argument with multiple options (none, entropy scaling, softplus).
                pred_cmi = self.value_network(x_masked) * entropy
            else:
                pred_cmi = self.value_network(x_masked)

            # Select next feature.
            best = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            random = torch.tensor(np.random.choice(self.mask_size, size=len(x)), device=x.device)
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            actions = exploit * best + (1 - exploit) * random
            mask = torch.max(mask, ind_to_onehot(actions, self.mask_size))

            # Predictor loss.
            x_masked = self.mask_layer(x, mask)
            pred_with_next_feature = self.predictor(x_masked)
            loss_with_next_feature = self.loss_fn(pred_with_next_feature, y)

            # Value network loss.
            delta = loss_without_next_feature - loss_with_next_feature.detach()
            value_network_loss = nn.functional.mse_loss(pred_cmi[torch.arange(len(x)), actions], delta)

            # Update for next step.
            loss_without_next_feature = loss_with_next_feature.detach()
            pred_without_next_feature = pred_with_next_feature.detach()

            # Calculate gradients.
            total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature)
            self.manual_backward(total_loss / (self.max_features + 1))

            # Update total loss.
            value_network_loss_total += torch.mean(value_network_loss)
            pred_loss_total += torch.mean(loss_with_next_feature)

        # Take optimizer step.
        opt.step()
        return {
            'value_network_loss': value_network_loss_total / self.max_features,
            'predictor_loss': pred_loss_total / self.max_features}

    def training_epoch_end(self, outputs):
        # Get mean train losses.
        pred_loss = torch.stack([x['predictor_loss'] for x in outputs]).mean()
        value_network_loss = torch.stack([x['value_network_loss'] for x in outputs]).mean()

        # Log in progress bar.
        self.log('Predictor Loss Train', pred_loss, prog_bar=True, logger=False)
        self.log('Value Network Loss Train', value_network_loss, prog_bar=True, logger=False)

        # Log in tensorboard.
        self.logger.experiment.add_scalar('Predictor Loss/Train', pred_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Value Network Loss/Train', value_network_loss, self.current_epoch)

    # TODO this is based on final rather than average loss, which is not what we want.
    def validation_step(self, batch, batch_idx):
        # Setup.
        x, y = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        x_masked = self.mask_layer(x, mask)
        pred = self.predictor(x_masked)

        for _ in range(self.max_features):
            # Estimate CMI using value network.
            x_masked = self.mask_layer(x, mask)
            if self.use_entropy:
                # TODO again, sigmoid and softplus should be applied here, not in the network.
                entropy = get_entropy(pred).unsqueeze(1)
                pred_cmi = self.value_network(x_masked) * entropy
            else:
                pred_cmi = self.value_network(x_masked)

            # Select next feature, ensure no repeats.
            pred_cmi -= 1e6 * mask
            best_feature_index = torch.argmax(pred_cmi / self.feature_costs, dim=1)
            mask = torch.max(mask, ind_to_onehot(best_feature_index, self.mask_size))

            # Make prediction.
            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked)

        return pred, y

    def validation_epoch_end(self, outputs):
        pred_list, y_list = zip(*outputs)
        pred_loss = self.loss_fn(torch.cat(pred_list), torch.cat(y_list)).mean()
        val_loss = self.val_loss_fn(torch.cat(pred_list), torch.cat(y_list))

        # Log in progress bar.
        self.log('Predictor Loss Val', pred_loss, prog_bar=True, logger=False)
        self.log('Performance_Val', val_loss, prog_bar=True, logger=False)
        self.log('Eps Value', self.eps, prog_bar=True, logger=False)

        # Log in tensorboard.
        self.logger.experiment.add_scalar('Predictor Loss/Val', pred_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Predictor Performance/Val', val_loss, self.current_epoch)
        self.logger.experiment.add_scalar('Eps Value', self.eps, self.current_epoch)

        # Take lr scheduler step using loss.
        sch = self.lr_schedulers()
        sch.step(pred_loss)

        # Check for loss improvement.
        if pred_loss == sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.early_stopping_epochs:
            # Decay epsilon.
            print(f'Decaying eps to {self.eps}, step = {self.num_epsilon_steps + 1}')
            self.eps *= self.eps_decay
            self.num_bad_epochs = 0
            self.num_epsilon_steps += 1
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
            print('Must specify stopping criterion. Recommended usage is via `inference` function.')

    def predict_step(self, batch, batch_idx):
        # Setup.
        if len(batch) == 2:
            x, y = batch
        else:
            x = batch
        mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        accept_sample = torch.ones(len(x), dtype=bool, device=x.device)

        for step in range(self.mask_size):
            # Estimate CMI using value network.
            x_masked = self.mask_layer(x, mask)
            pred = self.predictor(x_masked)
            if self.use_entropy:
                # TODO fix this after sigmoid is removed from network
                entropy = get_entropy(pred.detach()).unsqueeze(1)
                pred_cmi = self.value_network(x_masked) * entropy
            else:
                pred_cmi = self.value_network(x_masked)

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
        pred = self.predictor(x_masked)
        if len(batch) == 2:
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

        TODO list args
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
