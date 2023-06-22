import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from dime.utils import get_entropy, get_confidence, selection_without_lamda, ind_to_onehot
from tqdm import tqdm


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
                 # TODO rename this to eps_factor or eps_decay
                 eps_decay_rate=0.2,
                 # TODO delete this, it's redundant with ep_steps
                 eps_decay=True,
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
        if feature_costs is None:
            feature_costs = torch.ones(self.mask_size)
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs)
        self.register_buffer('feature_costs', feature_costs)

        # Save loss functions.
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

        # Save CMI estimation hyperparameters.
        self.eps = eps
        self.eps_factor = eps_decay_rate
        self.eps_steps = eps_steps
        self.use_entropy = use_entropy

        # Lightning module setup.
        self.automatic_optimization = False
        # self.save_hyperparameters()

    def on_fit_start(self):
        # TODO revisit this later, might not need these.
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
            if self.use_entropy:
                # TODO why is torch.tensor needed here?
                entropy = torch.tensor(get_entropy(pred_without_next_feature.detach()), device=x.device).unsqueeze(1)
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
                entropy = torch.tensor(get_entropy(pred), device=x.device).unsqueeze(1)
                pred_cmi = self.value_network(x_masked) * entropy
            else:
                self.value_network(x_masked)

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
        # TODO remove need for numpy
        val_loss = self.val_loss_fn(torch.cat(pred_list), torch.cat(y_list).cpu().data.numpy())

        # Log in progress bar.
        self.log('Predictor Loss Val', pred_loss, prog_bar=True, logger=False)
        self.log('Performance_Val', val_loss, prog_bar=True, logger=False)

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
            print(f'Decaying eps to {self.eps}, step = {self.num_epsilon_steps}')
            self.eps *= self.eps_factor
            self.num_bad_epochs = 0
            self.num_epsilon_steps += 1
            if self.num_epsilon_steps >= self.eps_steps:
                self.trainer.should_stop = True

            # TODO should we reset all optimizer hyperparameters, not just the learning rate?
            # TODO should we also reset the scheduler?
            # Reset optimizer learning rate.
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

    # TODO maybe we can rely on a built-in Lightning function for this?
    def evaluate(self,
                 test_dataloader,
                 performance_func,
                 feature_costs=None,
                 # TODO maybe this can be a string or boolean? There can't be that many options.
                 selection_func=selection_without_lamda,
                 evaluation_mode='lamda-penalty',
                 # TODO remove this later
                 use_entropy=True,
                 # TODO instead of kwargs, maybe we should indicate the expected arguments for each evaluation mode?
                 **kwargs):
        '''
        Evaluate the value network given a specified stopping criterion.

        # TODO list args.
        '''
        # Setup.
        device = next(self.predictor.parameters()).device
        # TODO maybe we should just return predictions instead of loss?
        # loss = nn.CrossEntropyLoss(reduction='none')
        if feature_costs is None:
            feature_costs = self.feature_costs
        else:
            if isinstance(feature_costs, np.ndarray):
                feature_costs = torch.tensor(feature_costs)
            feature_costs = feature_costs.to(self.device)

        # TODO this is boilerplate.
        self.value_network.eval()
        self.predictor.eval()

        # For saving results.
        final_masks = []
        pred_list = []
        y_list = []

        # TODO this is boilerplate.
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_dataloader)):
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                # Setup.
                mask = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=device)
                accept_sample = torch.ones(len(x), dtype=bool, device=device)

                for step in range(self.mask_size):
                    # Estimate CMI using value network.
                    x_masked = self.mask_layer(x, mask)
                    pred = self.predictor(x_masked)
                    if self.use_entropy:
                        # TODO fix this after sigmoid is removed from network
                        entropy = torch.tensor(get_entropy(pred.detach()), device=x.device).unsqueeze(1)
                        pred_cmi = self.value_network(x_masked) * entropy
                    else:
                        pred_cmi = self.value_network(x_masked)

                    # Determine best feature.
                    pred_cmi -= 1e6 * mask
                    best_feature_index = torch.argmax(pred_cmi / feature_costs, dim=1)
                    selection = ind_to_onehot(best_feature_index, self.mask_size)

                    # Stopping criteria.
                    # TODO change names for each option
                    if evaluation_mode == 'lamda-penalty':
                        # Check for sufficiently large CMI.
                        lamda = kwargs['lamda']
                        accept_sample = torch.max(pred_cmi / feature_costs, dim=1).values > lamda

                    elif evaluation_mode == 'fixed-budget':
                        # Check for remaining budget.
                        budget = kwargs['budget']
                        features_selected = torch.max(mask, selection)
                        accept_sample = torch.sum(features_selected * feature_costs, dim=1) <= budget

                    elif evaluation_mode == 'confidence':
                        # Check for sufficient confidence.
                        min_confidence = kwargs['min_confidence']
                        confidences = get_confidence(pred)
                        accept_sample = confidences < min_confidence

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
                final_masks.append(mask.cpu())
                pred_list.append(pred.cpu())
                y_list.append(y.cpu())

        # TODO should rename these
        # TODO should maybe not return performance, just predictions and masks
        # Return multiple metrics for downstream analysis
        return_dict = dict(
            final_masks=torch.cat(final_masks).numpy(),
            pred_list=torch.cat(pred_list),
            y_list=torch.cat(y_list),
            performance=performance_func(torch.cat(pred_list), torch.cat(y_list)),
        )
        return return_dict
