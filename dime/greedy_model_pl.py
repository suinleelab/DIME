# PyTorch lightning version

import torch
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn as nn
from dime.utils import get_entropy, get_confidence, selection_without_lamda
import numpy as np
from tqdm import tqdm

# Convert index to one-hot encoding
def ind_to_onehot(inds, n):
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot

class GreedyCMIEstimatorPL(pl.LightningModule):
    def __init__(self, value_network, predictor, mask_layer,
            lr,
            max_features,
            eps,
            loss_fn,
            val_loss_fn,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True,
            val_loss_mode='min',
            early_stopping_epochs=None,
            eps_decay_rate=0.2,
            min_eps=1e-8,
            eps_decay=False,
            feature_costs=None,
            lamda=None,
            use_entropy=True):

            super().__init__()

            # Save networks
            self.value_network = value_network
            self.predictor = predictor
            self.mask_layer = mask_layer

            # Save hyperparameters
            self.lr = lr
            self.max_features = max_features
            self.eps = eps
            self.patience = patience
            self.verbose = verbose
            self.factor = factor

            # Save performance functions
            self.loss_fn = loss_fn
            self.val_loss_mode = val_loss_mode
            self.val_loss_fn = val_loss_fn
            
            self.min_lr = min_lr
            self.early_stopping_epochs = early_stopping_epochs
            self.eps_decay_rate = eps_decay_rate
            self.min_eps = min_eps
            self.eps_decay = eps_decay
            self.feature_costs = feature_costs
            self.lamda = lamda
            self.use_entropy = use_entropy
            self.mask_size = self.mask_layer.mask_size
            self.mse_loss_fn = torch.nn.MSELoss(reduction='none')
            self.automatic_optimization = False
            # self.save_hyperparameters()

    def on_fit_start(self):
        self.num_steps = 1
        self.num_bad_epochs = 0
        if self.early_stopping_epochs is None:
            self.early_stopping_epochs = self.patience + 1

    def on_train_batch_start(self, batch, batch_idx):
        if self.num_steps > 10: return -1
    
    def training_step(self, batch, batch_idx):
        # Prepare optimizer.
        opt = self.optimizers()
        opt.zero_grad()
        
        x, y = batch

        # Create feature cost matrix on first iteration
        if self.current_epoch==0 and batch_idx == 0 and self.num_steps==1:
            if self.feature_costs is None:
                self.feature_costs = torch.ones((len(x), self.mask_size), device=x.device)
            else:
                self.feature_costs = torch.tensor(self.feature_costs, device=x.device)
                self.feature_costs = torch.tile(self.feature_costs, (x.shape[0], 1))

        # Setup.
        m_hard = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)
        value_network_loss_total = 0
        pred_loss_total = 0

        # Predictor loss with the features selected so far 
        x_masked = self.mask_layer(x, m_hard)

        pred_without_next_feature = self.predictor(x_masked)

        loss_without_next_feature = self.loss_fn(pred_without_next_feature, y)

        for _ in range(self.max_features):
            # Estimate CMI using value network, use_entropy  is used to bound predicted CMIs
            x_masked = self.mask_layer(x, m_hard)
            pred_CMI = self.value_network(x_masked) * torch.tensor(get_entropy(pred_without_next_feature.detach()), device=x.device).unsqueeze(1) \
                if self.use_entropy else self.value_network(x_masked)
            

            # Select the next feature
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            if self.lamda is None:
                best = torch.argmax(pred_CMI/self.feature_costs, dim=1)
            else:
                best = torch.argmax(pred_CMI - self.lamda * self.feature_costs, dim=1)
            random = torch.tensor(np.random.choice(self.mask_size, size=len(x)), device=x.device)
            actions = exploit * best + (1 - exploit) * random
            hard = ind_to_onehot(actions, self.mask_size)
            m_hard = torch.max(m_hard, hard)

            # Predictor loss including the next feature
            x_masked = self.mask_layer(x, m_hard)

            pred_with_next_feature = self.predictor(x_masked)

            # CE loss for predictor
            loss_with_next_feature = self.loss_fn(pred_with_next_feature, y)

            # MSE loss for value_network
            value_network_loss = self.mse_loss_fn(pred_CMI[torch.arange(len(x)), actions], 
            loss_without_next_feature.detach() - loss_with_next_feature.detach())
            
            # Update previous loss and prediction to be used in the next selection step
            loss_without_next_feature = loss_with_next_feature
            pred_without_next_feature = pred_with_next_feature

            total_loss = torch.mean(value_network_loss) + torch.mean(loss_with_next_feature)
            # Calculate gradients
            self.manual_backward(total_loss / self.max_features)
            
            value_network_loss_total += torch.mean(value_network_loss)
            pred_loss_total += torch.mean(loss_with_next_feature)

        # Take optimizer step           
        opt.step()

        return {
                "value_network_loss": value_network_loss_total / self.max_features, 
                "loss": pred_loss_total / self.max_features}
    
    def training_epoch_end(self, outputs):
        # Get mean train losses
        pred_loss = torch.stack([x['loss'] for x in outputs]).mean()
        value_network_loss = torch.stack([x['value_network_loss'] for x in outputs]).mean()

        # Log in tensorboard
        self.logger.experiment.add_scalar("Predictor Loss/Train",
                                            pred_loss,
                                            self.current_epoch)
        
        self.logger.experiment.add_scalar("Value Network Loss/Train",
                                            value_network_loss,
                                            self.current_epoch)
        
        self.log('Predictor Loss Train', pred_loss, prog_bar=True, logger=False)
        self.log('Value Network Loss Train', value_network_loss, prog_bar=True, logger=False)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Setup.
        m_hard = torch.zeros(len(x), self.mask_size, dtype=x.dtype, device=x.device)

        pred_loss_total_val = 0
        pred = None

        for _ in range(self.max_features):
            # Estimate CMI using value network, use_entropy  is used to bound predicted CMIs
            x_masked = self.mask_layer(x, m_hard)
            pred_without_next_feature = self.predictor(x_masked)
            pred_CMI = self.value_network(x_masked) * torch.tensor(get_entropy(pred_without_next_feature.detach()), device=x.device).unsqueeze(1) \
                            if self.use_entropy else self.value_network(x_masked)

            # Select next feature and ensure no repeats
            pred_CMI -= 1e6 * m_hard
            if self.lamda is None:
                best_feature_index = torch.argmax(pred_CMI/self.feature_costs, dim=1)
            else:
                best_feature_index = torch.argmax(pred_CMI - self.lamda * self.feature_costs, dim=1)

            # Convert to one-hot
            hard = ind_to_onehot(best_feature_index, self.mask_size)

            # Update mask
            m_hard = torch.max(m_hard, hard)

            # Make prediction
            x_masked = self.mask_layer(x, m_hard)
            pred = self.predictor(x_masked)
                
            pred_val_loss = self.loss_fn(pred, y)

            # Calculate CE loss
            pred_loss_total_val += torch.mean(pred_val_loss)
                
        return pred, y, (pred_loss_total_val / self.max_features).reshape(1)
    
    def predict_step(self, batch, batch_idx, lamda=2):
        x, y = batch
        return y

    def on_predict_epoch_end(self, outputs):
        return {"outputs": torch.tensor(outputs)}
    
    def validation_epoch_end(self, outputs):
        pred_list, y_list, pred_loss_list = zip(*outputs)
        val_loss_fn_output = self.val_loss_fn(torch.cat(pred_list), torch.cat(y_list).cpu().data.numpy())
        pred_loss_val = torch.cat(pred_loss_list).mean()

        # Log in tensorboard
        self.logger.experiment.add_scalar("Predictor Loss/Val",
                                            pred_loss_val,
                                            self.current_epoch)
        
        self.logger.experiment.add_scalar("Predictor Performance/Val",
                                            val_loss_fn_output,
                                            self.current_epoch)

        self.logger.experiment.add_scalar("Eps Value",
                                            self.eps,
                                            self.current_epoch)
        
        self.log('Predictor Loss Val', pred_loss_val, prog_bar=True, logger=False)
        self.log('Performance_Val', val_loss_fn_output, prog_bar=True, logger=False)

        # Take lr scheduler step.
        sch = self.lr_schedulers()
        sch.step(pred_loss_val)

        if pred_loss_val == sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.early_stopping_epochs:
            # Val loss is not improving, decay epsilon and restart training
            print(f"Decaying eps: {self.eps}")
            print(f"Step Num: {self.num_steps}")
            self.eps *= self.eps_decay_rate
            self.num_bad_epochs = 0
            self.num_steps += 1
            
            # Reset optimizer learning rate
            for g in self.optimizers().param_groups:
                g['lr'] = self.lr
        
        if self.current_epoch == (self.trainer.max_epochs - 1):
            self.num_steps += 1
            
    def configure_optimizers(self):
        opt = optim.Adam(set(list(self.predictor.parameters()) + list(self.value_network.parameters())), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=self.val_loss_mode, factor=self.factor,
                patience=self.patience, min_lr=self.min_lr, verbose=self.verbose)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'pred_loss_val'
        }

    def evaluate(self, 
                 test_dataloader, 
                 performance_func, 
                 feature_costs=None, 
                 selection_func=selection_without_lamda, 
                 evaluation_mode="lamda-penalty",
                 use_entropy=True,
                 **kwargs):
        '''
        Evaluate the value network and get predictions with stopping criteria
        '''
        #setup
        value_network = self.value_network
        predictor = self.predictor
        mask_layer = self.mask_layer
        loss = nn.CrossEntropyLoss(reduction='none')

        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(test_dataloader.dataset))
            assert len(x.shape) == 1
            mask_size = len(x)

        value_network.eval()
        predictor.eval()

        device = next(self.predictor.parameters()).device
        final_masks = []
        pred_list = []
        y_list = []
        masks_dict = {}
        entropy_dict = {}
        pred_cmi_dict = {}
        loss_dict = {}

        with torch.no_grad():
            batch_loss = 0
            # Val step
            for i, batch in enumerate(tqdm(test_dataloader)):
                x, y = batch

                x = x.to(device)
                y = y.to(device)

                # Create feature cost matrix on first iteration
                if i==0:
                    if feature_costs is None:
                        feature_costs = torch.ones((len(x), mask_size), device=device)
                    else:
                        feature_costs = torch.tensor(feature_costs, device=device)
                        feature_costs = torch.tile(feature_costs, (x.shape[0], 1))
                # mask
                m_hard = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                budget_exhausted = False
                accept_sample = torch.full((x.shape[0],), True, device=device)
                budget_not_used_up =  torch.full((x.shape[0],), True, device=device)
                iteration = 0

                while not budget_exhausted:
                    # Estimate CMI using value network, use_entropy  is used to bound predicted CMIs
                    x_masked = mask_layer(x, m_hard)
                    pred = predictor(x_masked)
                    pred_CMI = value_network(x_masked) * torch.tensor(get_entropy(pred.detach()), device=x.device).unsqueeze(1) \
                                    if use_entropy else value_network(x_masked)

                    # Save different metric dicts
                    if iteration not in pred_cmi_dict:
                        pred_cmi_dict[iteration] = [pred_CMI.cpu()]
                    else:
                        pred_cmi_dict[iteration].append(pred_CMI.cpu())

                    if iteration not in entropy_dict:
                        entropy_dict[iteration] = [torch.tensor(get_entropy(pred))]
                    else:
                        entropy_dict[iteration].append(torch.tensor(get_entropy(pred)))

                    if iteration not in loss_dict:
                        loss_dict[iteration] = [loss(pred, y).cpu().numpy()]
                    else:
                        loss_dict[iteration].append(loss(pred, y).cpu().numpy())

                    # Penalized Policy Stopping Criteria
                    if evaluation_mode == 'lamda-penalty':
                        lamda = kwargs['lamda']
                        # print(pred_CMI)
                        adjusted_CMI = pred_CMI - lamda * feature_costs
                        check_neg_CMI = torch.max(adjusted_CMI, dim=1).values > 0

                        
                        if sum(check_neg_CMI) == 0 or torch.max(torch.sum(m_hard, dim=1)) == mask_size:
                            # max adjusted CMI for all samples is negative, this batch is done
                            budget_exhausted = True
                            break

                        # Once a sample is done, it should not be selected again
                        accept_sample = torch.bitwise_and(accept_sample, check_neg_CMI)

                        # Select next feature and ensure no repeats
                        pred_CMI -= 1e6 * m_hard
                        best_feature_index = selection_func(pred_CMI, feature_costs, lamda)
                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Update mask
                        m_hard[accept_sample] = torch.max(m_hard[accept_sample], hard[accept_sample])
                    
                    # Budget-constrained stopping criteria
                    elif evaluation_mode == 'fixed-budget':
                        budget_allowed = kwargs['budget']

                        if sum(accept_sample) == 0:
                            # All samples have their max budget reached
                            budget_exhausted = True
                            break

                        check_neg_CMI = torch.max(pred_CMI, dim=1).values > 0
                        if sum(check_neg_CMI) == 0:
                            # max adjusted CMI for all samples is negative, this batch is done
                            budget_exhausted = True
                            break

                        # Once a sample is done, it should not be selected again
                        accept_sample = torch.bitwise_and(accept_sample, check_neg_CMI)

                        pred_CMI -= 1e6 * m_hard

                        best_feature_index = selection_func(pred_CMI, feature_costs, None)

                        # print(f"best feature index: {best_feature_index}")

                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Check if budget is still left to get more features
                        features_selected = torch.max(m_hard, hard)
                        budget_not_used_up = torch.sum(features_selected, dim=1) <= budget_allowed
                        accept_sample = torch.bitwise_and(accept_sample, budget_not_used_up)

                        # Update mask
                        m_hard[accept_sample] = torch.max(m_hard[accept_sample], hard[accept_sample])
                    
                    # Confidence constrained stopping criteria
                    elif evaluation_mode == 'confidence':
                        min_confidence = kwargs['min_confidence']
                        x_masked = mask_layer(x, m_hard)
                        pred = predictor(x_masked)

                        confidences = get_confidence(pred)
                        
                        # All samples have reached minimum confidence level or all features have been selected
                        if sum(accept_sample) == 0 or torch.max(torch.sum(m_hard, dim=1)) == mask_size:
                            budget_exhausted = True
                            break

                        check_neg_CMI = torch.max(pred_CMI, dim=1).values > 0
                        if sum(check_neg_CMI) == 0:
                            # max predicted CMI for all samples is negative, this batch is done
                            budget_exhausted = True
                            break

                        # Once a sample is done, it should not be selected again
                        accept_sample = torch.bitwise_and(accept_sample, check_neg_CMI)

                        # Select next feature and ensure no repeats
                        pred_CMI -= 1e6 * m_hard

                        best_feature_index = selection_func(pred_CMI, feature_costs, None)
                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Select samples for which minimum confidence has not been reached yet
                        confidence_not_reached = confidences < min_confidence
                        accept_sample = torch.bitwise_and(accept_sample, confidence_not_reached)

                        # Update mask
                        m_hard[accept_sample] = torch.max(m_hard[accept_sample], hard[accept_sample])
                    
                    if iteration not in masks_dict:
                        masks_dict[iteration] = [m_hard.cpu()]
                    else:
                        masks_dict[iteration].append(m_hard.cpu())
                    
                    iteration += 1

                # Add to final mask list
                final_masks.append(m_hard.cpu())

                # Make prediction
                x_masked = mask_layer(x, m_hard)
                pred = predictor(x_masked)

                pred_list.append(pred.cpu())
                y_list.append(y.cpu())

        masks_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), masks_dict.items()))
        pred_cmi_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), pred_cmi_dict.items()))
        entropy_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), entropy_dict.items()))

        # Return multiple metrics for downstream analysis
        return_dict = dict(
            performance=performance_func(torch.cat(pred_list), torch.cat(y_list)),
            final_masks=torch.cat(final_masks).detach().numpy(),
            masks_dict=masks_dict,
            pred_list=torch.cat(pred_list),
            y_list=torch.cat(y_list),
            pred_cmi_dict=pred_cmi_dict,
            entropy_dict=entropy_dict,
            loss_dict=loss_dict
        )
        return return_dict
            


            