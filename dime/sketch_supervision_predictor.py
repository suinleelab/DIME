import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from dime.utils import generate_uniform_mask, restore_parameters, get_entropy, get_confidence, selection_without_lamda, accuracy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def ind_to_onehot(inds, n):
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot

class SketchSupervisionPredictor(nn.Module):
    def __init__(self, value_network, trained_predictor, sketch_predictor, mask_layer):
        super().__init__()
        self.value_network = value_network
        self.trained_predictor = trained_predictor
        self.sketch_predictor = sketch_predictor
        self.mask_layer = mask_layer
    
    def fit(self,
            train_dataloader,
            val_dataloader,
            lr,
            nepochs,
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
            tensorboard_file_name_suffix="max_features_10_eps_0.1",
            eps_decay=False,
            feature_costs=None,
            lamda=None,
            use_entropy=True):
        value_network = self.value_network
        trained_predictor = self.trained_predictor
        sketch_predictor = self.sketch_predictor
        mask_layer = self.mask_layer

        writer = SummaryWriter(filename_suffix=tensorboard_file_name_suffix)

        device = next(trained_predictor.parameters()).device
        mse_loss_fn = nn.MSELoss(reduction='none')

        # For tracking best model.
        # best_value_network = None
        # best_predictor = None
        # best_loss = float('inf')

        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 2
        
        num_steps = 1
        best_val_loss_fn_output = 0
        global_step = 0
        best_predictor = None
        best_value_network = None

        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            print("hello")
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val_dataloader.dataset))
            assert len(x.shape) == 1
            mask_size = len(x)
        
        # Set up optimizer and scheduler
        opt = optim.Adam(list(sketch_predictor.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)

        # Freeze the value network backbone trained without using the sketch
        for param in value_network.parameters():
            param.requires_grad = False

        for param in trained_predictor.parameters():
            param.requires_grad = False

        for epoch in range(nepochs):
            batch_value_network_loss = []
            batch_pred_loss = []

            value_network.eval()
            trained_predictor.eval()
            sketch_predictor.train()

            # Train step
            for i, batch in enumerate(tqdm(train_dataloader)):
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, x_sketch, y = batch
                    x_sketch = x_sketch.to(device)

                x = x.to(device)
                y = y.to(device)
                sketch_predictor.zero_grad()

                # Create feature cost matrix on first iteration
                if epoch==0 and i==0 and num_steps==1:
                    if feature_costs is None:
                        feature_costs = torch.ones((len(x), mask_size), device=device)
                    else:
                        feature_costs = torch.tensor(feature_costs, device=device)
                        feature_costs = torch.tile(feature_costs, (x.shape[0], 1))
                    
                # print(f"Feature cost: {feature_costs[:, 0]}")

                # Setup.
                m_hard = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                pred_loss_total = 0

                x_masked = mask_layer(x, m_hard)
                pred_without_next_feature = trained_predictor(x_masked)

                for _ in range(max_features):
                    # Estimate CMI using value network
                    x_masked = mask_layer(x, m_hard)
                    pred_CMI = value_network(x_masked) * torch.tensor(get_entropy(pred_without_next_feature.detach()), device=device).unsqueeze(1) \
                                        if use_entropy else value_network(x_masked)

                    if lamda is None:
                        best = torch.argmax(pred_CMI/feature_costs, dim=1)
                    else:
                        best = torch.argmax(pred_CMI - lamda * feature_costs, dim=1)

                    # print(actions)
                    hard = ind_to_onehot(best, mask_size)
                    m_hard = torch.max(m_hard, hard)

                    # Predictor loss including the next feature
                    x_masked = mask_layer(x, m_hard)
                    pred = sketch_predictor(x_masked, x_sketch)
                    pred_with_next_feature = trained_predictor(x_masked)
                    pred_without_next_feature = pred_with_next_feature

                    loss = loss_fn(pred, y)
                    
                    total_loss = torch.mean(loss)
                    (total_loss / max_features).backward()
                    
                    pred_loss_total += torch.mean(loss)
                
                # Take optimizer step
                opt.step()
                batch_pred_loss.append(pred_loss_total / max_features)

            writer.add_scalar("eps", eps, global_step)
                
            # Validation step
            sketch_predictor.eval()
            pred_list = []
            y_list = []
            batch_pred_loss_val = []
            with torch.no_grad():

                # Val step
                for i, batch in enumerate(tqdm(val_dataloader)):
                    if len(batch) == 2:
                        x, y = batch
                    else:
                        x, x_sketch, y = batch
                        x_sketch = x_sketch.to(device)

                    x = x.to(device)
                    y = y.to(device)

                    # Setup.
                    m_hard = torch.zeros(len(x), mask_size, dtype=x.dtype, device=device)
                    pred_loss_total_val = 0
                    pred = None

                    for _ in range(max_features):
                        # Estimate CMI using value network
                        x_masked = mask_layer(x, m_hard)
                        pred_without_next_feature = trained_predictor(x_masked)
                        pred_CMI = value_network(x_masked) * torch.tensor(get_entropy(pred_without_next_feature.detach()), device=device).unsqueeze(1) \
                                if use_entropy else value_network(x_masked)

                        # Select next feature and ensure no repeats
                        pred_CMI -= 1e6 * m_hard
                        if lamda is None:
                            best_feature_index = torch.argmax(pred_CMI/feature_costs, dim=1)
                        else:
                            best_feature_index = torch.argmax(pred_CMI - lamda * feature_costs, dim=1)
                        # print(best_feature_index)
                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Update mask
                        m_hard = torch.max(m_hard, hard)

                        # Make prediction
                        x_masked = mask_layer(x, m_hard)
                        pred = sketch_predictor(x_masked, x_sketch)

                        pred_val_loss = loss_fn(pred, y)

                        # Calculate CE loss
                        pred_loss_total_val += torch.mean(pred_val_loss)

                    pred_list.append(pred.cpu())
                    y_list.append(y.cpu())
                    
                    batch_pred_loss_val.append(pred_loss_total_val / max_features)
                    
            val_loss_fn_output = val_loss_fn(torch.cat(y_list), torch.cat(pred_list))

            pred_loss_epoch_train = sum(batch_pred_loss) / len(train_dataloader)
            pred_loss_epoch_val = sum(batch_pred_loss_val) / len(val_dataloader)
            if epoch % 1 == 0:
                value_network_loss_epoch_train = sum(batch_value_network_loss) / len(train_dataloader)
                print(f"Epoch: {epoch} Predictor Loss Train: {pred_loss_epoch_train}")
                print(f"Predictor Loss Val: {pred_loss_epoch_val} {val_loss_fn.__name__}:{val_loss_fn_output}")
                writer.add_scalar("Predictor Loss/Train", pred_loss_epoch_train, global_step)
                writer.add_scalar("Predictor Loss/Val", pred_loss_epoch_val, global_step)
                writer.add_scalar("Predictor AUC/Val", val_loss_fn_output, global_step)


            scheduler.step(pred_loss_epoch_val)
            global_step += 1

            if val_loss_fn_output > best_val_loss_fn_output:
                best_val_loss_fn_output = val_loss_fn_output
                torch.save(sketch_predictor.state_dict(), f'results/sketch_predictor_trained_{tensorboard_file_name_suffix}.pth')

                value_network.to(device)
            
            # Check if best model.
            if pred_loss_epoch_val == scheduler.best:
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # if num_bad_epochs > patience + 1:
            #     if eps_decay:
            #         # Val loss not improving, reduce epsilon and reset learning rate
            #         print("Decaying eps")
            #         print(f"Old eps: {eps}")
            #         # eps = eps * ((1 - eps_decay_rate) ** time)
            #         # time += 5
            #         eps *= 0.5
            #         print(f"New eps: {eps}")
            #         for g in opt.param_groups:
            #             g['lr'] = 1e-3
            #         num_bad_epochs = 0
            #     # else:
            #     #     if verbose:
            #     #         print(f'Stopping early at epoch {epoch+1}')
            #     #     break

            # if eps < min_eps and eps_decay:
            #     break

            if num_bad_epochs > early_stopping_epochs:
                # Val loss is not improving, decay epsilon and restart training
                eps *= eps_decay_rate
                num_bad_epochs = 0
                num_steps += 1
                break
                
            writer.flush()
        
    def evaluate(self, 
                 test_dataloader, 
                 performance_func, 
                 feature_costs=None, 
                 selection_func=selection_without_lamda, 
                 evaluation_mode="lamda-penalty",
                 use_entropy=True,
                 semi_supervised=False,
                 **kwargs):
        '''
        Evaluate the value network and get predictions with stopping criteria
        '''
        #setup
        value_network = self.value_network
        predictor = self.trained_predictor
        sketch_predictor = self.sketch_predictor

        mask_layer = self.mask_layer
        loss = nn.CrossEntropyLoss()

        # Determine mask size.
        if hasattr(mask_layer, 'mask_size') and (mask_layer.mask_size is not None):
            print('hello')
            mask_size = mask_layer.mask_size
        else:
            # Must be tabular (1d data).
            x, y = next(iter(test_dataloader.dataset))
            assert len(x.shape) == 1
            mask_size = len(x)

        value_network.eval()
        predictor.eval()

        device = next(predictor.parameters()).device
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
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, x_sketch, y = batch
                    x_sketch = x_sketch.to(device)

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
                    # Estimate CMI using value network
                    x_masked = mask_layer(x, m_hard)

                    # if not semi_supervised:
                    pred = predictor(x_masked)

                    pred_CMI = value_network(x_masked) * torch.tensor(get_entropy(pred.detach()), device=device).unsqueeze(1) \
                                        if use_entropy else value_network(x_masked)
                    # else:
                    #     pred = predictor(x_masked, x_sketch)

                    #     pred_CMI = value_network(x_masked, x_sketch) * torch.tensor(get_entropy(pred.detach()), device=device).unsqueeze(1) \
                    #                         if use_entropy else value_network(x_masked, x_sketch)
                    

                    if iteration not in pred_cmi_dict:
                        pred_cmi_dict[iteration] = [pred_CMI.cpu()]
                    else:
                        pred_cmi_dict[iteration].append(pred_CMI.cpu())

                    if iteration not in entropy_dict:
                        entropy_dict[iteration] = [torch.tensor(get_entropy(pred))]
                    else:
                        entropy_dict[iteration].append(torch.tensor(get_entropy(pred)))

                    if iteration not in loss_dict:
                        loss_dict[iteration] = [loss(pred, y)]
                    else:
                        loss_dict[iteration].append(loss(pred, y))

                    if evaluation_mode == 'lamda-penalty':
                        lamda = kwargs['lamda']
                        adjusted_CMI = pred_CMI - lamda * feature_costs
                        check_neg_CMI = torch.max(adjusted_CMI, dim=1).values > 0

                        if sum(check_neg_CMI) == 0 or torch.max(torch.sum(m_hard, dim=1)) == mask_size:
                            # max adjusted CMI for all samples is negative, this batch is done
                            budget_exhausted = True
                            break

                        # Once a sample is done, it should not be selected again
                        accept_sample = torch.bitwise_and(accept_sample, check_neg_CMI) 
                        # print(f"accept sample={accept_sample}")

                        # Select next feature and ensure no repeats
                        pred_CMI -= 1e6 * m_hard
                        # print(pred_CMI)
                        best_feature_index = selection_func(pred_CMI, feature_costs, lamda)
                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Update mask
                        m_hard[accept_sample] = torch.max(m_hard[accept_sample], hard[accept_sample])
                    
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
                        
                        # Select next feature and ensure no repeats
                        pred_CMI -= 1e6 * m_hard

                        best_feature_index = selection_func(pred_CMI, feature_costs, None)

                        # print(f"best feature index: {best_feature_index}")

                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Check if budget is still left to get more features
                        features_selected = torch.max(m_hard, hard)
                        budget_not_used_up = torch.sum(features_selected * feature_costs, dim=1) <= budget_allowed
                        accept_sample = torch.bitwise_and(accept_sample, budget_not_used_up)

                        # print(f"accept sample: {accept_sample}")
                        # Update mask
                        m_hard[accept_sample] = torch.max(m_hard[accept_sample], hard[accept_sample])

                    elif evaluation_mode == 'entropy':
                        min_entropy = kwargs['min_entropy']
                        x_masked = mask_layer(x, m_hard)
                        if not semi_supervised:
                            pred = predictor(x_masked)
                        else:
                            pred = predictor(x_masked, x_sketch)

                        entropies = get_entropy(pred)
                        
                        if sum(accept_sample) == 0:
                            budget_exhausted = True
                            break

                        check_neg_CMI = torch.max(pred_CMI, dim=1).values > 0
                        if sum(check_neg_CMI) == 0:
                            # max adjusted CMI for all samples is negative, this batch is done
                            budget_exhausted = True
                            break

                        # Once a sample is done, it should not be selected again
                        accept_sample = torch.bitwise_and(accept_sample, check_neg_CMI)

                        # Select next feature and ensure no repeats
                        pred_CMI -= 1e6 * m_hard

                        # Select samples for which minimum entropy has not been reached yet
                        min_entropy_not_reached = entropies > min_entropy
                        accept_sample = torch.bitwise_and(accept_sample, torch.tensor(min_entropy_not_reached, device=device))

                        best_feature_index = selection_func(pred_CMI, feature_costs, None)
                        hard = ind_to_onehot(best_feature_index, mask_size)

                        # Update mask
                        m_hard[accept_sample] = torch.max(m_hard[accept_sample], hard[accept_sample])
                    
                    elif evaluation_mode == 'confidence':
                        min_confidence = kwargs['min_confidence']
                        x_masked = mask_layer(x, m_hard)
                        if not semi_supervised:
                            pred = predictor(x_masked)
                        else:
                            pred = predictor(x_masked, x_sketch)

                        confidences = get_confidence(pred)
                        # print(confidences)
                        
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
                if not semi_supervised:
                    pred = predictor(x_masked)
                else:
                    pred = sketch_predictor(x_masked, x_sketch)

                pred_list.append(pred.cpu())
                y_list.append(y.cpu())

        # print(masks_dict[0].shape)
        # Concatenate mask dict entries
        masks_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), masks_dict.items()))
        pred_cmi_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), pred_cmi_dict.items()))
        entropy_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), entropy_dict.items()))
        # loss_dict = dict(map(lambda kv: (kv[0], torch.cat(kv[1]).detach().numpy()), loss_dict.items()))

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
