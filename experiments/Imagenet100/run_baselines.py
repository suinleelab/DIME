import sys
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dime.data_utils import MaskLayer2d
from dime.vit import PredictorViT, SelectorViT
from dime import MaskingPretrainer
from dime.utils import StaticMaskLayer2d, ConcreteMask2d
from torchvision import transforms
from torchmetrics import Accuracy
import timm
sys.path.append('../')
sys.path.append('../../')
from baselines import cae, hardattention, dfs
from baseline_models.base_model import BaseModel
from baseline_models.hard_attention_model import HardAttention

vit_model_options = ['vit_small_patch16_224', 'vit_tiny_patch16_224', 'vit_base_patch16_224']
resnet_model_options = ['resnet18', 'resnet34', 'resnet50', 'resnet101']

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method', type=str, default='cae',
                    choices=['cae', 'hard_attn', 'dfs'])
parser.add_argument('--mask_width', type=int,
                    default=14,
                    choices=[7, 14],
                    help="Mask width to use in the mask layer")
parser.add_argument('--pretrained_model_name', type=str,
                    default='vit_small_patch16_224',
                    choices=vit_model_options+resnet_model_options,
                    help="Name of the pretrained model to use")
parser.add_argument('--pretrain_checkpoint', type=str,
                    default=None,
                    help="Name of the pretrained checkpoint to use")
parser.add_argument('--training_phase', type=str,
                    default='first',
                    help="Name of the trianing phase")

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()
    num_classes = 100
    acc_metric = Accuracy(task='multiclass', num_classes=num_classes)

    # Load train dataset, split into train/val
    image_size = 224
    mask_width = args.mask_width
    # network_type = args.backbone
    pretrained_model_name = args.pretrained_model_name

    mask_layer = MaskLayer2d(append=False, mask_width=mask_width, patch_size=image_size/mask_width)
        
    device = torch.device('cuda', args.gpu)
    dataset_path = "/projects/<labname>/<username>/ImageNet100"

    norm_constants = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Setup for data loading.
    if args.method == 'hard_attn':
        # Setup for data loading.
        transforms_train = transforms.Compose([
                            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

        transforms_test = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    else:
        # Setup for data loading.
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*norm_constants),
        ])

        transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(*norm_constants),
        ])

    # Get Datasets
    train_dataset_train_transforms = ImageFolder(dataset_path+'/train.X1', transforms_train)
    train_dataset_all_len = len(train_dataset_train_transforms)

    # Get train and val indices
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(train_dataset_all_len, size=int(train_dataset_all_len*0.1), replace=False))
    train_inds = np.setdiff1d(np.arange(train_dataset_all_len), val_inds)

    train_dataset = torch.utils.data.Subset(train_dataset_train_transforms, train_inds)

    train_dataset_test_transforms = ImageFolder(dataset_path+'/train.X1', transforms_test)
    val_dataset = torch.utils.data.Subset(train_dataset_test_transforms, val_inds)

    test_dataset = ImageFolder(dataset_path+'/val.X', transforms_test)

    print(f'Train samples = {len(train_dataset)}, val samples = {len(val_dataset)}, test samples = {len(test_dataset)}')

    # Prepare dataloaders.
    mbsize = 32
    train_dataloader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=mbsize, pin_memory=True, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=mbsize, pin_memory=True, drop_last=True, num_workers=4)

    d_in = image_size * image_size
    d_out = 100

    num_features = [1] + list(range(5, 40, 5))
    mask_width = 14
    patch_size = image_size / mask_width

    results_dict = {
            'acc': {},
            'features': {}
    }   

    if args.method == 'cae':
        num_restarts = 1
        
        for num in num_features:
            # Train model with differentiable feature selection.
            backbone = timm.create_model(pretrained_model_name, pretrained=True)
            model = PredictorViT(backbone, num_classes=100)
            # model = get_mlp_network(d_in, d_out)
            selector_layer = ConcreteMask2d(mask_width, patch_size, num)
            diff_selector = cae.DifferentiableSelector(model, selector_layer).to(device)
            diff_selector.fit(
                train_dataloader,
                val_dataloader,
                lr=1e-5,
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
            
            print(f"selected_features={selected_features}")
            # Prepare module to mask all but top features
            inds = torch.tensor(np.isin(np.arange(mask_width ** 2), selected_features) * 1, device=device)
            mask = inds.reshape(mask_width, mask_width)
            mask_layer = StaticMaskLayer2d(mask, patch_size)

            best_loss = np.inf
            for _ in range(num_restarts):
                # Train model.
                backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
                predictor = PredictorViT(backbone, num_classes=100)
                model = nn.Sequential(mask_layer, predictor)
                basemodel = BaseModel(model).to(device)
                basemodel.fit(
                    train_dataloader,
                    val_dataloader,
                    lr=1e-5,
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
        with open(f'results/mnist_{args.method}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
    
    elif args.method == 'hard_attn':
        # nf = 512
        # nz = 1024
        # nh = 2048
        nf = 256
        nz = 512
        nh = 1024
        classes = 100
        imsz = 224
        gz = 16
        nsfL = 6
        T = 15
        training_phase = args.training_phase

        # overwrite ccebal for first two training phases
        if training_phase == 'first':
            ccebal = 1
        elif training_phase == 'second':
            ccebal = 0
        elif training_phase == 'third':
            ccebal = 16

        model = HardAttention(T, nsfL, nf, nh, nz, classes, gz, imsz, ccebal, training_phase, args.pretrain_checkpoint).to(device)
        hardattention.HardAttentionTrainer(model,
                                           T,
                                           device,
                                           train_dataloader,
                                           val_dataloader,
                                           test_dataloader,
                                           nepochs=500,
                                           lr=0.001,
                                           tensorboard_file_name_suffix="hard_attn_logs",
                                           path="hard_attn_results",
                                           training_phase=training_phase)

    elif args.method == 'dfs':
        max_features = 50
        mask_layer = MaskLayer2d(append=False, mask_width=mask_width, patch_size=image_size/mask_width)
        backbone = timm.create_model(pretrained_model_name, pretrained=True)
        predictor = PredictorViT(backbone, num_classes=num_classes)
        selector = SelectorViT(backbone)

        # Pretrain predictor
        pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
        pretrain.fit(
            train_dataset,
            val_dataset,
            mbsize,
            lr=1e-5,
            min_lr=1e-5,
            nepochs=3,
            loss_fn=nn.CrossEntropyLoss(),
            patience=5,
            verbose=True)

        # Train selector and predictor jointly.
        gdfs = dfs.GreedyDynamicSelection(selector, predictor, mask_layer).to(device)
        gdfs.fit(
            train_dataloader,
            val_dataloader,
            lr=1e-5,
            min_lr=1e-5,
            nepochs=3,
            max_features=max_features,
            loss_fn=nn.CrossEntropyLoss(),
            patience=5,
            verbose=True)

        # Evaluate.
        for num in num_features:
            acc = gdfs.evaluate(test_dataloader, num, acc_metric)
            results_dict['acc'][num] = acc
            print(f'Num = {num}, Acc = {100*acc:.2f}')
       
        print(results_dict)
        with open(f'results/Imagenet100_{args.method}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        # Save model
        gdfs.cpu()
        torch.save(gdfs, f'results/Imagenet100_{args.method}.pt')