from torchvision import transforms
from dime.data_utils import HistopathologyDownsampledEdgeDataset
import timm
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AUROC


if __name__ == "__main__":
    auc_metric = AUROC(task='multiclass', num_classes=2)
    run_description = "vit_no_mask_canny_edge_image"
    writer = SummaryWriter(filename_suffix=run_description)
    norm_constants = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    image_size = 224

    # Setup for data loading.
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants)
    ])

    transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*norm_constants)
    ])

    data_dir = '/projects/<labname>/<username>/hist_data/MHIST/'

    # Get train and test datasets
    df = pd.read_csv(data_dir + 'annotations.csv')
    train_dataset = HistopathologyDownsampledEdgeDataset(data_dir + 'images/', df.loc[df['Partition'] == 'train'],
                                                         transforms_train)
    test_dataset = HistopathologyDownsampledEdgeDataset(data_dir + 'images/', df.loc[df['Partition'] == 'test'],
                                                        transforms_test)
    test_dataset_len = len(test_dataset)

    # Split test dataset into val
    np.random.seed(0)
    val_inds = np.sort(np.random.choice(test_dataset_len, size=int(test_dataset_len*0.5), replace=False))
    test_inds = np.setdiff1d(np.arange(test_dataset_len), val_inds)

    val_dataset = torch.utils.data.Subset(test_dataset, val_inds)
    test_dataset = torch.utils.data.Subset(test_dataset, test_inds)

    # Prepare dataloaders.
    mbsize = 32
    train_dataloader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=mbsize, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=mbsize, pin_memory=True, num_workers=4)

    device = torch.device('cuda:1')
    model = timm.create_model("vit_small_patch16_224", pretrained=True)
    model.head = torch.nn.Linear(model.embed_dim, 2)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.2, patience=5,
            min_lr=1e-8, verbose=True)

    for epoch in range(250):
        model.train()
        train_batch_loss = 0
        val_batch_loss = 0
        val_pred_list = []
        val_y_list = []

        for i, (x, x_sketch, y) in enumerate(tqdm(train_dataloader)):
            x = x_sketch.to(device)
            y = y.to(device)
            
            pred = model(x)
            train_loss = criterion(pred, y)
            train_batch_loss += train_loss.item()
            train_loss.backward()
            opt.step()
            model.zero_grad()
        
        model.eval()

        with torch.no_grad():
            for i, (x, x_sketch, y) in enumerate(tqdm(val_dataloader)):
                x = x_sketch.to(device)
                y = y.to(device)
            
                pred = model(x)
                val_loss = criterion(pred, y)
                val_batch_loss += val_loss.item()
                val_pred_list.append(pred.cpu())
                val_y_list.append(y.cpu())

            scheduler.step(val_batch_loss/len(val_dataloader))
        
        writer.add_scalar("Loss/Train", train_batch_loss/len(train_dataloader), epoch)
        writer.add_scalar("Loss/Val", val_batch_loss/len(val_dataloader), epoch)
        writer.add_scalar("Performance/Val", auc_metric(torch.cat(val_pred_list), torch.cat(val_y_list)), epoch)

        print(f"Epoch: {epoch}, Train Loss: {train_batch_loss/len(train_dataloader)}, \
              Val Loss: {val_batch_loss/len(val_dataloader)}, \
              Val Performance: {auc_metric(torch.cat(val_pred_list), torch.cat(val_y_list))}")