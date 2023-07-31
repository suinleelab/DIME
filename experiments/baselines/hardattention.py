# Borrowed from https://github.com/samrudhdhirangrej/Probabilistic-Hard-Attention/blob/80c925afa4c7f9171b8c7690b2549468ec686531/nsf.py
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def HardAttentionTrainer(model, T, device, train_loader, val_loader, test_loader, nepochs, lr,
                         tensorboard_file_name_suffix="logs", path="results", training_phase="first"):
    
    optimizerG = optim.Adam(list([p for p in model.parameters() if p.requires_grad == True]), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizerG,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=25,
                                                     threshold=0.01,
                                                     threshold_mode='abs',
                                                     cooldown=0,
                                                     min_lr=0,
                                                     eps=1e-08)

    logger = SummaryWriter(log_dir=f"hard_attn_logs_phase_{training_phase}",
                           filename_suffix=tensorboard_file_name_suffix)

    for epoch in range(nepochs):
        logger.add_scalar('learning_rate/epoch_lr', optimizerG.param_groups[-1]['lr'], epoch)
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        model.train()
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            label = label.to(device)
            loss, acc = model(data, label)
            train_loss += (loss.item()*data.size(0))
            train_acc += (acc[-1].item()*data.size(0))
            optimizerG.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizerG.step()

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(tqdm(val_loader)):
                data = data.to(device)
                label = label.to(device)
                loss, acc = model(data, label)
                val_loss += (loss.item()*data.size(0))
                val_acc += (acc[-1].item()*data.size(0))

            # if (batch_idx%100)==0:
            #     with open(PATH+'/print.txt','a+') as f:
            #         f.write('epoch {} batch {}/{} loss flow {:.2f} acc {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n'.format(epoch, batch_idx, len(train_loader), loss.item(), *acc))

            # if batch_idx==10:
            #     break

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
    
        # with open(PATH+'/print.txt','a+') as f:
        print('final train loss {:.2f} acc {:.2f} epoch {:.2f} \n'.format(train_loss, train_acc, epoch))
        print('final val loss {:.2f} acc {:.2f} epoch {:.2f} \n'.format(val_loss, val_acc, epoch))
        
        os.makedirs(path, exist_ok=True)  # succeeds even if directory exists.

        torch.save([model.state_dict(), optimizerG.state_dict()],
                    path+'/weights_f_training_phase_{0}_imsz_224_lr_0.0001_detach_ccebal_1_auc.pth'.format(training_phase))
        
        np.savez(path+f'/train_loss_acc_{training_phase}_imsz_224_lr_0.0001_detach_ccebal_1_auc.pth', loss=train_loss, acc=train_acc)

        logger.add_scalar('loss/epoch_train', train_loss, epoch)
        logger.add_scalar('acc/epoch_train_{}'.format(T-1), train_acc, epoch)
        logger.add_scalar('loss/epoch_val', val_loss, epoch)
        logger.add_scalar('acc/epoch_val_{}'.format(T-1), val_acc, epoch)
        scheduler.step(train_loss)
    
        test_acc = [0 for _ in range(T)]
        model.eval()
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                _, acc = model.sample(data, label)    
            
            for i in range(T):
                test_acc[i] += (acc[i].item() * data.size(0))
        
            # if batch_idx%100==0:
            #     with open(PATH+'/print.txt','a+') as f:
            #         f.write('test epoch {} batch {}/{} acc {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n'.format(epoch, batch_idx, len(test_loader), *acc))

            # if batch_idx==10:
            #     break

        for i in range(T):
            test_acc[i] /= len(test_loader.dataset)
            logger.add_scalar('acc/epoch_test_{}'.format(i), test_acc[i], epoch)
        
            # with open(PATH+'/print.txt','a+') as f:
            print('final test acc {:.2f} epoch {:.2f}\n'.format(test_acc[i], epoch))
            
        np.savez(path+f'/test_acc_{training_phase}_imsz_224_lr_0.0001_detach_ccebal_1_auc.npz', acc = test_acc)

        
