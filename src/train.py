import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

import argparse
import os
import numpy as np

from config import *
from utilize import adjust_learning_rate, MyRandomHorizontalFlip
import net_loc
import res_loc
import dataloader


def train(config):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.arch == 'vgg':
        saliency_model = net_loc.Model(config.pretrained_model_path)
    elif config.arch == 'res':
        saliency_model = res_loc.Model(config.pretrained_model_path)
    else:
        raise NotImplementedError

    if (config.multi_GPU):
        saliency_model = nn.DataParallel(saliency_model)
        saliency_model.to(device)
    else:
        saliency_model.to(device)
    
    if (config.data_augment):
        data_augment = MyRandomHorizontalFlip()
    else:
        data_augment = None

    transforms_h = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_l = transforms.Compose([
        transforms.Resize((150, 200)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_target = transforms.Compose([
        transforms.Resize((19, 25)),
        transforms.ToTensor()
    ])
    
    train_dataset = dataloader.SALICON(
        config.imgs_train_path, 
        config.imgs_val_path, 
        config.maps_train_path, 
        config.maps_val_path, 
        augment=data_augment,
        transform_h=transforms_h, 
        transform_l=transforms_l, 
        target=transforms_target, 
        mode='train'
    )
    val_dataset = dataloader.SALICON(
        config.imgs_train_path, 
        config.imgs_val_path, 
        config.maps_train_path, 
        config.maps_val_path,
        transform_h=transforms_h, 
        transform_l=transforms_l, 
        target=transforms_target, 
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, pin_memory=True)
    
    criterion = nn.MSELoss(reduction='sum')
    #optimizer = optim.SGD(saliency_model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    optimizer = optim.Adam(saliency_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    saliency_model.train()

    log_train_loss = np.asarray([])
    log_val_loss = np.asarray([])

    for epoch in range(config.num_epochs):
        # adjust lr
        if ((epoch+1) % config.lr_decay == 0):
            optimizer = adjust_learning_rate(optimizer, epoch, config.lr_factor, config.lr_decay, config.lr)
        
        # training phase
        for iter_train, (img_high, img_low, maps) in enumerate(train_loader):

            img_high, img_low, maps = img_high.to(device), img_low.to(device), maps.to(device)

            optimizer.zero_grad()

            output = saliency_model(img_high, img_low)
            
            loss = criterion(output, maps)

            loss.backward()
            optimizer.step()

            log_train_loss = np.append(log_train_loss, loss.item())

            if (((iter_train+1) % config.display_iter) == 0):
                print("Train - Epoch {}/{}, iter_train {}: loss {:.4f}. lr: {}".format(epoch+1, config.num_epochs, iter_train+1, loss.item(), optimizer.param_groups[0]['lr']))
            
            if ((iter_train+1) % config.snapshot_iter == 0):
                # Validation Stage
                saliency_model.eval()
                validate_loss = 0
                val_data_size = 0

                for iter_val, (img_high, img_low, maps) in enumerate(val_loader):

                    img_high, img_low, maps = img_high.to(device), img_low.to(device), maps.to(device)

                    output = saliency_model(img_high, img_low)
                    
                    val_loss = criterion(output, maps)
                    validate_loss += val_loss.item()
                    val_data_size = iter_val + 1
                
                log_val_loss = np.append(log_val_loss, validate_loss/val_data_size)
                print('Validation - Epoch {}/{} - loss {:.4f}. lr: {}'.format(epoch+1, config.num_epochs, validate_loss/val_data_size, optimizer.param_groups[0]['lr']))

                # save
                print('Saving model to {} ...'.format(config.snapshots_folder), end=' ')
                model_path = config.snapshots_folder + config.model_name + "_" + str(epoch+1) + '_' + str(iter_train+1)+ '.pth'
                torch.save(saliency_model.module.state_dict() if config.multi_GPU else saliency_model.state_dict(), model_path)
                print('done!')

                saliency_model.train() 

        print('Saving log to {} ...'.format(config.log_folder), end=' ')
        np.save(config.log_folder + 'train_log', log_train_loss)
        np.save(config.log_folder + 'val_log', log_val_loss)
        print('done!')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

	# Input Parameters
    
    parser.add_argument('--imgs_train_path', type=str, default=imgs_train_path)
    parser.add_argument('--maps_train_path', type=str, default=maps_train_path)
    parser.add_argument('--imgs_val_path', type=str, default=imgs_val_path)
    parser.add_argument('--maps_val_path', type=str, default=maps_val_path)
    parser.add_argument('--pretrained_model_path', type=str, default=pretrained_model_path)
    parser.add_argument('--snapshots_folder', type=str, default=snapshots_path)
    parser.add_argument('--log_folder', type=str, default=log_path)
    parser.add_argument('--data_augment', type=bool, default=using_data_augment)
    parser.add_argument('--multi_GPU', type=bool, default=using_multi_gpu)
    parser.add_argument('--model_name', type=str, default=model_name)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--lr_factor', type=float, default=lr_factor)
    parser.add_argument('--lr_decay', type=int, default=lr_decay)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--num_epochs', type=int, default=nb_epoch)
    parser.add_argument('--train_batch_size', type=int, default=train_b_s)
    parser.add_argument('--val_batch_size', type=int, default=val_b_s)
    parser.add_argument('--display_iter', type=int, default=display_iter)
    parser.add_argument('--snapshot_iter', type=int, default=snapshot_iter)  
    parser.add_argument('--arch', type=str, default=arch)  

    config = parser.parse_args()
    
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.log_folder):
        os.mkdir(config.log_folder)
    
    train(config)
  
