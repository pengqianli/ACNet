import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import glob
import argparse
import os
import time
from tqdm import tqdm

import net_loc


def inference_maps(config):
    img_list = glob.glob(config.image_path + '*.jpg')

    device = torch.device('cpu')

    high_transforms = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(), 
    ])
    low_transforms = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),  
    ])
    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    saliency_net = net_loc.Model().to(device)
    saliency_net.load_state_dict(torch.load(config.model_path))

    saliency_net.eval()

    num_img = len(img_list)
    for i in tqdm(range(num_img), ncols=80):
        each_img = img_list[i]
        img = Image.open(each_img)
        w, h = img.size
 
        img_h = high_transforms(img)
        img_l = low_transforms(img)

        if (img_h.size(0) == 1):
            img_h = torch.cat((img_h, img_h, img_h), 0)
            img_l = torch.cat((img_l, img_l, img_l), 0)

            img_h = transform(img_h)
            img_l = transform(img_l)
        else:
            img_h = transform(img_h)
            img_l = transform(img_l)
 
        img_h = torch.unsqueeze(img_h, 0).to(device)
        img_l = torch.unsqueeze(img_l, 0).to(device)

        img_name = each_img.split('\\')[-1]

        prediction = saliency_net(img_h, img_l)

        saliency_map = prediction.cpu()
        saliency_map = saliency_map.detach().numpy()
        saliency_map = scipy.misc.imresize(saliency_map[0, 0, :, :], (h, w), interp='bicubic')

        scipy.misc.imsave(config.pred_folder + img_name, saliency_map)

def get_img(config, model_list):
    if (model_list != []):
        for model_path in model_list:
            config.model_path = model_path
            model_name = model_path.split('\\')[-1].split('.')[0]
            config.pred_folder = '/xxx/Pytorch/MyNet/ACNet/predictions/' + model_name + '/'
            if not os.path.exists(config.pred_folder):
                os.mkdir(config.pred_folder)
            print(config.model_path)
            inference_maps(config)
    else:
        inference_maps(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, default='/xxx/Datasets/FP/Toronto/image/') # your test images' path 
    parser.add_argument('--model_path', type=str, default='/xxx/MyNet/ACNet/ckpt/model_name.pth') # trained model checkpoint's path
    parser.add_argument('--pred_folder', type=str, default='/xxx/MyNet/ACNet/2020/Toronto/') # path to save predicted results 
    parser.add_argument('--show_results', type=bool, default=False)

    config = parser.parse_args()

    model_list = sorted(glob.glob('/xxx/Pytorch/MyNet/ACNet/snapshots/' + '*'))
    #model_list = []
    
    if not os.path.exists(config.pred_folder):
        os.mkdir(config.pred_folder)
    
    inference_maps(config)
    #get_img(config, model_list)
            