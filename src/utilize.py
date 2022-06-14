import torchvision.transforms.functional as F
from torchvision import transforms
import random
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
from tqdm import tqdm
import glob

# 学习率调整函数
def adjust_learning_rate(optimizer, epoch, lr_factor, lr_decay, lr):
    lr = lr * (lr_factor ** ((epoch+1) // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


# 自定义数据增强函数
class MyRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, gt):
        if random.random() < self.p:
            return F.hflip(img), F.hflip(gt)
        return img, gt
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def resize_img(img, aspect_ratio=0.75):
    w, h = img.size # w : h == 4 : 3
    if (h / w < aspect_ratio): # h变大
        pad = ( 0, int((w * aspect_ratio - h) / 2) )
    else: # w变大
        pad = ( int((h / aspect_ratio - w) / 2), 0 )
    pad_transform = transforms.Pad(padding=pad, fill=0)
    img = pad_transform(img) 
    return img


if __name__ == '__main__':
    img_list = glob.glob('E:\\数据集\\MIT1003\\train\\images - 副本\\' + '*')
    # for i in range(len(img_list)):
    #     img_path = img_list[i]
    #     img = Image.open(img_path)
    #     w, h = img.size
    #     print(h/w)
        
    # for i in tqdm(range(len(img_list)), ncols=80):
    #     img_path = img_list[i]
    #     img = Image.open(img_path)
    #     #img = resize_img(img)
    #     img_name = img_path.split('\\')[-1].split('.')[0]
    #     scipy.misc.imsave('E:\\数据集\\MIT1003\\padded\\val\\images\\' + img_name + '.jpg', img)


    for i in range(len(img_list))