#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# training batch size
train_b_s = 4
# validating batch size
val_b_s = 4
# number of epochs
nb_epoch = 20
# learining rate
lr = 0.0001
# learining rate decay
lr_decay = 5
lr_factor = 0.1
# weight_decay
weight_decay = 0.0005
# momentum
momentum = 0.9
# display intermediate results 
display_iter = 50
# saving intermediate models
snapshot_iter = 625
# if using multi GPUs
using_multi_gpu = False
# if using data augment
using_data_augment = False
# number of rows of input images
shape_r_in = 480
# number of cols of input images
shape_c_in = 640
# number of rows of downsampled maps
shape_r_gt = 30
# number of cols of downsampled maps
shape_c_gt = 40
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# backbone
arch = 'vgg'

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = "/xxx/saliency/SALICON/2015/images/train/"
# path of training maps
maps_train_path = "/xxx/saliency/SALICON/2015/maps/train/"
# path of training fixation maps
fixs_train_path = "/xxx/saliency/SALICON/2015/fixations/train/"
# path of validation images
imgs_val_path = "/xxx/saliency/SALICON/2015/images/val/"
# path of validation maps
maps_val_path = "/xxx/saliency/SALICON/2015/maps/val/"
# path of validation fixation maps
fixs_val_path = "/xxx/saliency/SALICON/2015/fixations/val/"
# pre-trained model path
pretrained_model_path = "/xxx/pretrain_model/vgg16-397923af.pth"

#########################################################################
# Fine-tune MIT1003 SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = "/xxx/saliency/SALICON/2015/images/train/"
# path of training maps
maps_train_path = "/xxx/saliency/SALICON/2015/maps/train/"
# path of training fixation maps
fixs_train_path = "/xxx/saliency/SALICON/2015/fixations/train/"
# path of validation images
imgs_val_path = "/xxx/saliency/SALICON/2015/images/val/"
# path of validation maps
maps_val_path = "/xxx/saliency/SALICON/2015/maps/val/"
# path of validation fixation maps
fixs_val_path = "/xxx/saliency/SALICON/2015/fixations/val/"
# pre-trained model path
pretrained_model_path = "/xxx/pretrain_model/vgg16-397923af.pth"

#########################################################################
# SAVING SETTINGS										            	#
#########################################################################
# model name
model_name = 'acnet'
# path of saving models
snapshots_path = "./" + model_name + "_snapshots/"
# path of saving logs
log_path = "./" + model_name + "_logs/"