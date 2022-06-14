import torch 
import net_loc
import res_loc
import argparse

def check_load_model(config):
    device = torch.device('cpu')
    saliency_net = net_loc.Model().to(device)
    # saliency_net = res_loc.Model().to(device)
    saliency_net.load_state_dict(torch.load(config.model_path))

    saliency_net.eval()
    print('load model successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='C:\\Users\\lipengqian\\Desktop\\ACNet开源\\checkpoints\\acnet_vgg_2017.pth')
    
    config = parser.parse_args()
    check_load_model(config)