#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import sys
import random
import tqdm
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import utils
from utils.model import *
from utils.sagan import *
from utils.causal_model import *
from utils.viz import (
    viz_graph,
    viz_heatmap,
)
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

wandb.init(
    project="(causal)DEAR", 
    entity="anseunghwan",
    tags=["fully_supervised", "pendulum", "EDA"],
)
#%%
import argparse
def get_args(debug):
	parser = argparse.ArgumentParser('parameters')
 
	parser.add_argument('--num', type=int, default=2, 
						help='model version')

	if debug:
		return parser.parse_args(args=[])
	else:    
		return parser.parse_args()
#%%
# import yaml
# def load_config(args):
#     config_path = "./config/{}.yaml".format(args["dataset"])
#     with open(config_path, 'r') as config_file:
#         config = yaml.load(config_file, Loader=yaml.FullLoader)
#     for key in args.keys():
#         if key in config.keys():
#             args[key] = config[key]
#     return args
#%%
def main():
    #%%
    
    args = vars(get_args(debug=True))
    args["dataset"] = "pendulum"
    
    if args["dataset"] == "celeba":
        artifact = wandb.use_artifact('anseunghwan/(causal)DEAR/model_{}:v{}'.format(args["dataset"], args["num"]), type='model')
    else:
        artifact = wandb.use_artifact('anseunghwan/(causal)DEAR/model:v{}'.format(args["num"]), type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    args["cuda"] = torch.cuda.is_available()
    wandb.config.update(args)
    
    if 'pendulum' in args["dataset"]:
        label_idx = range(4)
    else:
        if args["labels"] == 'smile':
            label_idx = [31, 20, 19, 21, 23, 13]
        elif args["labels"] == 'age':
            label_idx = [39, 20, 28, 18, 13, 3]
        else:
            raise NotImplementedError("Not supported structure.")
    num_label = len(label_idx)

    random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["cuda"]:
        torch.cuda.manual_seed(args["seed"])
    global device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    #%%
    """dataset"""
    if args["dataset"] == "pendulum":
        class CustomDataset(Dataset): 
            def __init__(self, args):
                train_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/train') if x.endswith('png')]
                train_x = []
                for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                    train_x.append(np.transpose(
                        np.array(
                        Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])).resize((args["image_size"], args["image_size"]))
                        )[:, :, :3], (2, 0, 1)))
                self.x_data = (np.array(train_x).astype(float) - 127.5) / 127.5
                
                label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
                label = label - label.mean(axis=0)
                self.std = label.std(axis=0)
                """bounded label: normalize to (0, 1)"""
                if args["sup_type"] == 'ce': 
                    label = (label - label.min(axis=0)) / (label.max(axis=0) - label.min(axis=0))
                elif args["sup_type"] == 'l2': 
                    label = (label - label.mean(axis=0)) / label.std(axis=0)
                self.y_data = label
                self.name = ['light', 'angle', 'length', 'position']

            def __len__(self): 
                return len(self.x_data)

            def __getitem__(self, idx): 
                x = torch.FloatTensor(self.x_data[idx])
                y = torch.FloatTensor(self.y_data[idx])
                return x, y
        
        dataset = CustomDataset(args)
        train_loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
    
    elif args["dataset"] == "celeba": 
        train_loader = None
        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args["image_size"], args["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_dir = './utils/causal_data/celeba'
        if not os.path.exists(data_dir): 
            os.makedirs(data_dir)
        train_set = datasets.CelebA(data_dir, split='train', download=True, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args["batch_size"], 
                                                    shuffle=True, pin_memory=False,
                                                    drop_last=True, num_workers=0)
    #%%
    if 'scm' in args["prior"]:
        A = torch.zeros((num_label, num_label))
        if args["labels"] == 'smile':
            A[0, 2:6] = 1
            A[1, 4] = 1
        elif args["labels"] == 'age':
            A[0, 2:6] = 1
            A[1, 2:4] = 1
        elif args["labels"] == 'pend':
            A[0, 2:4] = 1
            A[1, 2:4] = 1
    else:
        A = None
    #%%
    """model load"""
    model_dir = artifact.download()
    model = BGM(
        args["latent_dim"], 
        args["g_conv_dim"], 
        args["image_size"],
        args["enc_dist"], 
        args["enc_arch"], 
        args["enc_fc_size"], 
        args["enc_noise_dim"], 
        args["dec_dist"],
        args["prior"], 
        num_label, 
        A
    )
    discriminator = BigJointDiscriminator(
        args["latent_dim"], 
        args["d_conv_dim"], 
        args["image_size"],
        args["dis_fc_size"]
    )
    if args["cuda"]:
        if args["dataset"] == "celeba":
            model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(args["dataset"])))
            discriminator.load_state_dict(torch.load(model_dir + '/discriminator_{}.pth'.format(args["dataset"])))
        else:
            model.load_state_dict(torch.load(model_dir + '/model.pth'))
            discriminator.load_state_dict(torch.load(model_dir + '/discriminator.pth'))
        model = model.to(device)
        discriminator = discriminator.to(device)
    else:
        if args["dataset"] == "celeba":
            model.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format(args["dataset"]), 
                                            map_location=torch.device('cpu')))
            discriminator.load_state_dict(torch.load(model_dir + '/discriminator_{}.pth'.format(args["dataset"]), 
                                                    map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_dir + '/model.pth', 
                                            map_location=torch.device('cpu')))
            discriminator.load_state_dict(torch.load(model_dir + '/discriminator.pth', 
                                                    map_location=torch.device('cpu')))
    #%%
    """estimated causal matrix"""
    print('DAG:{}'.format(model.prior.A))
    B_est = model.prior.A.detach().cpu().numpy()
    fig = viz_heatmap(np.flipud(B_est), size=(7, 7))
    wandb.log({'B_est': wandb.Image(fig)})
    #%%
    """do-intervention"""
    n = 9
    gap = 3
    traversals = torch.linspace(-gap, gap, steps=n)
    
    dim = model.num_label if model.num_label is not None else model.latent_dim
    z = torch.zeros(1, args["latent_dim"], device=device)
    z = z.expand(n, model.latent_dim)
    
    fig, ax = plt.subplots(4, 9, figsize=(10, 4))
    
    for idx in range(dim):
        z_inv = model.prior.enc_nlr(z)
        z_eps = model.prior.get_eps(z_inv)
        
        z_new = z.clone()
        z_new[:, idx] = traversals
        z_new_inv = model.prior.enc_nlr(z_new[:, :dim])
        
        for j in range(dim):
            if j == idx:
                continue
            else:
                z_new_inv[:, j] = torch.matmul(z[:, :j], model.prior.A[:j, j]) + z_eps[:, j]
        
        label_z = model.prior.prior_nlr(z_new_inv[:, :dim])
        other_z = z[:, dim:]
        
        with torch.no_grad():
            xhat = model.decoder(torch.cat([label_z, other_z], dim=1))
            for k in range(n):
                ax[idx, k].imshow((xhat[k].permute(1, 2, 0) + 1) / 2)
                ax[idx, k].axis('off')
    
    plt.savefig('{}/do.png'.format(model_dir), bbox_inches='tight')
    # plt.show()
    plt.close()
    
    wandb.log({'do intervention ({})'.format(', '.join(dataset.name)): wandb.Image(fig)})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%