#%%
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
#%%
"""
Reference:
[1]: https://github.com/huawei-noah/trustworthyAI/blob/master/research/CausalVAE/inference_pendeulum.py
"""
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torch.utils import data
import torch.utils.data as Data
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import random
from pprint import pprint
from PIL import Image
import os
import tqdm
import pandas as pd

from utils.util import _h_A
import utils.util as ut
from utils.mask_vae_pendulum import CausalVAE
from utils.viz import (
    viz_graph,
    viz_heatmap,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
    project="(causal)CausalVAE", 
    entity="anseunghwan",
    tags=["Metric", "pendulum"],
)
#%%
import argparse
def get_args(debug):
	parser = argparse.ArgumentParser('parameters')
	# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
	parser.add_argument('--num', type=int, default=0, 
						help='model version')

	if debug:
		return parser.parse_args(args=[])
	else:    
		return parser.parse_args()
#%%
def main():
    #%%
    
    args = vars(get_args(debug=True)) # default configuration

    """model load"""
    artifact = wandb.use_artifact('anseunghwan/(causal)CausalVAE/model_{}:v{}'.format('CausalVAE', args["num"]), type='model')
    for key, item in artifact.metadata.items():
        args[key] = item
    pprint(args)
    model_dir = artifact.download()
    
    args["cuda"] = torch.cuda.is_available()
    global device
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    wandb.config.update(args)

    ut.set_random_seed(args["seed"])
    torch.manual_seed(args["seed"])
    if args["cuda"]:
        torch.cuda.manual_seed(args["seed"])

    lvae = CausalVAE(z_dim=args["z_dim"], device=device, image_size=args["image_size"]).to(device)
    if args["cuda"]:
        lvae.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('CausalVAE')))
    else:
        lvae.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('CausalVAE'), map_location=torch.device('cpu')))
    #%%
 
    """dataset"""
    class CustomDataset(Dataset): 
        def __init__(self, args):
            train_imgs = [x for x in os.listdir('./utils/causal_data/pendulum/train') if x.endswith('png')]
            train_x = []
            for i in tqdm.tqdm(range(len(train_imgs)), desc="train data loading"):
                train_x.append(np.array(
                    Image.open("./utils/causal_data/pendulum/train/{}".format(train_imgs[i])).resize((args["image_size"], args["image_size"]))
                    )[:, :, :3])
            self.x_data = np.array(train_x).astype(float) / 255.
            
            label = np.array([x[:-4].split('_')[1:] for x in train_imgs]).astype(float)
            self.std = label.std(axis=0)
            """label standardization"""
            if args["label_standardization"]: 
                label -= label.mean(axis=0)
                label /= label.std(axis=0)
            self.y_data = label
            self.name = ['light', 'angle', 'length', 'position']

        def __len__(self): 
            return len(self.x_data)

        def __getitem__(self, idx): 
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y

    dataset = CustomDataset(args)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
    #%%
    """import baseline classifier"""
    artifact = wandb.use_artifact('anseunghwan/(proposal)CausalVAE/model_classifier:v{}'.format(0), type='model')
    model_dir = artifact.download()
    from utils.model_classifier import Classifier
    """masking"""
    # if args["dataset"] == 'pendulum':
    mask = []
    # light
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[:20, ...] = 1
    mask.append(m)
    # angle
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[20:51, ...] = 1
    mask.append(m)
    # shadow
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    m = torch.zeros(args["image_size"], args["image_size"], 3)
    m[51:, ...] = 1
    mask.append(m)
    
    args["node"] = 4
        
    # elif args["dataset"] == 'celeba':
    #     raise NotImplementedError('Not yet for CELEBA dataset!')
    
    classifier = Classifier(mask, args, device) 
    if args["cuda"]:
        classifier.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('classifier')))
    else:
        classifier.load_state_dict(torch.load(model_dir + '/model_{}.pth'.format('classifier'), map_location=torch.device('cpu')))
    #%%
    """intervention range"""
    decode_m_max = []
    decode_m_min = []
    # e_tilde_max = []
    # e_tilde_min = []
    f_z1_max = []
    f_z1_min = []
    for u, l in tqdm.tqdm(dataloader):
        if args["cuda"]:
            u = u.cuda()
            l = l.cuda()
        
        with torch.no_grad():
            _, _, decode_m, _, _, e_tilde, f_z1, _, _, _ = lvae.encode(u, l, sample=False)
            
        decode_m_max.append(decode_m.max().detach().cpu().numpy())
        decode_m_min.append(decode_m.min().detach().cpu().numpy())
        # e_tilde_max.append(e_tilde.max().detach().cpu().numpy())
        # e_tilde_min.append(e_tilde.min().detach().cpu().numpy())
        f_z1_max.append(f_z1.max().detach().cpu().numpy())
        f_z1_min.append(f_z1.min().detach().cpu().numpy())
    decode_m_range = (min(decode_m_min), max(decode_m_max))
    # e_tilde_range = (min(e_tilde_min), max(e_tilde_max))
    f_z1_range = (min(f_z1_min), max(f_z1_max))
    causal_range = [decode_m_range, f_z1_range]
    #%%
    """metric"""
    dim = 4
    ACE_dict_lower = {x:[] for x in dataset.name}
    ACE_dict_upper = {x:[] for x in dataset.name}
    s = 'length'
    c = 'light'
    for s in ['light', 'angle', 'length', 'position']:
        for c in ['light', 'angle', 'length', 'position']:
            ACE_lower = 0
            ACE_upper = 0
            
            dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
            for x_batch, y_batch in tqdm.tqdm(iter(dataloader)):
                if args["cuda"]:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                do_index = dataset.name.index(s)
                
                with torch.no_grad():
                    score = []
                    if do_index < 2:
                        for val in [causal_range[0][0], causal_range[0][1]]:
                            _, _, _, _, reconstructed_image, _= lvae.negative_elbo_bound(u, l, do_index, sample = False, adj=torch.tensor(val))
                            reconstructed_image = torch.sigmoid(reconstructed_image)
                            
                            """factor classification"""
                            score.append(torch.sigmoid(classifier(reconstructed_image))[:, dataset.name.index(c)])
                            
                    else:
                        for val in [causal_range[1][0], causal_range[1][1]]:
                            _, _, _, _, reconstructed_image, _= lvae.negative_elbo_bound(u, l, do_index, sample = False, adj=torch.tensor(val))
                            reconstructed_image = torch.sigmoid(reconstructed_image)
                            
                            """factor classification"""
                            score.append(torch.sigmoid(classifier(reconstructed_image))[:, dataset.name.index(c)])
                    
                    ACE_lower += (score[0] - score[1]).sum()
                    ACE_upper += (score[0] - score[1]).abs().sum()
                    
            ACE_lower /= dataset.__len__()
            ACE_upper /= dataset.__len__()
            ACE_dict_lower[s] = ACE_dict_lower.get(s) + [(c, ACE_lower.abs().item())]
            ACE_dict_upper[s] = ACE_dict_upper.get(s) + [(c, ACE_upper.item())]
    #%%
    ACE_mat_lower = np.zeros((dim, dim))
    for i, c in enumerate(dataset.name):
        ACE_mat_lower[i, :] = [x[1] for x in ACE_dict_lower[c]]
    ACE_mat_upper = np.zeros((dim, dim))
    for i, c in enumerate(dataset.name):
        ACE_mat_upper[i, :] = [x[1] for x in ACE_dict_upper[c]]
    
    fig = viz_heatmap(np.flipud(ACE_mat_lower), size=(7, 7))
    wandb.log({'ACE(lower)': wandb.Image(fig)})
    fig = viz_heatmap(np.flipud(ACE_mat_upper), size=(7, 7))
    wandb.log({'ACE(upper)': wandb.Image(fig)})
    
    # save as csv
    pd.DataFrame(ACE_mat_lower.round(3), columns=dataset.name, index=dataset.name).to_csv('./assets/ACE_lower_causalvae.csv')
    pd.DataFrame(ACE_mat_upper.round(3), columns=dataset.name, index=dataset.name).to_csv('./assets/ACE_upper_causalvae.csv')
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%