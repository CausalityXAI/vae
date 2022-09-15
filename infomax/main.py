#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.simulation import (
    set_random_seed,
)

from utils.model import (
    VAE,
    Discriminator
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
    project="VAE", 
    entity="anseunghwan",
    tags=["InfoMax", "MNIST"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')

    parser.add_argument("--z_dim", default=2, type=int,
                        help="the number of nodes")
    
    parser.add_argument('--epochs', default=5, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--lr_D', default=0.00001, type=float,
                        help='learning rate')
    
    parser.add_argument('--beta', default=1, type=float,
                        help='observation noise')
    parser.add_argument('--gamma', default=10, type=float,
                        help='weight of f-divergence (lower bound of information)')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def permute_dims(z, device):
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z

def train(dataloader, model, discriminator, config, optimizer, optimizer_D, device):
    logs = {
        'loss': [], 
        'recon': [],
        'KL': [],
    }
    # for debugging
    for i in range(config["z_dim"]):
        logs['posterior_variance{}'.format(i+1)] = []
    
    for (x_batch, y_batch) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        
        if config["cuda"]:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        
        with torch.autograd.set_detect_anomaly(True):    
            xhat, mu, logvar, z = model(x_batch)
            
            loss_ = []
            
            """reconstruction"""
            n = x_batch.size(0)
            recon = F.binary_cross_entropy(xhat, x_batch, reduction='sum').div(n)
            loss_.append(('recon', recon))
            
            """KL-Divergence"""
            # -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
            KL = torch.pow(mu, 2).sum(axis=1)
            KL -= logvar.sum(axis=1)
            KL += torch.exp(logvar).sum(axis=1)
            KL -= config["z_dim"]
            KL *= 0.5
            KL = KL.mean()
            loss_.append(('KL', KL))
            
            """mutual information"""
            D_joint = discriminator(x_batch, z)
            z_perm = permute_dims(z, device)
            D_marginal = discriminator(x_batch, z_perm)
            MI = -(D_joint.mean() - torch.exp(D_marginal - 1).mean())
            
            ### posterior variance: for debugging
            logvar_ = logvar.mean(axis=0)
            for i in range(config["z_dim"]):
                loss_.append(('posterior_variance{}'.format(i+1), torch.exp(logvar_[i])))
            
            loss = recon + config["beta"] * KL + config["gamma"] * MI
            loss_.append(('loss', loss))
            
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            loss.backward(retain_graph=True)
            MI.backward()
            optimizer.step()
            optimizer_D.step()
            
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs, xhat
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])

    """dataset"""
    training_dataset = datasets.MNIST('./assets/MNIST', train=True, download=True, transform=transforms.ToTensor())
    # test_dataset = datasets.MNIST('./assets/MNIST', train=False, download=True, transform=transforms.ToTensor())

    dataloader = DataLoader(training_dataset, batch_size=config["batch_size"], shuffle=True)
    # testloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    
    model = VAE(config["z_dim"], device) 
    model = model.to(device)
    discriminator = Discriminator(config["z_dim"])
    discriminator = discriminator.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config["lr_D"]
    )
    
    wandb.watch(model, log_freq=1) # tracking gradients
    wandb.watch(discriminator, log_freq=1) # tracking gradients
    model.train()
    discriminator.train()
    
    for epoch in range(config["epochs"]):
        logs, xhat = train(dataloader, model, discriminator, config, optimizer, optimizer_D, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
            
        # if epoch % 10 == 0:
        plt.figure(figsize=(4, 4))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
            plt.axis('off')
        plt.savefig('./assets/tmp_image_{}.png'.format(epoch))
        plt.close()
    
    """reconstruction result"""
    fig = plt.figure(figsize=(4, 4))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow((xhat[i].cpu().detach().numpy() + 1) / 2)
        plt.axis('off')
    plt.savefig('./assets/recon.png')
    plt.close()
    wandb.log({'reconstruction': wandb.Image(fig)})
    
    """model save"""
    torch.save(model.state_dict(), './assets/model.pth')
    artifact = wandb.Artifact('model_{}'.format('infomax'), 
                              type='model',
                              metadata=config) # description=""
    artifact.add_file('./assets/model.pth')
    artifact.add_file('./main.py'.format('infomax'))
    artifact.add_file('./utils/model.py')
    wandb.log_artifact(artifact)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%