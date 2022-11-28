#%%
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from modules.simulation import set_random_seed

from modules.model import TVAE

from modules.data_transformer import DataTransformer

from modules.train import train
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

run = wandb.init(
    project="CausalDisentangled", 
    entity="anseunghwan",
    tags=["Tabular", "TVAE"],
)
#%%
import argparse
import ast

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')

    parser.add_argument("--latent_dim", default=3, type=int,
                        help="the number of nodes")
    
    # optimization options
    parser.add_argument('--epochs', default=200, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.005 type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, 
                        help='weight decay parameter')
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """dataset"""
    df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df = df.drop(columns=['ID'])
    continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
    df = df[continuous].iloc[:4000]
    
    transformer = DataTransformer()
    transformer.fit(df)
    train_data = transformer.transform(df)
    dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(device))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False)    
    
    config["input_dim"] = transformer.output_dimensions
    #%%
    model = TVAE(config, device).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    #%%
    model.train()
    
    for epoch in range(config["epochs"]):
        logs = train(transformer.output_info_list, dataset, dataloader, model, config, optimizer, device)
        
        print_input = "[epoch {:03d}]".format(epoch + 1)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y)) for x, y in logs.items()])
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
    #%%
    # model.eval()
    # torch.manual_seed(config["seed"])
    # steps = dataset.__len__() // config["batch_size"] + 1
    # data = []
    # with torch.no_grad():
    #     for _ in range(steps):
    #         mean = torch.zeros(config["batch_size"], config["latent_dim"])
    #         std = mean + 1
    #         noise = torch.normal(mean=mean, std=std).to(device)
    #         fake = model.decoder(noise)
    #         fake = torch.tanh(fake)
    #         data.append(fake.numpy())
    # data = np.concatenate(data, axis=0)
    # data = data[:dataset.__len__()]
    # sample_df = transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    # #%%
    # from causallearn.search.ConstraintBased.PC import pc
    # from causallearn.utils.GraphUtils import GraphUtils
    
    # if not os.path.exists('./assets/loan'):
    #     os.makedirs('./assets/loan')
    
    # df_ = (df - df.mean(axis=0)) / df.std(axis=0)
    # train = df_.iloc[:4000]
    
    # cg = pc(data=train.to_numpy(), 
    #         alpha=0.05, 
    #         indep_test='chisq') 
    # print(cg.G)
    # trainG = cg.G.graph
    
    # # visualization
    # pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    # pdy.write_png('./assets/loan/dag_train_loan.png')
    # fig = Image.open('./assets/loan/dag_train_loan.png')
    # # wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    # #%%    
    # cg = pc(data=sample_df.to_numpy(), 
    #         alpha=0.05, 
    #         indep_test='fisherz') 
    # print(cg.G)
    
    # # SHD: https://arxiv.org/pdf/1306.1043.pdf
    # sampleSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    # nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    # flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    # nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    # sampleSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    # # wandb.log({'SHD (Sample)': sampleSHD})
    
    # # visualization
    # pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    # pdy.write_png('./assets/loan/dag_recon_sample_loan.png')
    # fig = Image.open('./assets/loan/dag_recon_sample_loan.png')
    # # wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    """model save"""
    torch.save(model.state_dict(), './assets/TVAE.pth')
    artifact = wandb.Artifact('TVAE', 
                            type='model',
                            metadata=config) # description=""
    artifact.add_file('./assets/TVAE.pth')
    artifact.add_file('./main.py')
    artifact.add_file('./modules/model.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%