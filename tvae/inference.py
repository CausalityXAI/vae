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
    tags=["Tabular", "TVAE", "Inference"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--num', type=int, default=0, 
                        help='model version')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    #%%
    config = vars(get_args(debug=False)) # default configuration
    
    """model load"""
    artifact = wandb.use_artifact(
        'anseunghwan/CausalDisentangled/TVAE:v{}'.format(config["num"]), type='model')
    for key, item in artifact.metadata.items():
        config[key] = item
    model_dir = artifact.download()
    
    config["cuda"] = torch.cuda.is_available()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    #%%
    """model"""
    model = TVAE(config, device).to(device)
    
    if config["cuda"]:
        model.load_state_dict(
            torch.load(
                model_dir + '/TVAE.pth'))
    else:
        model.load_state_dict(
            torch.load(
                model_dir + '/TVAE.pth', 
                    map_location=torch.device('cpu')))
    
    model.eval()
    #%%
    df = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    df = df.drop(columns=['ID'])
    continuous = ['CCAvg', 'Mortgage', 'Income', 'Experience', 'Age']
    df = df[continuous]
    
    df_ = (df - df.mean(axis=0)) / df.std(axis=0)
    train = df_.iloc[:4000]
    test = df_.iloc[4000:]
    
    transformer = DataTransformer()
    transformer.fit(df.iloc[:4000])
    train_data = transformer.transform(df.iloc[:4000])
    dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(device))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=False)    
    
    config["input_dim"] = transformer.output_dimensions
    #%%
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    
    if not os.path.exists('./assets/loan'):
        os.makedirs('./assets/loan')
    
    cg = pc(data=train.to_numpy(), 
            alpha=0.05, 
            indep_test='chisq') 
    print(cg.G)
    trainG = cg.G.graph
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=df.columns)
    pdy.write_png('./assets/loan/dag_train_loan.png')
    fig = Image.open('./assets/loan/dag_train_loan.png')
    wandb.log({'Baseline DAG (Train)': wandb.Image(fig)})
    #%%
    train_recon = []
    for (x_batch, ) in tqdm.tqdm(iter(dataloader), desc="inner loop"):
        if config["cuda"]:
            x_batch = x_batch.cuda()
        
        with torch.no_grad():
            out = model(x_batch, deterministic=True)
        train_recon.append(out[-1])
    train_recon = torch.cat(train_recon, dim=0)
    train_recon = transformer.inverse_transform(train_recon, model.sigma.detach().cpu().numpy())
    #%%
    cg = pc(data=train_recon.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    trainSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    trainSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    wandb.log({'SHD (Train)': trainSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=train_recon.columns)
    pdy.write_png('./assets/loan/dag_recon_train_loan.png')
    fig = Image.open('./assets/loan/dag_recon_train_loan.png')
    wandb.log({'Reconstructed DAG (Train)': wandb.Image(fig)})
    #%%
    torch.manual_seed(config["seed"])
    steps = len(train) // config["batch_size"] + 1
    data = []
    with torch.no_grad():
        for _ in range(steps):
            mean = torch.zeros(config["batch_size"], config["latent_dim"])
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(device)
            fake = model.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.numpy())
    data = np.concatenate(data, axis=0)
    data = data[:len(train)]
    sample_df = transformer.inverse_transform(data, model.sigma.detach().cpu().numpy())
    #%%
    cg = pc(data=sample_df.to_numpy(), 
            alpha=0.05, 
            indep_test='fisherz') 
    print(cg.G)
    
    # SHD: https://arxiv.org/pdf/1306.1043.pdf
    sampleSHD = (np.triu(trainG) != np.triu(cg.G.graph)).sum() # unmatch in upper-triangular
    nonzero_idx = np.where(np.triu(cg.G.graph) != 0)
    flag = np.triu(trainG)[nonzero_idx] == np.triu(cg.G.graph)[nonzero_idx]
    nonzero_idx = (nonzero_idx[1][flag], nonzero_idx[0][flag])
    sampleSHD += (np.tril(trainG)[nonzero_idx] != np.tril(cg.G.graph)[nonzero_idx]).sum()
    wandb.log({'SHD (Sample)': sampleSHD})
    
    # visualization
    pdy = GraphUtils.to_pydot(cg.G, labels=sample_df.columns)
    pdy.write_png('./assets/loan/dag_recon_sample_loan.png')
    fig = Image.open('./assets/loan/dag_recon_sample_loan.png')
    wandb.log({'Reconstructed DAG (Sampled)': wandb.Image(fig)})
    #%%
    """Machine Learning Efficacy"""
    import statsmodels.api as sm
    
    # Baseline
    covariates = [x for x in train.columns if x != 'CCAvg']
    linreg = sm.OLS(train['CCAvg'], train[covariates]).fit()
    linreg.summary()
    pred = linreg.predict(test[covariates])
    rsq_baseline = 1 - (test['CCAvg'] - pred).pow(2).sum() / np.var(test['CCAvg']) / len(test)
    print("Baseline R-squared: {:.2f}".format(rsq_baseline))
    wandb.log({'R^2 (Baseline)': rsq_baseline})
    #%%
    # Train
    covariates = [x for x in sample_df.columns if x != 'CCAvg']
    linreg = sm.OLS(sample_df['CCAvg'], sample_df[covariates]).fit()
    linreg.summary()
    pred = linreg.predict(test[covariates])
    rsq = 1 - (test['CCAvg'] - pred).pow(2).sum() / np.var(test['CCAvg']) / len(test)
    print("TVAE R-squared: {:.2f}".format(rsq))
    wandb.log({'R^2 (Sample)': rsq})
    #%%
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%