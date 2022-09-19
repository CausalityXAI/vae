#%%
#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; 
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
#%%
import torch
from codebase import utils as ut
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import numpy as np
import math
import time
from torch.utils import data
from utils import get_batch_unin_dataset_withlabel, _h_A
import matplotlib.pyplot as plt
import random
import torch.utils.data as Data
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from codebase import utils as ut
from codebase.models.mask_vae_pendulum import CausalVAE
import argparse
from pprint import pprint
cuda = torch.cuda.is_available()
from torchvision.utils import save_image
import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epoch_max',   type=int, default=101,    help="Number of training epochs")
parser.add_argument('--iter_save',   type=int, default=5, help="Save model every n epochs")
parser.add_argument('--run',         type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',       type=int, default=1,     help="Flag for training")
parser.add_argument('--color',       type=int, default=False,     help="Flag for color")
parser.add_argument('--toy',       type=str, default="pendulum_mask",     help="Flag for toy")
args = parser.parse_args(args=[])
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
def _sigmoid(x):
	I = torch.eye(x.size()[0]).to(device)
	x = torch.inverse(I + torch.exp(-x))
	return x
#%% 
class DeterministicWarmup(object):
	"""
	Linear deterministic warm-up as described in
	[S?nderby 2016].
	"""
	def __init__(self, n=100, t_max=1):
		self.t = 0
		self.t_max = t_max
		self.inc = 1/n

	def __iter__(self):
		return self

	def __next__(self):
		t = self.t + self.inc

		self.t = self.t_max if t > self.t_max else t
		return self.t
layout = [
	('model={:s}',  'causalvae'),
	('run={:04d}', args.run),
	('color=True', args.color),
	('toy={:s}', str(args.toy))
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
lvae = CausalVAE(name=model_name, z_dim=16, device=device).to(device)
if not os.path.exists('./figs_vae/'): 
	os.makedirs('./figs_vae/')
#%%
dataset_dir = './causal_data/causal_data/pendulum'
train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 64)
test_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 1)
optimizer = torch.optim.Adam(lvae.parameters(), lr=1e-3, betas=(0.9, 0.999))
beta = DeterministicWarmup(n=100, t_max=1) # Linear warm-up from 0 to 1 over 50 epoch
#%%
def save_model_by_name(model, global_step):
	save_dir = os.path.join('checkpoints', model.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(global_step))
	state = model.state_dict()
	torch.save(state, file_path)
	print('Saved to {}'.format(file_path))
#%%
for epoch in range(args.epoch_max):
	lvae.train()
	total_loss = 0
	total_rec = 0
	total_kl = 0
	for u, l in tqdm.tqdm(train_dataset):
		# break
		# u.shape
		# l.shape
		optimizer.zero_grad()
		#u = torch.bernoulli(u.to(device).reshape(u.size(0), -1))
		u = u.cuda()
		"""FIXME"""
		u = transforms.Resize([96, 96])(u)
		l = l.cuda()
		# u = u.to(device)
		L, kl, rec, reconstructed_image,_ = lvae.negative_elbo_bound(u,l,sample = False)
  
		###############################################################################
		# q_m, q_v = lvae.enc.encode(u.to(device))

        # q_m, q_v = q_m.reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim]),torch.ones(q_m.size()[0], lvae.z1_dim,lvae.z2_dim).to(device)

        # decode_m, decode_v = lvae.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], lvae.z1_dim,lvae.z2_dim).to(device))
        # decode_m, decode_v = decode_m.reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim]),decode_v
		# m_zm, m_zv = lvae.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim]),decode_v.reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim])
		# m_u = lvae.dag.mask_u(l.to(device))
          
		# f_z = lvae.mask_z.mix(m_zm).reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim]).to(device)
		# e_tilde = lvae.attn.attention(decode_m.reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim]).to(device),q_m.reshape([q_m.size()[0], lvae.z1_dim,lvae.z2_dim]).to(device))[0]
              
		# f_z1 = f_z+e_tilde
		# g_u = lvae.mask_u.mix(m_u).to(device)
		# z_given_dag = ut.conditional_sample_gaussian(f_z1, m_zv*0.001)
        
        # decoded_bernoulli_logits,x1,x2,x3,x4 = lvae.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], lvae.z_dim]), l.to(device))
        
        # rec = ut.log_bernoulli_with_logits(u, decoded_bernoulli_logits.reshape(u.size()))
        # rec = -torch.mean(rec)

        # p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        # cp_m, cp_v = ut.condition_prior(lvae.scale, l, lvae.z2_dim)
        # # cp_v = torch.ones([q_m.size()[0],lvae.z1_dim,lvae.z2_dim]).to(device)
        # cp_v = torch.ones([cp_m.size()[0],lvae.z1_dim,lvae.z2_dim]).to(device)
        
        # cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device)) 
        # # sample = torch.randn(cp_m.size()).to(device)
		# # cp_m + (cp_v**0.5)*sample
        
        # kl = torch.zeros(1).to(device)
        # kl = ut.kl_normal(q_m.view(-1,lvae.z_dim).to(device), q_v.view(-1,lvae.z_dim).to(device), p_m.view(-1,lvae.z_dim).to(device), p_v.view(-1,lvae.z_dim).to(device))
        
        # i = 0
        # for i in range(lvae.z1_dim):
        #     kl = kl + beta*ut.kl_normal(decode_m[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        #     decode_m.shape
        #     cp_v.shape
        #     cp_m.shape
        #     cp_v.shape
        # kl = torch.mean(kl)
        # mask_kl = torch.zeros(1).to(device)
        # mask_kl2 = torch.zeros(1).to(device)

        # for i in range(4):
        #     mask_kl = mask_kl + 1*ut.kl_normal(f_z1[:,i,:].to(device), cp_v[:,i,:].to(device),cp_m[:,i,:].to(device), cp_v[:,i,:].to(device))
        
        
        # u_loss = torch.nn.MSELoss()
        # mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        # nelbo = rec + kl + mask_l
		###############################################################################
		
		dag_param = lvae.dag.A
		
		#dag_reg = dag_regularization(dag_param)
		h_a = _h_A(dag_param, dag_param.size()[0])
		L = L + 3*h_a + 0.5*h_a*h_a #- torch.norm(dag_param) 
   
   
		L.backward()
		optimizer.step()
		#optimizer.zero_grad()

		total_loss += L.item()
		total_kl += kl.item() 
		total_rec += rec.item() 

		m = len(train_dataset)
		save_image(u[0], 'figs_vae/reconstructed_image_true_{}.png'.format(epoch), normalize = True) 
		"""FIXME"""
		# save_image(reconstructed_image[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch), normalize = True) 
		save_image(reconstructed_image[0], 'figs_vae/reconstructed_image_{}.png'.format(epoch),  range = (0,1)) 
		
	# if epoch % 1 == 0:
	print(str(epoch)+' loss:'+str(total_loss/m)+' kl:'+str(total_kl/m)+' rec:'+str(total_rec/m)+'m:' + str(m))

	# if epoch % args.iter_save == 0:
	# 	ut.save_model_by_name(lvae, epoch)
ut.save_model_by_name(lvae, epoch)
#%%
B_est = lvae.dag.A
import pandas as pd
pd.DataFrame(B_est.cpu().detach().numpy()).to_csv(r'D:\trustworthyAI-master\research\CausalVAE\checkpoints\B_est.csv')
#%%