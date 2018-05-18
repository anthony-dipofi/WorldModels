import retro
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import visdom
import sys
import os

import random

import MBP_EncoderDecoder as MBP_ED
import MBP_MLP

import WM_VAE

import WM_MDN_RNN as WM_RNN


import pickle


def main():
	save_file_name = "gen_MDNRNN_Data_test4"
	training_data_folder = "data/GreenHillZone/"

	files = os.listdir(training_data_folder)
	obss = []
	rewards = []
	actions = []
	for f in files:
		infile = open(training_data_folder+f,'rb')
		data=pickle.load(infile)
		obss.append(data[0])
		rewards.append(data[1])
		actions.append(data[2])


	obss = torch.Tensor(np.concatenate(obss)).cuda()
	rewards = torch.Tensor(np.concatenate(rewards)).cuda()
	actions = torch.Tensor(np.concatenate(actions)).cuda()
	data_length = len(rewards)
	print(obss.shape)

	batch_size = 32
	z_size = 256

	img_encoder = torch.load("test3_img_encoder.pth")

	prior = torch.load("test3_prior_mlp.pth")

	encodings = []

	for i in range(0,data_length-(data_length%batch_size)-batch_size,batch_size):

		batch_start = i
		img_batch = obss[batch_start:batch_start+batch_size]
		rew_batch = rewards[batch_start:batch_start+batch_size]
		act_batch = actions[batch_start:batch_start+batch_size]

		#print(img_batch.shape)
		obs_ten = img_batch
		obs_ten = obs_ten.permute(0,3,1,2)

		img_in = Variable(obs_ten,requires_grad=False)
		img_e = img_encoder(img_in)

		e_dist = prior(img_e)
		e_dist[:,:z_size] = torch.abs(e_dist[:,:z_size])+0.0001
		encodings.append(e_dist.data)

		print(i,"/",data_length-(data_length%batch_size)-batch_size)

	encodings = torch.stack(encodings).view([-1,z_size*2])

	torch.save(encodings,save_file_name+"_VAE_encodings.pth")


if __name__ == '__main__':
	main()
