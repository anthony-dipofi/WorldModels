import retro
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import PIL
import torchvision.transforms.functional
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
	training_steps = 100000
	batch_size = 1 #Untested with batch_size > 1
	learn_rate = 0.00015
	save_file_name = "test4"
	training_data_file = "data/GreenHillZone/"

	infile = open("gen_MDNRNN_Data_test4_VAE_encodings.pth",'rb')
	#gen_MDNRNN_Data_test4_VAE_encodings.pth
	#test3_VAE_encodings.pth
	data = torch.load(infile).cuda()
	print(data)
	data_length = len(data)
	z_size = 256
	seq_len = 20
	num_dists = 3
	time_stride = 1 #used to predict further into the future, untested with time_stride > 1

	wm_rnn = WM_RNN.WM_LSTM(z_size, hidden_units=512).cuda()
	wm_mdn = WM_RNN.WM_MDN(1024, num_dists, z_size, 512).cuda()

	v_out = torch.load("test3_v_out_mlp.pth").cuda()
	img_decoder = torch.load("test3_img_decoder.pth").cuda()

	params = list(wm_rnn.parameters())+ \
			 list(wm_mdn.parameters())

	optimizer = optim.Adam(params, lr=learn_rate, weight_decay=3*10**-6)

	vis = visdom.Visdom()
	graph_step = 50

	batch_returns = torch.ones(training_steps)
	batch_returns_chart = vis.line(torch.Tensor([0]))

	for i in range(training_steps):

		optimizer.zero_grad()

		batch_start = random.randint(0, data_length - batch_size - seq_len - 2)
		batch = torch.zeros([seq_len,batch_size,z_size]).cuda()
		target_batch = torch.zeros([seq_len,batch_size,z_size]).cuda()
		data_mus = data[batch_start:batch_start+batch_size+seq_len+time_stride,z_size:]
		data_sigmas = data[batch_start:batch_start+batch_size+seq_len+time_stride,:z_size]
		for j in range(batch_size):
			v = data_mus + data_sigmas*Variable(torch.randn(data_mus.shape)).cuda()
			batch[:,j,:] = v[j:seq_len+j]
			target_batch[:,j,:] = v[j+time_stride:seq_len+time_stride]


		batch = Variable(batch,requires_grad=True)
		o,h = wm_rnn(batch)
		pis, sigmas, mus = wm_mdn(o)

		##########################################################################################
		#https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
		def gaussian_distribution(y, mu, sigma, pi):
			z_normals = torch.distributions.Normal(mu, sigma)
			z_prob = z_normals.log_prob(y).clamp(np.log(0.0001), -np.log(0.0001)).exp()
			r = (z_prob * torch.unsqueeze(pi,3))
			result = r.sum(2)
			return result

		def mdn_loss_fn(pi, sigma, mu, y):
			result = gaussian_distribution(y, mu, sigma, pi)
			result = -torch.log(result+0.0001)
			result = torch.mean(result)
			return result
		##########################################################################################

		mus = torch.squeeze(mus).view([seq_len,batch_size,num_dists,z_size])
		sigmas = torch.squeeze(sigmas).view([seq_len,batch_size,num_dists,z_size])
		ys = torch.stack([target_batch]*num_dists,dim=2)

		loss = mdn_loss_fn(pis,sigmas, mus,ys)

		loss.backward()
		#torch.nn.utils.clip_grad_value_(wm_rnn.parameters(),0.1)

		optimizer.step()

		batch_returns[i] = loss.data[0]

		if(i%graph_step==0 and i>0):
			  vis.updateTrace(X=torch.arange(i-graph_step,i),Y=batch_returns[i-graph_step:i], append = True, win=batch_returns_chart, name="Batch Returns")

		if(i%250 == 0):
			print("pis",pis,pis.shape)
			z_normals = torch.distributions.Normal(mus, sigmas.clamp(0.0001))
			z_prob = z_normals.sample().detach()
			print("z_prob",z_prob,z_prob.shape)
			result = (z_prob * torch.unsqueeze(pis,3)).sum(2)
			print("result",result)
			print("mus",mus,mus.shape)
			print("sigmas",sigmas,sigmas.shape)
			print("o",o,o.shape)
			print("result",result)
			print("target_batch",torch.squeeze(target_batch))
			print(i,loss.data)
		if(i%1000==0 and i>0):
			z_normals = torch.distributions.Normal(mus, sigmas.clamp(0.0001))
			z_prob = z_normals.sample()
			print("z_prob",z_prob,z_prob.shape)
			vt = (z_prob * torch.unsqueeze(pis,3)).sum(2)
			print("vt",vt,vt.shape)
			print("target_batch",target_batch,target_batch.shape)
			vt = torch.cat([vt[-1],target_batch[-1]])
			print(vt)
			vo = v_out(vt)
			img_de = img_decoder(vo)
			vis.images(img_de.data.cpu().numpy(),opts=dict(caption='train ep.'+str(i)))
			#vis.video(img_de.data.cpu().numpy())

	torch.save(wm_rnn,save_file_name+"_WM_RNN.pth")
	torch.save(wm_mdn,save_file_name+"_WM_MDN.pth")


if __name__ == '__main__':
	main()
