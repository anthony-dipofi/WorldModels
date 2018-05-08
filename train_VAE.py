import retro
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional
import numpy as np
import visdom
import sys
import os

import random

import MBP_EncoderDecoder as MBP_ED
import MBP_MLP

#import pyro
#import pyro.distributions as dist

import pickle


def main():
	#env = retro.make(game='Airstriker-Genesis', state='Level1')
	#env = retro.make(game='SonicTheHedgehog-Genesis')#, state='GreenHillZone.Act1')
	#files = os.listdir("data")
	training_steps = 200000
	batch_size = 16 #16
	test_batch_size = 64
	learn_rate = 0.00005 #0.00003
	save_file_name = "test6"
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

	all_obs = np.concatenate(obss)
	all_rewards = np.concatenate(rewards)
	all_actions = np.concatenate(actions)
	all_data_length = len(all_rewards)

	shuffle = np.random.permutation(all_data_length)

	all_obss = all_obs[shuffle]
	all_rewards = all_rewards[shuffle]
	all_actions = all_actions[shuffle]

	all_obss = torch.Tensor(all_obss)#.cuda()
	all_rewards = torch.Tensor(all_rewards)#.cuda()
	all_actions = torch.Tensor(all_actions)#.cuda()
	all_data_length = len(all_rewards)

	test_obss = all_obss[int(0.9*all_data_length):].cuda()
	test_rewards = all_rewards[int(0.9*all_data_length):].cuda()
	test_actions = all_rewards[int(0.9*all_data_length):].cuda()
	test_data_length = len(test_rewards)

	obss = all_obss[:int(0.9*all_data_length)].cuda()
	rewards = all_rewards[:int(0.9*all_data_length)].cuda()
	actions = all_rewards[:int(0.9*all_data_length)].cuda()
	data_length = len(test_rewards)

	print(all_obss.shape)

	v_size = 256

	img_encoder = MBP_ED.ImageEncoder(num_resblocks=6, res_depth=64, bottleneck_depth=32, output_dim=v_size*2, strides=(2,1,2,1,2,1), input_shape=(64,64),res_input_shape = (64,64),linear_in=4096)
	img_decoder = MBP_ED.ImageDecoder(num_resblocks=6, res_depth=64, bottleneck_depth=32, input_dim=v_size*2, strides=(1,2,1,2,1,2), output_shape=(64,64),res_output_shape = (64,64),linear_out=4096)

	img_encoder = img_encoder.cuda()
	img_decoder = img_decoder.cuda()

	prior = MBP_MLP.DistributionalMLP(512,512,[512]).cuda()
	v_out = MBP_MLP.DistributionalMLP(256,512,[512]).cuda()

	params = list(img_encoder.parameters())+ \
			 list(img_decoder.parameters())+ \
			 list(prior.parameters())+ \
			 list(v_out.parameters())


	optimizer = optim.Adam(params, lr=learn_rate, weight_decay=3*10**-6)

	vis = visdom.Visdom()
	graph_step = 250

	batch_losses = torch.ones(training_steps)
	test_losses = []
	batch_loss_chart = vis.line(torch.Tensor([0]))
	test_chart = vis.line(torch.Tensor([0]))

	for i in range(training_steps):

		optimizer.zero_grad()

		ind = torch.LongTensor(np.random.choice((data_length-1),batch_size,replace=False)).cuda()
		img_batch = obss[ind]
		rew_batch = rewards[ind]
		act_batch = actions[ind]

		'''batch_start = 10#random.randint(0, data_length - batch_size - 2)
		img_batch = obss[batch_start:batch_start+batch_size]
		rew_batch = rewards[batch_start:batch_start+batch_size]
		act_batch = actions[batch_start:batch_start+batch_size]'''

		obs_ten = img_batch
		obs_ten = obs_ten.permute(0,3,1,2)
		img_in = Variable(obs_ten,requires_grad=True)
		img_e = img_encoder(img_in)
		e_dist = prior(img_e)
		mu = e_dist[:,v_size:]
		#most VAEs I have seen use exp here to ensure sigma is > 0  but doing so seems to cause the model to diverge and abs seems to works well ¯\_(ツ)_/¯
		sigma = torch.abs(e_dist[:,:v_size])+0.0001#torch.exp(e_dist[:,:v_size])#torch.abs(e_dist[:,:v_size])+0.0001
		v = mu + sigma*Variable(torch.randn(mu.shape)).cuda()
		vo = v_out(v)
		img_de = img_decoder(vo)
		kl_loss = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

		dist_loss = torch.sum((img_in.detach()/255-img_de/255)**2)
		loss = dist_loss +kl_loss
		#print ("loss",loss)
		loss.backward()#retain_graph=True)
		optimizer.step()

		batch_losses[i] = loss.data[0]
		if(i%graph_step == 0):
			print("mu",mu[-1])
			print("sigma",sigma[-1])
			print("v",v[-1])
			print("in",img_in[-1])
			print("out",img_de[-1])
			print(i,loss.data)

			ind = torch.LongTensor(np.random.choice((test_data_length-1),test_batch_size,replace=False)).cuda()
			img_batch = test_obss[ind]
			rew_batch = test_rewards[ind]
			act_batch = test_actions[ind]

			img_batch = img_batch.permute(0,3,1,2)
			img_in_test = Variable(img_batch,requires_grad=False)
			img_e = img_encoder(img_in_test)
			e_dist = prior(img_e)
			mu = e_dist[:,v_size:]
			sigma = torch.abs(e_dist[:,:v_size])+0.0001
			v = mu + sigma*Variable(torch.randn(mu.shape)).cuda()
			vo = v_out(v)
			img_de_test = img_decoder(vo)
			kl_loss = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

			dist_loss = torch.sum((img_in_test.detach()/255-img_de_test/255)**2)
			loss = dist_loss +kl_loss
			test_losses.append(loss.data[0] * (batch_size/test_batch_size))

			if(i>0):
				vis.updateTrace(X=torch.arange(i-graph_step,i),Y=batch_losses[i-graph_step:i], append = True, win=batch_loss_chart, name="Batch Losses")
				vis.updateTrace(X=torch.arange(i,i+graph_step,step=graph_step),Y=torch.Tensor([test_losses[-1]]), append = True, win=test_chart, name="test Losses")

			if(i%2500==0 and i>0):
				vis.images([img_de.data[0],img_in.data[0]],opts=dict(caption='train ep.'+str(i)))
				vis.images([img_de_test.data[0],img_in_test.data[0]],opts=dict(caption='test ep. '+str(i)))


	torch.save(img_encoder,save_file_name+"_img_encoder.pth")
	torch.save(img_decoder,save_file_name+"_img_decoder.pth")
	torch.save(prior,save_file_name+"_prior_mlp.pth")
	torch.save(v_out,save_file_name+"_v_out_mlp.pth")


if __name__ == '__main__':
	main()
