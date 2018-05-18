import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class WM_LSTM(nn.Module): #NOTE: the MBP paper and the DNC paper it is based on use a variant of deep lstms where both the initial input and the hidden states of intermediate layer are used in the input to later layers
	def __init__(self,input_size, hidden_units):
		super(WM_LSTM,self).__init__()
		self.lstm1 = torch.nn.LSTM(input_size = input_size, num_layers = 1, hidden_size = hidden_units)
		#self.lstm2 = torch.nn.LSTM(input_size = input_size+hidden_units, num_layers = 1, hidden_size = hidden_units)
		self.lstm2 = torch.nn.LSTM(input_size = hidden_units, num_layers = 1, hidden_size = hidden_units)
		for p in self.lstm1.parameters():
			torch.nn.init.normal(p,0,0.1)
		for p in self.lstm2.parameters():
			torch.nn.init.normal(p,0,0.1)

	def forward(self,v):
		o1,h1 = self.lstm1(v)
		#v = Variable(torch.cat([v,o1])).cuda()
		o2,h2 = self.lstm2(o1)
		out = torch.cat([o1,o2],dim=2)
		hidden = torch.cat([h1[0],h2[0]],dim=1)
		return out, hidden#F.log_softmax(out)


class WM_TemporalCONV(nn.Module):
	def __init__(self,input_size, input_len=32):
		super(WM_TemporalCONV,self).__init__()
		self.input_size = input_size
		self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(5,1), stride=[2,1])
		self.conv2 = torch.nn.Conv2d(3, 5, kernel_size=(5,1), stride=[2,1])
		self.conv3 = torch.nn.Conv2d(5, 5, kernel_size=(5,1))
		self.lin = torch.nn.Linear(5*input_size, 4*input_size)
		for i in range(1,len(list(self.modules()))):
			list(self.modules())[i].weight.data.normal_(0.0,0.2)

	def forward(self,v):
		v = v.permute([1,0,2])
		v = v.unsqueeze(1)
		v = F.relu(self.conv1(v))
		v = F.relu(self.conv2(v))
		v = F.relu(self.conv3(v))

		v = v.view([-1,5*self.input_size])
		v = 2*F.tanh(self.lin(v))
		return v.unsqueeze(0)


class WM_MDN(nn.Module):
	def __init__(self, input_size, num_dists, z_dim, hidden_size):
		super(WM_MDN,self).__init__()
		self.num_dists = num_dists
		self.z_dim = z_dim

		self.hidden = torch.nn.Linear(input_size, hidden_size)
		self.z_pi = torch.nn.Linear(hidden_size,num_dists)
		self.z_sigma = torch.nn.Linear(hidden_size, num_dists*z_dim)
		self.z_mu = torch.nn.Linear(hidden_size, num_dists*z_dim)

	def forward(self,hidden_states):
		batch_size = len(hidden_states[0])
		mix = F.tanh(self.hidden(hidden_states))
		pis = F.softmax(self.z_pi(mix),dim=2)
		mus = self.z_mu(mix)
		#sigmas = torch.exp(torch.log(torch.clamp(self.z_sigma(mix),0.000001)))
		sigmas = torch.abs(self.z_sigma(mix))+0.000001

		return pis, sigmas, mus
