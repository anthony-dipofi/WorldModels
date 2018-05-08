import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class DistributionalMLP(nn.Module):
	def __init__(self, input_size,output_size, hidden_units=[256,256]):
		super(DistributionalMLP,self).__init__()

		sizes = [input_size]+hidden_units+[output_size]
		#self.linear torch.linear
		self.layers = torch.nn.ModuleList()
		for i in range(len(sizes)-1):
			self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))

		for i in range(1,len(self.layers)):
			self.layers[i].weight.data.normal_(0.0,0.02)

	def forward(self, v):
		for i in range(len(self.layers)-1):
			#print("v",i,v)
			v = F.tanh(self.layers[i](v))
		#v = F.tanh(self.layer1(v))
		#v = F.tanh(self.layer2(v))
		#v = v.view([-1,self.linear_in])
		#v = F.tanh(self.linear(v))
		v = self.layers[-1](v)
		return v
