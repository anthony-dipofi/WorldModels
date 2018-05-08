import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class ImageEncoderResidualBlock(nn.Module):
	def __init__(self, depth, bottleneck_depth, stride):
		super(ImageEncoderResidualBlock,self).__init__()
		self.stride = stride

		self.conv1 = nn.Conv2d( depth,  depth, (1, 1))
		self.conv2 = nn.Conv2d( depth,  bottleneck_depth, (3, 3),padding=(1,1),stride=stride)
		self.conv3 = nn.Conv2d( bottleneck_depth,  depth, (1, 1))

		if(self.stride>1):
			self.conv_res = nn.Conv2d(depth, depth, (1,1),stride = (stride,stride))

		for i in range(1,len(list(self.modules()))):
			list(self.modules())[i].weight.data.normal_(0.0,0.01)

	def forward(self, ins):
		v = ins
		#print("v1",v.shape)

		v = F.relu(self.conv1(v))
		#print("v2",v.shape)
		v = F.relu(self.conv2(v))
		#print("v3",v.shape)
		v = F.relu(self.conv3(v))
		#print("v4",v.shape)

		#print("stride",self.stride)
		res = ins if self.stride <= 1 else self.conv_res(ins)
		#print("res",res.shape)
		v = v + res

		return v

class ImageEncoder(nn.Module):
	def __init__(self, num_resblocks=6, res_depth=64, bottleneck_depth=32, output_dim=500, strides=(2,1,2,1,2,1), input_shape=(320,224),res_input_shape = (32,32),linear_in=1024):
		super(ImageEncoder,self).__init__() #sonic image size (320 x 224 x 3)

		self.linear_in = linear_in
		self.num_resblocks = num_resblocks
		rescale = (input_shape[0]//res_input_shape[0], input_shape[1]//res_input_shape[1])
		print("rescale",rescale)
		self.conv_initial = torch.nn.Conv2d(3, res_depth, kernel_size=rescale, stride=rescale)
		self.resblocks = torch.nn.ModuleList()
		for i in range(num_resblocks):
			self.resblocks.append(ImageEncoderResidualBlock(res_depth,bottleneck_depth,strides[i]))

		self.linear = torch.nn.Linear(linear_in, output_dim)

	def forward(self, v):
		#print("v-1",v.shape)
		v = F.relu(self.conv_initial(v))
		#print("v0",v.shape)

		for i in range(self.num_resblocks):
			v = self.resblocks[i](v)

		v = v.view([-1,self.linear_in])
		v = F.tanh(self.linear(v))
		return v


class ImageDecoderResidualBlock(nn.Module):
	def __init__(self, depth, bottleneck_depth, stride):
		super(ImageDecoderResidualBlock,self).__init__()
		#only works with strides of 1 or 2

		self.stride = stride

		self.conv1 = nn.ConvTranspose2d(depth, bottleneck_depth, (1, 1))
		if(stride>1):
			self.conv2 = nn.ConvTranspose2d(bottleneck_depth, depth, (3, 3),padding=(0,0),stride=2)
		else:
			self.conv2 = nn.ConvTranspose2d(bottleneck_depth, depth, (3, 3),padding=(1,1),stride=1)
		self.conv3 = nn.ConvTranspose2d( depth,  depth, (1, 1))

		if(self.stride>1):
			self.conv_res = nn.ConvTranspose2d(depth, depth, (stride,stride),stride = stride)

		for i in range(1,len(list(self.modules()))):
			list(self.modules())[i].weight.data.normal_(0.0,0.01)

	def forward(self, ins):
		v = ins

		#print("v1",v.shape)
		v = F.relu(self.conv1(v))
		#print("v2",v.shape)
		if(self.stride > 1):
			v = F.relu(self.conv2(v))[:,:,:-1,:-1]
		else:
			v = F.relu(self.conv2(v))
		#print("v3",v.shape)
		v = F.relu(self.conv3(v))
		#print("v4",v.shape)

		res = ins if self.stride <= 1 else self.conv_res(ins)
		#print("res",res.shape)

		v = v + res
		return v

class ImageDecoder(nn.Module):
	def __init__(self, num_resblocks=6, res_depth=64, bottleneck_depth=32, input_dim=500, strides=(2,1,2,1,2,1),output_shape=(320,224),res_output_shape = (32,32),linear_out=1024):
		super(ImageDecoder,self).__init__() #sonic image size (320 x 224 x 3)

		self.num_resblocks = num_resblocks
		self.linear_out = linear_out
		self.input_dim = input_dim
		self.linear = torch.nn.Linear(input_dim,linear_out)
		self.res_depth = res_depth

		self.resblocks = torch.nn.ModuleList()
		for i in range(num_resblocks):
			self.resblocks.append(ImageDecoderResidualBlock(res_depth,bottleneck_depth,strides[i]))

		rescale = (output_shape[0]//res_output_shape[0],output_shape[1]//res_output_shape[1])
		self.conv_initial = torch.nn.ConvTranspose2d(res_depth, 3, kernel_size=rescale, stride=rescale)

	def forward(self, v):

		#print("v",v.shape)
		v = v.view([-1,self.input_dim])

		#print("v",v.shape)
		v = self.linear(v)

		#print("v",v.shape)
		v = v.view([-1,self.res_depth,8,8])
		#print("v",v.shape)
		for i in range(self.num_resblocks):
			v = self.resblocks[i](v)

		#v = F.relu(self.conv_initial(v))
		#v = torch.clamp(v,0,255)
		v = 255*F.sigmoid(self.conv_initial(v))
		#v=255*F.sigmoid(v)
		return v

class ImageEncoderSimplified1(nn.Module):
	def __init__(self):
		super(ImageEncoderSimplified1,self).__init__() #sonic image size (320 x 224 x 3)

		self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=(10,7), stride=(10,7))
		self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1,padding=1)
		self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2,padding=1)
		self.linear = torch.nn.Linear(1024,128)

		for i in range(1,len(list(self.modules()))):
			list(self.modules())[i].weight.data.normal_(0.0,0.2)

	def forward(self, v):
		#print("v0",v.shape)
		v = F.relu(self.conv1(v))
		#print("v1",v.shape)
		v = F.relu(self.conv2(v))+v
		#print("v2",v.shape)
		v = F.relu(self.conv3(v))
		#print("v3",v.shape)

		#v = v.view([-1])
		v = F.tanh(self.linear(v))
		return v

class ImageDecoderSimplified1(nn.Module):
	def __init__(self):
		super(ImageDecoderSimplified1,self).__init__() #sonic image size (320 x 224 x 3)

		self.linear = torch.nn.Linear(128,512)

		self.conv1 = torch.nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)#,padding=1)
		self.conv2 = torch.nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1,padding=1)
		self.conv3 = torch.nn.ConvTranspose2d(8, 3, kernel_size=(10,7), stride=(10,7))

		for i in range(1,len(list(self.modules()))):
			list(self.modules())[i].weight.data.normal_(0.0,0.2)

	def forward(self, v):
		#print("v-1",v.shape)


		v = F.tanh(self.linear(v))
		v = v.view([-1,8,8,8])
		#print("v0",v.shape)

		v = F.relu(self.conv1(v))[:,:,:-1,:-1]
		#print("v1",v.shape)
		v = F.relu(self.conv2(v))+v
		#print("v2",v.shape)
		v = F.relu(self.conv3(v))
		#print("v3",v.shape)
		return v
