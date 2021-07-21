import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim 
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1=nn.Linear(100,64,bias=True)
		self.act1=nn.ReLU()
		self.fc2=nn.Linear(64,512,bias=True)
		self.act2=nn.ReLU()
		self.fc3=nn.Linear(512,1152,bias=True)
		self.bn1=nn.BatchNorm1d(num_features=1152)
		self.convt1=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2)
		self.bn2=nn.BatchNorm2d(64)
		self.act3=nn.ReLU()
		self.convt2=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2)
		self.bn3=nn.BatchNorm2d(32)
		self.act4=nn.ReLU()
		self.convt3=nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=4,stride=2,padding=2)
		self.act5=nn.Tanh()

	def forward(self,input_tensor):
		output=self.fc1(input_tensor)
		output=self.act1(output)
		output=self.fc2(output)
		output=self.act2(output)
		output=self.fc3(output)
		output=self.bn1(output)
		output=output.view(-1,128,3,3)
		output=self.convt1(output)
		output=self.bn2(output)
		output=self.act3(output)
		output=self.convt2(output)
		output=self.bn3(output)
		output=self.act4(output)
		output=self.convt3(output)
		output=self.act5(output)
		return output
