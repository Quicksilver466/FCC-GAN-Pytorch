import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim 
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2)
		#self.bn1=nn.BatchNorm2d(32)
		self.act1=nn.LeakyReLU(negative_slope=0.2)
		self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2)
		#self.bn2=nn.BatchNorm2d(64)
		self.act2=nn.LeakyReLU(negative_slope=0.2)
		self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
		#self.bn3=nn.BatchNorm2d(128)
		self.act3=nn.LeakyReLU(negative_slope=0.2)
		self.fc1=nn.Linear(1152,512)
		self.act4=nn.LeakyReLU(negative_slope=0.2)
		self.fc2=nn.Linear(512,64)
		self.act5=nn.LeakyReLU(negative_slope=0.2)
		self.fc3=nn.Linear(64,16)
		self.act6=nn.LeakyReLU(negative_slope=0.2)
		self.fc4=nn.Linear(16,1)
		self.act7=nn.Sigmoid()

	def forward(self,input_tensor):
		output=self.conv1(input_tensor)
		#output=self.bn1(output)
		output=self.act1(output)
		output=self.conv2(output)
		#output=self.bn2(output)
		output=self.act2(output)
		output=self.conv3(output)
		#output=self.bn3(output)
		output=self.act3(output)
		output=output.view(-1,128*3*3)
		output=self.fc1(output)
		output=self.act4(output)
		output=self.fc2(output)
		output=self.act5(output)
		output=self.fc3(output)
		output=self.act6(output)
		output=self.fc4(output)
		output=self.act7(output)
		return output
