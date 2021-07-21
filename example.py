import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch import optim 
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from generator_fcc import Generator
from discriminator_fcc import Discriminator

def gen_sample(path='./Saved_Models/gan_fcc_1_modelG_state_dict.pth'):
	gen=Generator()
	disc=Discriminator()

	gen.load_state_dict(torch.load(path))
	gen=gen.eval()
	inp=torch.randn(1,100)
	op=gen(inp)
	with torch.no_grad():
		output_img=np.array(op[0])
		output_img=output_img.reshape(28,28)
		plt.imshow(output_img,cmap='gray');
		plt.show()
		
gen_sample()
