# FCC-GAN - Pytorch
<p>This is a very basic unofficial implementation of FCC-GAN in pytorch, only on the MNIST dataset.
The Generator and Discriminator models are classes in generator_fcc.py and discriminator_fcc.py. Other than that the pretrained models itself and the state_dict are
stored in Saved_Models folder.</p>
<p>As for implementation details there are few things which I have changed. Those changes and details include: 
<ol>
<li> I have trained the network for 15 epochs. This models learns about everything it has to in about 10 epochs.</li>
<li> In the Generator network the dimensions weren't matching up. That's why I had to change it, still I tried to keep the number of layers exactly the same. Overall
in Generator I only changed the last layer, where the kernel size was made 4 instead of 3 and a padding of 2 was added.</li>
<li> Same issue occurs in Discriminator. So just like before I tried to keep the number of layers exactly the same. The only change which has been made is that
 I added padding of 1 in the last convolutional layer.</li>
 <li> Now the most important and major change is removing batchnorm from Discriminator. Now when batchnorm was added the output of Generator was pretty much garbage
 and even after 10 epochs the Generator hadn't learned anything. From the loss of both the networks I concluded that adding the batchnorm layer in Discriminator 
 helps it converge a lot faster and thus its outputs from sigmoid are much closer to the edges(very close to 0 or 1) where the gradient is almost 0 essentially 
 indicating that the generator won't be learning much and as the training goes on the superior position of Discriminator solidifies even more. Thus removing the 
 batchnorm from Discriminator balances the networks and lets Generator learn meaningful information.</li>
 <li> Another major change is done in training process where instead of going with the classical loss function I have used the slightly tweaked one. You can find 
 more details about it in <a href="https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html">dcgan tutorial of pytorch</a>.</li>
 <li> I don't remember exactly if I used the weight initialization which was also taken from the <a href="https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html">dcgan tutorial of pytorch</a>.
Still that weight initialization doesn't have much effect but still if I had to say then the output with weight initialization seems slightely better but that 
variation also occurs between two different training runs.</li>
<li> Regarding other hyperparameters, I have kept the learning rate at 0.0001, weight decay at 0.000001, Leaky-ReLU negative slope at 0.2 and batch size of 32. The 
weight decay originally was 0.00001.</li>
<br>
<p>This implementation isn't perfect and can be improved in many aspects so suggestions are appreciated</p>
<a href="https://arxiv.org/abs/1905.02417">FCC-GAN: A Fully Connected and Convolutional Net Architecture for GANs by Sukarna Barua</a> 
