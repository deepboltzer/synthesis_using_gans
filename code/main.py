'''
A dcgan on different datasets. 

input discriminator: 3xndfxndf
output discriminator: scalar prob. that the input is frim the real data distr.
discriminator strucutre: The discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations.

input generator: latent vaector z, drawn from standard gaussian.
output generator: 3xngfxngf RGB image
generator structure: The generator is comprised of convolutional-transpose layers, batch norm layers, and ReLU activations.

Why do we use strided conv-transpose? 
The strided conv-transpose layers allow the latent vector to be transformed into a volume with the same shape as an image.
'''
# load libraries
from __future__ import print_function
#%matplotlib inline
import arg_parser
import dcgan
import os
import random
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sklearn import svm
from torch.autograd import Variable

opt = arg_parser.opt

# Size of z latent vector (i.e. size of generator input)
nz = int(opt.nz)

# Size of feature maps in generator
ngf = int(opt.ngf)

# Size of feature maps in discriminator
ndf = int(opt.ndf)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = int(opt.ngpu)

'''
Load the data and transform it to tensor dataset. 
Create a pytorch dataloader and decide on which device we want to run the code.
To test the settings, plot some training images. 64 in SUM.
'''

# We can use an image folder dataset the way we have it setup.
# Create the dataset

if opt.dataset in ['imagenet','celebA']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'kitchen':
    dataset = dset.LSUN(root=opt.dataroot, classes=['kitchen_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3

# Create the dataloader
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# Decide which device we want to run on
device = torch.device("cuda:" + str(opt.gpu) if opt.cuda else "cpu")
print(device)
# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

'''
---------------------------
weight initialization
---------------------------

From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.2. 
The weights_init function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to meet this criteria. 
This function is applied to the models immediately after initialization.

'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

'''
Create a generator from the dcgan class. Load a checkpoint if it is available.
'''


# Create the generator
netG = dcgan.Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# load the model to continue training
if opt.model != '':
    netG.load_state_dict(arg_parser.checkpoint['netG'])
    #netG.eval()

# Print the model
print(netG)

# Create the Discriminator
netD = dcgan.Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# load the model to continue training

if opt.model != '':
    netD.load_state_dict(arg_parser.checkpoint['netD'])
    #netD.eval()

# Print the model
print(netD)

'''
---------------------------
loss function and optimizers
---------------------------
We will use the Binary Cross Entropy loss (BCELoss).
this function provides the calculation of both log components in the objective function (i.e. log(D(x)) and log(1−D(G(z)))). 
Next, we define our real label as 1 and the fake label as 0. These labels will be used when calculating the losses of D and G, and this is also the convention used in the original GAN paper. 
Finally, we set up two separate optimizers, one for D and one for G. As specified in the DCGAN paper, both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5. For keeping track of the generator’s learning progression,
we will generate a fixed batch of latent vectors that are drawn from a Gaussian distribution (i.e. fixed_noise) . In the training loop, we will periodically input this fixed_noise into G, and over the iterations we will see images form out of the noise.
'''

# Initialize BCELoss function
criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)

if opt.cuda:
    print("CUDA TRUE")
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
#fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.model != '':
    optimizerD.load_state_dict(arg_parser.checkpoint['optimizerD'])
    optimizerG.load_state_dict(arg_parser.checkpoint['optimizerG'])


'''
---------------------------
training the network
---------------------------

Part 1 - Train the Discriminator

Recall, the goal of training the discriminator is to maximize the probability of correctly classifying a given input as real or fake. In terms of Goodfellow, 
we wish to “update the discriminator by ascending its stochastic gradient”. Practically, we want to maximize log(D(x))+log(1−D(G(z))).
Due to the separate mini-batch suggestion from ganhacks, we will calculate this in two steps. First, we will construct a batch of real samples from the training set, 
forward pass through D, calculate the loss (log(D(x))), then calculate the gradients in a backward pass. Secondly, we will 
construct a batch of fake samples with the current generator, forward pass this batch through D, calculate the loss (log(1−D(G(z)))), 
and accumulate the gradients with a backward pass. Now, with the gradients accumulated from both the all-real and all-fake batches, we call a step of the Discriminator’s optimizer.

Part 2 - Train the Generator

As stated in the original paper, we want to train the Generator by minimizing log(1−D(G(z)))
in an effort to generate better fakes. As mentioned, this was shown by Goodfellow to not provide 
sufficient gradients, especially early in the learning process. As a fix, we instead wish to maximize log(D(G(z))). In the code we accomplish this by: 
classifying the Generator output from Part 1 with the Discriminator, computing G’s loss using real labels as GT, computing G’s gradients in a backward pass, 
and finally updating G’s parameters with an optimizer step. It may seem counter-intuitive to use the real labels as GT labels for the loss function,
but this allows us to use the log(x) part of the BCELoss (rather than the log(1−x) part) which is exactly what we want.
'''

# Training Loop

# Lists to keep track of progress
img_list = []

start_epoch = 0
if opt.model != '':
    start_epoch = arg_parser.checkpoint['epoch']+1
    iters = arg_parser.checkpoint['iters']
    G_losses = arg_parser.checkpoint['G_losses']
    D_losses = arg_parser.checkpoint['D_losses']
else :
    G_losses = []
    D_losses = []
    iters = 0
    
print("Starting Training Loop...")
# For each epoch
for epoch in range(start_epoch,opt.niter):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        # Forward pass real batch through D
        output = netD(inputv)
        # Calculate loss on all-real batch
        errD_real = criterion(output, labelv)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.data.mean()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        # Generate fake image batch with G
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        # Classify all fake batch with D
        output = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.data.mean()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == opt.niter-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                netG.eval()
                fake = netG(fixed_noise)
                vutils.save_image(real_cpu,
                    '%s/img/samples/real_samples.png' % opt.outf,
                    normalize=True)
                vutils.save_image(fake.data,
                    '%s/img/samples/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
    # do checkpointing
    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),'optimizerG': optimizerG.state_dict(), 'optimizerD': optimizerD.state_dict(), 'epoch': epoch, 'iters': iters , 'G_losses': G_losses, 'D_losses':D_losses}, '%s/models/model_%d.pth' % (opt.outf, epoch))
