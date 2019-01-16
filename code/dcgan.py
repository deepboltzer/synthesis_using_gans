import torch
import torch.nn as nn
import arg_parser

opt = arg_parser.opt

# Number of channels
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = int(opt.nz)

# Size of feature maps in generator
ngf = int(opt.ngf)

# Size of feature maps in discriminator
ndf = int(opt.ndf)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = int(opt.ngpu)


'''
---------------------------
define the generator 
---------------------------
As a series of strided two dimensional convolutional transpose layers, each paired with a 2d batch norm layer and a relu activation.
The output of the generator is fed through a tanh function to return it to the input data range of [−1,1]. It is worth noting the 
existence of the batch norm functions after the conv-transpose layers, as this is a critical contribution of the DCGAN paper. These layers help with the flow of gradients during training.
'''
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


'''
---------------------------
define the discriminator
---------------------------
As mentioned, the discriminator, D, is a binary classification network that takes an image as input and outputs a scalar probability 
that the input image is real (as opposed to fake). Here, D takes a 3xndfxndf input image, processes it through a series of Conv2d, BatchNorm2d, 
and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. This architecture can be extended with more layers 
if necessary for the problem, but there is significance to the use of the strided convolution, BatchNorm, and LeakyReLUs. The DCGAN paper mentions 
it is a good practice to use strided convolution rather than pooling to downsample because it lets the network learn its own pooling function. 
Also batch norm and leaky relu functions promote healthy gradient flow which is critical for the learning process of both G and D.
'''

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.conv2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.conv3 = nn.Sequential(
           # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.conv4 = nn.Sequential(
           # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ) 
        self.conv5 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ) 
    # forward propagation through the network
    def forward(self, input):
        out_conv_1 = self.conv1(input)
        out_conv_2 = self.conv2(out_conv_1)
        out_conv_3 = self.conv3(out_conv_2)
        out_conv_4 = self.conv4(out_conv_3)
        out_conv_5 = self.conv5(out_conv_4)
        return out_conv_5.view(-1, 1).squeeze(1)
   
    # get the features from the conv layers which can be used as input for the svm
    def get_features(self,input):
        out_conv_1 = self.conv1(input)
        out_conv_2 = self.conv2(out_conv_1)
        out_conv_3 = self.conv3(out_conv_2)
        out_conv_4 = self.conv4(out_conv_3)
        out_conv_5 = self.conv5(out_conv_4)

        max_pool_1 = nn.MaxPool2d(int(out_conv_1.size(2) / 4))
        max_pool_2 = nn.MaxPool2d(int(out_conv_2.size(2) / 4))
        max_pool_3 = nn.MaxPool2d(int(out_conv_3.size(2) / 4))
        max_pool_4 = nn.MaxPool2d(int(out_conv_4.size(2) / 4))

        feature_vec_1 = max_pool_1(out_conv_1).view(input.size(0), -1).view(-1, 1).squeeze(1)
        feature_vec_2 = max_pool_2(out_conv_2).view(input.size(0), -1).view(-1, 1).squeeze(1)
        feature_vec_3 = max_pool_3(out_conv_3).view(input.size(0), -1).view(-1, 1).squeeze(1)
        feature_vec_4 = max_pool_4(out_conv_5).view(input.size(0), -1).view(-1, 1).squeeze(1)
        #feature_vec_5 = out_conv_3.view(input.size(0), -1).view(-1, 1).squeeze(1)

        print(feature_vec_3)
        print(feature_vec_4)


        return torch.cat((feature_vec_1,feature_vec_2,feature_vec_3,feature_vec_4),1)
    

