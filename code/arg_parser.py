import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn


'''
---------------------------
Definition of the inputs
---------------------------


    dataroot - the path to the root of the dataset folder. We will talk more about the dataset in the next section
    workers - the number of worker threads for loading the data with the DataLoader
    batch_size - the batch size used in training. The DCGAN paper uses a batch size of 128
    ImageSize - the spatial size of the images used for training. This implementation defaults to 64x64. If another size is desired, the structures of D and G must be changed. See here for more details
    nc - number of color channels in the input images. For color images this is 3
    nz - length of latent vector
    ngf - relates to the depth of feature maps carried through the generator
    ndf - sets the depth of feature maps propagated through the discriminator
    num_epochs - number of training epochs to run. Training for longer will probably lead to better results but will also take much longer
    lr - learning rate for training. As described in the DCGAN paper, this number should be 0.0002
    beta1 - beta1 hyperparameter for Adam optimizers. As described in paper, this number should be 0.5
    ngpu - number of GPUs available. If this is 0, code will run in CPU mode. If this number is greater than 0 it will run on that number of GPUs

'''

# add parser to specify parameters for the cgan
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='kitchen |imagenet | celebA')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu',  type=int ,default=0, help='gpu device')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--model', default='', help="path to model (to continue training)")
parser.add_argument('--svm', default='', help="train or validate with svm")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

# load the checkpoint if training should be continued
if opt.model != '':
    checkpoint = torch.load(opt.model)

# make output dirs and files
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Set random seem for reproducibility
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# give warning if cuda isnt used but available
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
