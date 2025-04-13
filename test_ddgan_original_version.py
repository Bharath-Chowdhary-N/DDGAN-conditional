# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np

import os

import cv2

import torchvision

import torch
torch.cuda.empty_cache()

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt

#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    print("coefficients: {}".format(coefficients))
    print("n_time: {}".format(n_time))
    print("T: {}".format(T))
    print("opt: {}".format(opt.nz))
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)#.to(x.device)
            x_0 = generator(x, t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()
        
    return x

def normalize_img(numpy_img):
    numpy_img = ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))
    return numpy_img

#%%
def sample_and_test(args):
    print(args)
    #torch.manual_seed(42)
    device = 'cuda:0'
    
    if args.dataset == 'cifar10':
        real_img_dir = 'pytorch_fid/cifar10_train_stat.npy'
    elif args.dataset == 'celeba_256':
        real_img_dir = 'pytorch_fid/celeba_256_stat.npy'
    elif args.dataset == 'lsun':
        real_img_dir = 'pytorch_fid/lsun_church_stat.npy'
    else:
        real_img_dir = args.real_img_dir
        print("////////////////////////////////////////////////",args.real_img_dir)
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    
    netG = NCSNpp(args).to(device)
    ckpt = torch.load('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format(args.dataset, args.exp, args.epoch_id), map_location=device)
    
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    torch.save(netG, "./gen_model.pth")
    
    
    T = get_time_schedule(args, device)
    
    pos_coeff = Posterior_Coefficients(args, device)
    
    print("pos:coeff : {}",pos_coeff)
        
    iters_needed = 640 //args.batch_size
    
    save_dir = "./generated_samples/{}".format(args.dataset)
    
    print("/////////////////////////////////",save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.compute_fid:
        for i in range(iters_needed):
            with torch.no_grad():
                x_t_1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
                x_t_1 = to_range_0_1(x_t_1)      
                fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args)
                
                import torchvision.transforms as T
                from PIL import Image
        
                transform = T.Resize((1*101,1*101))
        
                
     
        
                fake_sample = transform(fake_sample)
        
        
        
                torchvision.utils.save_image(fake_sample, './samples_Non_Lenses_fid.jpg')
                
                #fake_sample = to_range_0_1(fake_sample)
                for j, x in enumerate(fake_sample):
                    index = i * args.batch_size + j 
                    torchvision.utils.save_image(x, './generated_samples/{}/{}.jpg'.format(args.dataset, index))
                print('generating batch ', i)
        
        paths = [save_dir, real_img_dir]
        
        #############################################
        
          


        kwargs = {'batch_size': 100, 'device': device, 'dims': 2048}
        fid = calculate_fid_given_paths(paths=paths, **kwargs)
        print('FID = {}'.format(fid))
    else:
        print("///////////////////////////printing")
        x_t_1 = torch.randn(args.batch_size, args.num_channels,args.image_size, args.image_size).to(device)
        
        x_t_1 = torch.nn.functional.normalize(x_t_1,p=1.0)
        
        if True:
                
                size_1 = args.image_size
                
                print(np.shape(x_t_1))
                print(x_t_1.type())
                #x_t_1 = cv2.imread("../KiDS_DCGAN/KiDS_l_nl/KiDS_lens_non_lens/73.jpg")
                #x_t_1 = cv2.imread("../KiDS_DCGAN/KiDS_jpg_temp/37.jpg")
                #x_t_1 = cv2.imread("../KiDS_DCGAN/KiDS_sources/sources/16.jpg")
                
                #x_t_1 = cv2.imread("gen.jpg")
                x_t_1 = cv2.imread("256_t.png")
                #x_t_1 = cv2.imread("256_3.jpg")
                x_t_1 = cv2.cvtColor(x_t_1, cv2.COLOR_BGR2RGB) 
                
                
                
                #print(np.shape(x_t_1))        
                x_t_2 = cv2.resize(x_t_1, (size_1, size_1),
                       interpolation = cv2.INTER_LINEAR)
                
                gaussian = np.random.random((size_1, size_1, 3)).astype(np.float32)
                
                print("Min gaussian : {} , Max gaussian : {}".format(np.min(gaussian), np.max(gaussian)))
                
                gaussian = gaussian / 1
                
                plt.imshow(normalize_img(gaussian),cmap='gray')
                plt.colorbar()
                plt.savefig("gaussian.jpg")
                plt.close()
                    
                #x_t_2 = x_t_2 + gaussian       
                
                
                
                x_t_3 = np.zeros((args.batch_size,size_1,size_1,3))       
                
                plt.imshow(normalize_img(x_t_2)+gaussian,cmap='gray')
                plt.colorbar()
                plt.savefig("input_sample.jpg")
                plt.close()
                
                for ite in range(args.batch_size):
                    #x_t_3[ite] = x_t_2 
                    #x_t_3[ite] = 2*(normalize_img(x_t_2) + gaussian) - 1
                    x_t_3[ite] = normalize_img((normalize_img(x_t_2) + gaussian))
                    #x_t_3[ite] = (x_t_2) + (gaussian)                  
                       
                print(np.shape(x_t_3))  
                
                plt.imshow(x_t_3[1])
                plt.colorbar()
                plt.savefig("x_t_3_1.jpg")
                plt.close()
                
                plt.imshow(x_t_3[0])
                plt.colorbar()
                plt.savefig("x_t_3_0.jpg")
                plt.close()
                
                x_t_4 = torch.from_numpy(x_t_3)        
                
                print(np.shape(x_t_4))
                
                x_t_5 = x_t_4.permute(0,3,1,2)
                
                print(np.shape(x_t_5))
                
                x_t_1 = x_t_5.to(device)
                
                x_t_1 = x_t_1.float()
                print(x_t_1.type())
                
                #x_t_1 = (torch.nn.functional.normalize(x_t_1,p=1.0))
                #x_t_1 = 2*(torch.nn.functional.normalize(x_t_1,p=1.0)) - 1
        
        print("Max of array: {}".format(torch.max(x_t_1)))
        
        print("Min of array: {}".format(torch.min(x_t_1)))
        import torchvision.transforms as T
        from PIL import Image
        
        transform = T.Resize((1*101,1*101))
        
                
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args)
        #fake_sample = to_range_0_1(fake_sample)
        
        fake_sample = transform(fake_sample)
        
        #np.save("X_array",fake_sample.cpu().detach().numpy())
        
        

        #fake_sample = fake_sample.permute(0,2,3,1)
        in_num = np.random.randint(10000)
        torchvision.utils.save_image(fake_sample, './samples_Non_Lenses.jpg')
        #str_num = "npy_files_test/X_array_" + str(in_num)
        #np.save(str_num,fake_sample.cpu().detach().numpy())   
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy', help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
        



   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                
