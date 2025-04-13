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
import random
import cv2

import torchvision

import torch
torch.cuda.empty_cache()

from score_sde.models.ncsnpp_generator_adagn import NCSNpp
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision import datasets, transforms, utils
import torchvision.transforms as Transform
from PIL import Image
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

#def normalize_img(numpy_img):
#    numpy_img = ((numpy_img - np.min(numpy_img)) / (np.max(numpy_img) - np.min(numpy_img)))
#    return numpy_img

#def normalize_img(img):
import torch
import numpy as np

def normalize_img(img):
    if img is None:
        raise ValueError("Input image is None.")
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()  # Convert torch tensor to NumPy array

    if isinstance(img, np.ndarray):
        if img.ndim == 4:
            # Assuming the input is in the format (batch_size, channels, height, width)
            min_val = np.min(img)
            max_val = np.max(img)
            normalized_img = (img - min_val) / (max_val - min_val)
        elif img.ndim == 3:
            # Assuming the input is in the format (channels, height, width)
            min_val = np.min(img)
            max_val = np.max(img)
            normalized_img = (img - min_val) / (max_val - min_val)
        else:
            raise ValueError("Input array must have three or four dimensions.")
    else:
        raise ValueError("Input must be either a PyTorch tensor or a NumPy array.")
    
    return normalized_img

# ...


def pick_random_file(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a directory.")
        return None
    
    # Get a list of all files in the directory
    files = os.listdir(folder_path)
    
    # Filter out directories (if any)
    files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
    
    # Check if there are any files in the directory
    if not files:
        print("Error: No files found in the directory.")
        return None
    
    # Pick a random file from the list
    random_file = random.choice(files)
    
    return random_file


    
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
                
                import torchvision.transforms as Transform
                from PIL import Image
        
                transform = Transform.Resize((1*101,1*101))
        
                
     
        
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

        folder_path = "../target_folder_total/"
        for ite in range(1):
            print("-----------------ite : {} Done--------------------".format(ite))
            random_file = pick_random_file(folder_path)
            #random_file = "0.jpg"
            random_file_address = folder_path+random_file
            if random_file:
                print("Randomly picked file:", random_file_address)
            import torchvision.transforms as Transform
            from PIL import Image
            if True:
                        
                        #conditional_img = cv2.imread("gen3.jpg")
                        conditional_img = cv2.imread(random_file_address)
                        # ... (Rest of the existing code)

                        # Load the conditional image
                        #conditional_img = cv2.imread(conditional_image_path)
                        conditional_img = cv2.cvtColor(conditional_img, cv2.COLOR_BGR2RGB)
                        conditional_img = cv2.resize(conditional_img, (args.image_size, args.image_size),
                                                    interpolation=cv2.INTER_LINEAR)
                        print("------size of array----: {}".format(conditional_img.shape))
                        

                        # Normalize pixel values to [0, 1] range
                        #reshaped_array = conditional_img.reshape(len(conditional_img), -1, 3)

                        # Find minimum and maximum values along the second axis (101x101 pixels) for each image
                        #min_values = reshaped_array.min(axis=1, keepdims=True)
                        #max_values = reshaped_array.max(axis=1, keepdims=True)

                        # Normalize each image between 0 and 1
                        #normalized_array = (conditional_img - min_values) / (max_values - min_values)

                        # Reshape the normalized array back to its original shape
                        #normalized_image_array = normalized_array.reshape(len(conditional_img), 64, 64, 3)



                        #conditional_img  = torch.from_numpy(normalized_image_array).permute(2,0,1).unsqueeze(0).float().to(device)
                        conditional_img = torch.from_numpy(normalize_img(conditional_img)).permute(2,0,1).unsqueeze(0).float().to(device)

                        # Scale conditional image to match the expected input range
                        conditional_img = (conditional_img * 1) - 0
                        
                        print("Min noise conditional img :{} , Max noise conditional img: {}".format(torch.min(conditional_img), torch.max(conditional_img)))
                        conditional_img_batch = conditional_img.repeat(args.batch_size, 1, 1, 1)
                        # Use the conditional image to initialize the generation process
                        n_noise=False
                        if n_noise:
                                gaussian_noise = np.random.normal(0, 0.01, size=(args.batch_size, args.num_channels, args.image_size, args.image_size))
                                gaussian_noise = torch.from_numpy(gaussian_noise).float().to(device)
                                gaussian_noise = torch.clamp(gaussian_noise,-1,1)

                                # Add Gaussian noise to the conditional image
                                conditional_img_batch_with_noise = conditional_img_batch + gaussian_noise
                                
                                #print("------------torch_shape: {}".format(np.shape(conditional_img_batch_with_noise)))
                                
                                #conditional_img = torch.from_numpy(normalize_img(conditional_img_batch_with_noise)).permute(0,3,1,2).unsqueeze(0).float().to(device)
                                #conditional_img = torch.from_numpy(normalize_img(conditional_img_batch_with_noise.cpu().numpy())).permute(2,0,1).float().to(device)

                                # Scale conditional image to match the expected input range
                                conditional_img_batch_with_noise = normalize_img(conditional_img_batch_with_noise) #additional step
                                conditional_img_with_noise = (conditional_img_batch_with_noise * 1) - 0
                                #conditional_img_with_noise = conditional_img.repeat(args.batch_size, 1, 1, 1)
                                
                                #print("Min noise :{} , Max noise: {}".format(np.min(conditional_img_with_noise), np.max(conditional_img_with_noise)))
                                
                                display_true = True
                                if display_true:
                                        grid_img = torchvision.utils.make_grid(torch.from_numpy(conditional_img_with_noise), nrow=8, normalize=True)

                                        # Convert to NumPy array and transpose to (height, width, channels)
                                        grid_img_np = grid_img.permute(1, 2, 0).cpu().numpy()

                                        # Display the image
                                        plt.imshow(grid_img_np)
                                        plt.axis('off')
                                        plt.savefig('conditional_img_with_noise_grid.png',bbox_inches='tight', pad_inches=0)
                                
                                #conditional_img_with_noise = (conditional_img_batch_with_noise[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2
                                #conditional_img_with_noise = (conditional_img_with_noise * 255).astype(np.uint8)
                                #conditional_img_with_noise = torch.clamp(conditional_img_with_noise, 0, 1)

                                #cv2.imwrite('./conditional_img_with_noise.png', cv2.cvtColor(conditional_img_with_noise, cv2.COLOR_RGB2BGR))
                        
                        if n_noise:
                            x_t_1 = torch.from_numpy(conditional_img_with_noise).to(device)
                        else:   
                            x_t_1 = conditional_img_batch

                        # ... (Rest of the existing code)

                        # Generate samples
                        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
                        
                        # Transform and save the generated samples
                        transform = Transform.Resize((1 * 101, 1 * 101))
                        fake_sample = transform(fake_sample)

                        # Denormalize generated samples if necessary
                        #fake_sample = (fake_sample + 1) / 2
                        print("----------------------Reached here-----------------------------------------")
                        torchvision.utils.save_image(fake_sample, './samples_Non_Lenses_conditional_norm_sw_off.jpg', padding=0)

                        # ... (Rest of the existing code)
            
            print("Max of array: {}".format(torch.max(x_t_1)))
            
            print("Min of array: {}".format(torch.min(x_t_1)))
            
            transform = Transform.Resize((1*101,1*101))
            
                    
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1,T,  args)
            #fake_sample = to_range_0_1(fake_sample)
            
            fake_sample = transform(fake_sample)
            
            #np.save("X_array",fake_sample.cpu().detach().numpy())
            
            

            #fake_sample = fake_sample.permute(0,2,3,1)
            in_num = np.random.randint(1000000)
            fake_sample_transposed = np.transpose(fake_sample.cpu().detach().numpy(), (0, 2, 3, 1))
            #torchvision.utils.save_image(fake_sample, './samples_Non_Lenses_14k.jpg')
            #str_num = "npy_files_test/X_array_" + str(in_num)
            #np.save(str_num,fake_sample_transposed)   
    
            

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
    
   
                
