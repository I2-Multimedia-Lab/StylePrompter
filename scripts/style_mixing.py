import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
sys.path.append(".")
sys.path.append("..")
from argparse import Namespace
from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from models.style_prompter import StylePrompter
from models.style_prompter_baseline import StylePrompterBaseline
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils import common
from options.test_options import TestOptions
import torchvision.transforms as transforms


def run(input1, input2):
    test_opts = TestOptions().parse()
    os.makedirs(test_opts.exp_dir, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    print(opts)
    opts.update(vars(test_opts))
    model_type = opts['type']
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'tokens_num' not in opts:
        opts['tokens_num'] = 18
    if 'refined_layer' not in opts:
        opts['refined_layer'] = 4
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    if model_type == 'baseline':
        net = StylePrompterBaseline(opts)
    else:
        net = StylePrompter(opts)
    net.eval()
    net.cuda()

    with torch.no_grad():

        image1 = input1.cuda().float().unsqueeze(0)
        result_batch, latents1, features = net(image1, resize=False)
        result_batch = result_batch.cuda().float()
        result_1 = common.tensor2im(result_batch[0])
        im_save_path = os.path.join(test_opts.exp_dir, 'image1.jpg')
        Image.fromarray(np.array(result_1)).save(im_save_path)

        image2 = input2.cuda().float().unsqueeze(0)
        result_batch, latents2, features = net(image2, resize=False)
        result_batch = result_batch.cuda().float()
        result_2 = common.tensor2im(result_batch[0])
        im_save_path = os.path.join(test_opts.exp_dir, 'image2.jpg')
        Image.fromarray(np.array(result_2)).save(im_save_path)

        # gradually replacing
        for mixing_num in range(0, 18):
            mixed_latents = latents1[0].clone().detach()
            mixed_latents[:mixing_num] = latents2[0, :mixing_num]

            mixing_img = net.decode(mixed_latents.unsqueeze(0), resize=False)
            mixing_img = mixing_img.cuda().float()
            mixing_img = common.tensor2im(mixing_img[0])

            im_save_path = os.path.join(test_opts.exp_dir, f'mixing_gr_{mixing_num}.jpg')
            Image.fromarray(np.array(mixing_img)).save(im_save_path)

        # one-layer exchanging
        for mixing_num in range(0, 18):
            mixed_latents = latents1[0].clone().detach()
            mixed_latents[mixing_num] = latents2[0, mixing_num]

            mixing_img = net.decode(mixed_latents.unsqueeze(0), resize=False)
            mixing_img = mixing_img.cuda().float()
            mixing_img = common.tensor2im(mixing_img[0])

            im_save_path = os.path.join(test_opts.exp_dir, f'mixing_oe_{mixing_num}.jpg')
            Image.fromarray(np.array(mixing_img)).save(im_save_path)

        # interpolate
        interpolate_alpha = np.linspace(0, 1, 11)
        for idx, alpha in enumerate(interpolate_alpha):
            source_latent = latents1[0].clone().detach()
            ref_latent = latents2[0].clone().detach()
            mixed_latents = source_latent * interpolate_alpha[idx] + ref_latent * (1 - interpolate_alpha[idx])

            mixing_img = net.decode(mixed_latents.unsqueeze(0), resize=False)
            mixing_img = mixing_img.cuda().float()
            mixing_img = common.tensor2im(mixing_img[0])

            im_save_path = os.path.join(test_opts.exp_dir, f'mixing_interpolate_{idx}.jpg')
            Image.fromarray(np.array(mixing_img)).save(im_save_path)


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # path to the input image 1
    from_path = '/mnt/hdd/ZhuangChenyi/StylePrompter_github/input/0.jpg'
    from_im = Image.open(from_path)
    from_im = from_im.convert('RGB')
    from_im = transform(from_im)

    # path to the input image 2
    to_path = '/mnt/hdd/ZhuangChenyi/StylePrompter_github/input/1.jpg'
    to_im = Image.open(to_path)
    to_im = to_im.convert('RGB')
    to_im = transform(to_im)

    run(from_im, to_im)
