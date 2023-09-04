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

# edit_alpha = np.delete(np.linspace(1, 1.2, 7), 3)
edit_alpha = np.linspace(-1, -2, 3)


def run(image, direction=None):
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
        input_cuda = image.cuda().float().unsqueeze(0)

        result_batch, latents, features = net(input_cuda, resize=False)

        result_batch = result_batch.cuda().float()
        result = common.tensor2im(result_batch[0])

        im_save_path = os.path.join(test_opts.exp_dir, os.path.basename(from_path))
        Image.fromarray(np.array(result)).save(im_save_path)

        if direction is not None:
            for idx, alpha in enumerate(edit_alpha):
                edit_latents = latents + alpha * direction

                if model_type == 'baseline':
                    # without SMART
                    img = net.decode(edit_latents, resize=False)
                    # img, _ = net.decode(edit_latents, resize=False)
                    if opts.resize_outputs:
                        img = net.face_pool(img)
                    img = img.cuda().float()
                    result = common.tensor2im(img[0])
                    im_save_path = os.path.join(test_opts.exp_dir, f'{idx}_' + os.path.basename(from_path))
                    Image.fromarray(np.array(result)).save(im_save_path)
                else:
                    # with SMART
                    beta1 = 1
                    beta2 = 1
                    img = net.decode(edit_latents, features=features, beta=(beta1, beta2), resize=False)
                    if opts.resize_outputs:
                        img = net.face_pool(img)
                    img = img.cuda().float()
                    result = common.tensor2im(img[0])
                    im_save_path = os.path.join(
                        test_opts.exp_dir, f'{idx}_{beta1}_{beta2}_' + os.path.basename(from_path)
                    )
                    Image.fromarray(np.array(result)).save(im_save_path)


if __name__ == '__main__':

    # inference single image
    from_path = '/mnt/hdd/ZhuangChenyi/StylePrompter_github/input/0.jpg'  # path to the input image
    from_im = Image.open(from_path)
    from_im = from_im.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    from_im = transform(from_im)

    # for edit task
    direction_ckpt = None  # path to the direction of target attribute
    if direction_ckpt:
        edit_direction = torch.tensor(np.load(direction_ckpt)).cuda().float()
    else:
        edit_direction = None

    run(from_im, direction=None)
