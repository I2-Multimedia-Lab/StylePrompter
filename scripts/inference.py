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

edit_alpha = np.linspace(-1, 1, 21)


def run(direction=None):
    test_opts = TestOptions().parse()
    os.makedirs(test_opts.exp_dir, exist_ok=True)

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

        if direction is not None:
            edit_paths = [os.path.join(test_opts.exp_dir, f'edit_alpha_{i}') for i in range(len(edit_alpha))]
            [os.makedirs(edit_path, exist_ok=True) for edit_path in edit_paths]

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

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
    opts = Namespace(**opts)

    if model_type == 'baseline':
        net = StylePrompterBaseline(opts)
    else:
        net = StylePrompter(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()

            result_batch, latents, features = net(input_cuda, resize=False)

            toc = time.time()
            global_time.append(toc - tic)
            result_batch = result_batch.cuda().float()

            if direction is not None:
                for alpha in edit_alpha:
                    edit_latents = latents + alpha * direction
                    img = net.decode(edit_latents, features=features, beta=(1, 1), resize=False)
                    if opts.resize_outputs:
                        img = net.face_pool(img)
                    result_batch = torch.cat([result_batch, img])
                result_batch = result_batch.cuda().float()

        for i in range(opts.test_batch_size):
            result = common.tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]

            if opts.couple_outputs:
                input_im = common.log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                      np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)

            if direction is not None:
                for j in range(len(edit_alpha)):
                    result = common.tensor2im(result_batch[(j + 1) * opts.test_batch_size + i])
                    im_save_path = os.path.join(edit_paths[j], os.path.basename(im_path))
                    Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1

    result_str = 'Runtime {:.4f}+-{:.4f} for batch size {}'.format(
        np.mean(global_time), np.std(global_time), opts.test_batch_size
    )
    print(result_str)


if __name__ == '__main__':

    # for edit task
    direction_ckpt = None  # path to the direction of target attribute
    if direction_ckpt:
        edit_direction = torch.tensor(np.load(direction_ckpt)).cuda().float()
    else:
        edit_direction = None

    run(direction=edit_direction)
