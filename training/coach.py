import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.style_prompter import StylePrompter
from training.ranger import Ranger
from PIL import Image


class Coach:
    def __init__(self, opts):
        self.opts = opts

        if self.opts.last_step:
            self.global_step = self.opts.last_step + 1
        else:
            self.global_step = 0

        self.device = self.opts.device

        # Initialize network
        self.net = StylePrompter(self.opts).to(self.device)

        # params_num = sum([param.nelement() for param in self.net.encoder.parameters()])
        # print(params_num)
        # return

        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            if 'ffhq' in self.opts.dataset_type or 'celeb' in self.opts.dataset_type:
                self.id_loss = id_loss.IDLoss().to(self.device).eval()
            else:
                self.id_loss = moco_loss.MocoLoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        self.latest_id_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y = batch
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latents, features = self.net(x)
                loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat)

                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 50 == 0):
                    self.parse_and_log_images(x, y, y_hat, title='images/train', id_logs=id_logs)
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('Training Finished.')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            x, y = batch

            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                y_hat, latents, features = self.net(x)

                loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat)

            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(x, y, y_hat,
                                      title='images/test', id_logs=id_logs,
                                      subscript='{:04d}'.format(batch_idx))

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        print("Start training SMART.")
        params = self.net.decoder.module.get_trainable_params()
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        # image loss
        if self.opts.id_lambda > 0:
            loss_id = self.id_loss(y_hat, y, x, multi=self.opts.multi_id)
            loss_dict['loss_id'] = float(loss_id)
            loss = loss_id * self.opts.id_lambda

        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y, y_hat)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda

        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, y, y_hat, title, id_logs=None, subscript=None, display_count=2):
        im_data = []
        if id_logs is not None:
            score_data = []
        else:
            score_data = None
        for i in range(display_count):
            cur_im_data = {
                'Input': common.tensor2im(x[i]),
                'Target': common.tensor2im(y[i]),
                'Output': common.tensor2im(y_hat[i])
            }
            if id_logs is not None:
                cur_score_data = id_logs[:, i]
                score_data.append(cur_score_data)
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, score_data=score_data, subscript=subscript)

    def log_images(self, name, im_data, score_data=None, subscript=None, log_latest=False):
        fig = common.vis_images(im_data, score_log=score_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict
