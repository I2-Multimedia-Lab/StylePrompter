import matplotlib

matplotlib.use('Agg')
import math
import torch
from torch import nn
from models.encoders import swin_encoder
from models.SGGenerator import SGGenerator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class StylePrompter(nn.Module):

    def __init__(self, opts):
        super(StylePrompter, self).__init__()
        self.opts = opts
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = swin_encoder.SwinEncoder(tokens_num=self.opts.tokens_num, img_size=256, window_size=16)
        self.decoder = SGGenerator(n_latent=self.opts.tokens_num, refined_layer=self.opts.refined_layer, output_size=self.opts.output_size)
        self.encoder = nn.DataParallel(self.encoder)
        self.decoder = nn.DataParallel(self.decoder)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def random_samples(self, z, input_is_latent=False, resize=False):
        if input_is_latent:
            images = self.decoder(z, input_is_latent=True)
        else:
            if z.ndim > 2:
                raise Exception('shape should be (B, 512) if input is not latent')
            else:
                w = self.decoder.module.generator.style(z)
                if w.ndim < 3:
                    codes = w.unsqueeze(1).repeat(1, self.opts.tokens_num, 1)
                if self.opts.start_from_latent_avg:
                    if self.opts.learn_in_w:
                        codes = codes + self.latent_avg.repeat(z.shape[0], 1)
                    else:
                        codes = codes + self.latent_avg.repeat(z.shape[0], 1, 1)

                images = self.decoder(codes, input_is_latent=True)

        if resize:
            images = self.face_pool(images)
        return images

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading encoder weights from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            try:
                print('Loading decoder weights from checkpoint: {}'.format(self.opts.checkpoint_path))
                self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            except Exception as e:
                print('Decoder weights from checkpoint is mismatched.')
                print('Loading generator weights from pretrained!')
                ckpt = torch.load(self.opts.stylegan_weights, map_location='cpu')
                self.decoder.module.generator.load_state_dict(ckpt['g_ema'], strict=True)
            self.__load_latent_avg(ckpt)
        else:
            raise "Please train baseline at first."

    def encode(self, x):
        b, _, _, _ = x.shape
        # codes, mod_weight, mod_bias = self.encoder(x)
        with torch.no_grad():
            codes, features = self.encoder(x)
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(b, 1)
            else:
                codes = codes + self.latent_avg.repeat(b, 1, 1)
        return codes, features

    def decode(self,
               codes,
               features=None,
               resize=True,
               randomize_noise=True,
               beta=1.0,
               flow=None
               ):
        images = self.decoder(codes,
                              features=features,
                              input_is_latent=True,
                              randomize_noise=randomize_noise,
                              beta=beta, flow=flow)

        if resize:
            images = self.face_pool(images)
        return images

    def forward(self, x, resize=True, randomize_noise=True, return_latents=True, beta=1.0):

        b, _, _, _ = x.shape
        # codes, mod_weight, mod_bias = self.encoder(x)
        with torch.no_grad():
            codes, features = self.encoder(x)
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(b, 1)
            else:
                codes = codes + self.latent_avg.repeat(b, 1, 1)

        images = self.decoder(codes,
                              features=features,
                              input_is_latent=True,
                              randomize_noise=randomize_noise,
                              beta=beta)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, codes, features
        else:
            return images, None, None

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
