from argparse import ArgumentParser
from configs.paths_config import model_paths, dataset_paths


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--exp_dir', type=str, default='./output',
                                 help='Path to experiment output directory')
        self.parser.add_argument('--data_path', type=str, default='./input',
                                 help='Path to directory of images to evaluate')
        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=model_paths['ckpt'], type=str,
                                 help='Path to StylePrompter model checkpoint')

        self.parser.add_argument('--couple_outputs', action='store_true',
                                 help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at 1024x1024')
        self.parser.add_argument('--resize_factors', type=str, default=None,
                                 help='For super-res, comma-separated resize factors to use for inference.')

        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=1, type=int,
                                 help='Number of test/inference dataloader workers')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
