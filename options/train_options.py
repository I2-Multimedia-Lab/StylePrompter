from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', default='./results/', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 choices=['ffhq_encode', 'church_encode', 'afhq_encode'],
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--input_nc', default=3, type=int,
                                 help='Number of input image channels')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument('--tokens_num', default=18, type=int,
                                 help='Number of the embedded latent tokens, must fit to the output_size')
        self.parser.add_argument('--refined_layer', default=4, type=int,
                                 help='Layer index for SMART')

        self.parser.add_argument('--type', default='baseline', type=str,
                                 choices=['baseline', 'full'], help='train baseline or SMART')
        self.parser.add_argument('--device', default='cuda', type=str, help='device for training')
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=8, type=int,  help='Batch size for inference')
        self.parser.add_argument('--workers', default=6, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=6, type=int, help='Number of inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--start_from_latent_avg', default=True, type=bool,
                                 help='Whether to add average latents to the predicted latent codes')
        self.parser.add_argument('--learn_in_w', default=False, type=bool,
                                 help='Whether to learn in W space, default False as W+ space')

        self.parser.add_argument('--lpips_lambda', default=0.6, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--multi_id', default=False, type=bool,
                                 help='Whether to use multi-layer ID loss or not')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--align_lambda', default=0.1, type=float, help='align loss multiplier factor')

        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_ffhq'], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=model_paths['ckpt'], type=str,
                                 help='Path to StylePrompter model weights')

        self.parser.add_argument('--last_step', type=int, help='Set initial training steps for continue training')
        self.parser.add_argument('--max_steps', default=600000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=5000, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=5000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
