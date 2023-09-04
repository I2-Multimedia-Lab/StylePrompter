DATASET_PATH_STEM = ''
MODEL_PATH_STEM = ''

dataset_paths = {
    'celeba_train': DATASET_PATH_STEM + 'CelebAMask-HQ/CelebA-HQ-img',
    'celeba_train_w': DATASET_PATH_STEM + 'CelebAMask-HQ/CelebA-HQ-img',
    'celeba_test': DATASET_PATH_STEM + 'CelebAMask-HQ/first_1k',
    'ffhq': DATASET_PATH_STEM + 'FFHQ',
    'afhq_train': DATASET_PATH_STEM + 'afhq/train/wild',
    'afhq_test': DATASET_PATH_STEM + 'afhq/val/wild',
    'church_train': DATASET_PATH_STEM + 'lsun_church/train',
    'church_test': DATASET_PATH_STEM + 'lsun_church/val',
    'validate': DATASET_PATH_STEM + 'CelebAMask-HQ/CelebA-HQ-img',
    'test': '',
}

model_paths = {
    'ckpt': None,
    'stylegan_ffhq': MODEL_PATH_STEM + 'stylegan/stylegan2-ffhq-config-f.pt',
    'stylegan_afhq': MODEL_PATH_STEM + 'stylegan/afhqwild.pt',
    'stylegan_church': MODEL_PATH_STEM + 'stylegan/stylegan2-church-config-f.pt',
    'swinv2': MODEL_PATH_STEM + 'utility/swinv2_tiny_patch4_window16_256.pth',
    'ir_se50': MODEL_PATH_STEM + 'utility/model_ir_se50.pth',
    'moco': MODEL_PATH_STEM + 'utility/moco_v2_800ep_pretrain.pt',
    'parsing_net': MODEL_PATH_STEM + 'application/face-parsing/79999_iter.pth'
}
