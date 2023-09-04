import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']), strict=True)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.cos_loss = torch.nn.CosineEmbeddingLoss()

    def extract_feats(self, x, multi=False):
        if not multi:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        if multi:
            x_feats = self.facenet(x, return_features=True)
        else:
            x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x, multi=False):
        x_feats = self.extract_feats(x, multi=multi)
        y_feats = self.extract_feats(y, multi=multi)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat, multi=multi)
        cos_target = torch.ones((y.shape[0], 1)).float().cuda()

        if multi:
            loss = 0.0
            for i in range(len(y_feats)):
                y_feats_detached = y_feats[i].detach()
                loss += self.cos_loss(y_feats_detached.flatten(start_dim=1), y_hat_feats[i].flatten(start_dim=1), cos_target)

        else:
            y_feats_detached = y_feats.detach()
            loss = self.cos_loss(y_hat_feats.flatten(start_dim=1), y_feats_detached.flatten(start_dim=1), cos_target)

        return loss
