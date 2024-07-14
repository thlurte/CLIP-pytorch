
import torch
from torch import nn
from torch.nn import functional

from encoders import ImageEncoder, TextEncoder
from project import ProjectionHead

from config import get_config

config = get_config('/home/ahmed/lab/CLIP-pytorch/archive/Images','/home/ahmed/lab/CLIP-pytorch/archive')

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self,margin=1.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
        
    def forward(self,output1,output2,label):
        euclidean_distance = nn.functional.pairwise_distance(output1,output2)
        loss_contrastive = torch.mean((1-label)*torch.pow(euclidean_distance,2)+(label)*torch.pow(torch.clamp(self.margin-euclidean_distance,min=0.0),2))
        return loss_contrastive

class CLIP(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        
        self.config = config
        self.temprature = config['temprature']
        self.image_embedding = config['img_embedding_dim']
        self.text_embedding = config['txt_embedding_dim']
        
        self.image_encoder = ImageEncoder(self.config)
        self.text_encoder = TextEncoder(self.config)
        self.image_projection = ProjectionHead(embedding_dim=self.image_embedding, config=config)
        self.text_projection = ProjectionHead(embedding_dim=self.text_embedding, config=config)
        self.contrastive_loss = ContrastiveLoss()
        
    def forward(self,batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(input_ids=batch['input_ids'],attention_mask=batch["attention_mask"])
        
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        labels = torch.eye(batch['image'].size(0)).to(image_embeddings.device)

        loss = self.contrastive_loss(image_embeddings, text_embeddings, labels)
        return loss
        
        