
import torch
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
        
    def forward(self,batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(input_ids=batch['input_ids'],attention_mask=batch["attention_mask"])
        
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        logits = (text_embeddings @ image_embeddings.T) / self.temprature
        image_similarity = image_embeddings @ image_embeddings.T
        text_similarity = text_embeddings @ text_embeddings.T
        targets = functional.softmax(
            (image_similarity+text_similarity) / 2 * self.temprature, dim=-1
        )
        text_loss = cross_entropy(logits,targets, reduction='none')
        image_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (image_loss + text_loss) / 2.0
        return loss.mean()
        
        