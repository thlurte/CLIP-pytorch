import torch
import torch.nn
import timm
from transformers import DistilBertModel, DistilBertConfig

class ImageEncoder(torch.nn.Module):
    def __init__(self,config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_name = config['img_model']
        self.pretrained = config['pretrained']
        self.trainable = config['trainable']
        
        self.model = timm.create_model(self.model_name, self.pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requies_grad = self.trainable
            
    def forward(self,x):
        return self.model(x)
        
        
class TextEncoder(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_name = config['txt_model']
        self.pretrained = config['pretrained']
        self.trainable = config['trainable']
        
        if self.pretrained:
            self.model = DistilBertModel.from_pretrained(self.model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig)
            
        for p in self.model.parameters():
            p.requires_grad = self.trainable
            
        self.target_token_idx = 0
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx,:]