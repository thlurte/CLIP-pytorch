import torch

def get_config(img_path: str, cap_path: str):
    return {
        'debug': False,
        'image_path':img_path,
        'caption_path': cap_path,
        'batch_size': 32,
        'num_workers': 4,
        'head_lr': 1e-3,
        'image_encoder_lr': 1e-4,
        'text_encoder_lr': 1e-5,
        'weight_decay': 1e-3,
        'patience': 1,
        'factor': 0.8,
        'epoch': 4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'img_model': 'resnet50',
        'img_embedding_dim': 2048,
        'txt_embedding_dim': 768,
        'txt_model': 'distilbert-base-uncased',
        'max_length': 200,
        
        'pretrained': True,
        'trainable': True,
        'temprature': 1.0,
        
        'size': 224,
        
        'num_projection_layers': 1,
        'projection_dim': 256,
        'dropout': 0.1
    }
    