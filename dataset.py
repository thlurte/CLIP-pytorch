import cv2
import torch
import albumentations

class Dataset(torch.utils.data.dataset):
    def __init__(self,image_filenames, captions, tokenizer, transforms, config):
        
        # assert len(image_filenames) == len(list(captions)), "Image filenames and captions must have the same length"
        
        # Name of the Images
        self.image_filenames = image_filenames
        # Description of Images
        self.captions = list(captions)
        # Config
        self.config = config 
        
        # Encoded Captions
        self.encoded_captions = tokenizer(
            list(captions), 
            padding=True, 
            truncation = True,
            max_length = self.config['max_length']
        )
        
        self.transforms = transforms
        
    
    def __getitem__(self,idx: int):
        item = {
            key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()
        }
        
        image = cv2.imread(f"{self.config['image_path']}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.transforms(image=image)['image']
        
        # https://pytorch.org/docs/stable/generated/torch.permute.html 
        # https://stackoverflow.com/questions/77992977/how-does-tensor-permutation-work-in-pytorch
        # When we permute, the underlying data would remain the same but the interface would be different
        item['image'] = torch.tensor(image).permute(2,0,1).float()
        item['caption'] = self.captions[idx]
        
        return item
    
    def __len__(self):
        return len(self.captions)
    
    
def get_transforms(config: dict):
    return albumentations.Compose(
        [
            albumentations.Resize(config['size'],config['size'],always_apply=True),
            albumentations.Normalize(max_pixel_value=255.0,always_apply=True)
        ]
    )