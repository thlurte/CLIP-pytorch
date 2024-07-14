import torch
import torch.nn

class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.projection_dim = config['projection_dim']
        self.dropout = config['dropout']
        
        self.projection = torch.nn.Linear(embedding_dim, self.projection_dim)
        self.gelu = torch.nn.GELU()
        self.fc = torch.nn.Linear(self.projection_dim, self.projection_dim)
        
        self.dropout = torch.nn.Dropout(self.dropout)
        self.layer_norm = torch.nn.LayerNorm(self.projection_dim)
        
        
    def forward(self,x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
        