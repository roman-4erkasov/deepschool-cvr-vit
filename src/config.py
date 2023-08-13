import typing as tp
from omegaconf import OmegaConf
from pydantic import BaseModel


class DataConfig(BaseModel):
    data_path: str
    ann_file: str
    batch_size: int
    n_workers: int


class Config(BaseModel):
    task_nm: str
    lr: float
    data_config: DataConfig
    
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    qkv_dim: int
    mlp_hidden_size: int 
    n_layers: int
    n_heads: int
    n_classes: int
    attention_dropout_rate: float
    mlp_dropout_rate: float
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        cfg = OmegaConf.to_container(
            OmegaConf.load(path), 
            resolve=True
        )
        return cls(**cfg)
