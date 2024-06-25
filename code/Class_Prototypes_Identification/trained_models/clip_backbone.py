"""
pip install ftfy regex tqdm
conda install git
pip install git+https://github.com/openai/CLIP.git
"""
import clip
import torch


class image_encoder(torch.nn.Module):
    def __init__(self, version="ViT-B/32"):
        super().__init__()
        self.model, _ = clip.load(version)
        self.model.float()

    def forward(self, x):
        x = self.model.encode_image(x)
        return x
