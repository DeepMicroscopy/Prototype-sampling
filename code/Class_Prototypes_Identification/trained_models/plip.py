import torch
from transformers import CLIPModel, CLIPProcessor


# https://github.com/PathologyFoundation/plip
class image_encoder(torch.nn.Module):
    def __init__(self, model_name="vinid/plip"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.model = self.model.float()

    def forward(self, x):
        x = self.model.get_image_features(x)
        return x

class text_encoder(torch.nn.Module):
    def __init__(self, model_name="vinid/plip"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, x):
        x = self.processor(text=x, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        x = self.model.get_text_features(**x)
        return x
