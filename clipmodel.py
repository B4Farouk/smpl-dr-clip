import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import clip

import numpy as np
from matplotlib import pyplot as plt


class CLIPmodel:
    preprocess = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # Our custom preprocessing to keep image as tensor

    def __init__(self, device, prompt, model = "ViT-B/32"):
        self.device = device
        self.model, _ = clip.load(model)

        self.model.to(self.device)
        #self.model.cuda().eval()
        self.model.eval()

        tokenized_prompt = self.tokenize_prompt(prompt)
        self.prompt_feature = self.get_feature_prompt(tokenized_prompt)
        self.prompt_feature /= self.prompt_feature.norm(dim=-1, keepdim=True)

    @staticmethod
    def _argb2rgb_tensor(image_tensor):
        image_tensor = image_tensor[:,:,:3] # remove alpha component
        image_tensor = torch.permute(image_tensor, (2, 0, 1)) # from (W, H, 3) to (3, W, H)
        return image_tensor

    def train(self):
        self.model.train()

    def preprocess_image_tensor(self, img_t):
        img_t = CLIPmodel._argb2rgb_tensor(img_t.squeeze())
        prep_img_t = CLIPmodel.preprocess(img_t)
        return prep_img_t

    def tokenize_prompt(self, text):
        return clip.tokenize(["This is " + text]).cuda()

    def get_feature_prompt(self, tokenized_prompt):
        return self.model.encode_text(tokenized_prompt)#.float()

    def get_feature_image(self, img_t):
        return (self.model.encode_image(img_t).float(), )

    
    # one tensor image
    def get_cosine_similarity(self, img_t, eps=1e-8): 
        # Preprocess the image
        prep_img_t = self.preprocess_image_tensor(img_t)
        img_feature = self.get_feature_image(prep_img_t)

        # Normalize the feature
        print("pre-norm image feature",img_feature)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        print("post-norm image feature",img_feature)

        # Compute the cosine similarity
        similarity = nn.CosineSimilarity(dim=1, eps=eps)(img_feature, self.prompt_feature)
        #similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        return similarity
    

    def get_cosine_difference(self, img_t, eps=1e-8):
        return 1 - self.get_cosine_similarity(img_t, eps=eps)
 
