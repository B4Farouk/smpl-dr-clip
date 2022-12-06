import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import clip

import numpy as np
from matplotlib import pyplot as plt


class CLIPwrapper:
    preprocess = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # Our custom preprocessing to keep image as tensor

    def __init__(self, prompt, model = "ViT-B/32"):
        self.model, _ = clip.load(model)
        self.model.eval()
        self.prompt_feature = self.get_feature_from_prompt(prompt)

    @staticmethod
    def _argb2rgb_tensor(img_t):
      # Check for different channels orders (to remove when not needed)
        c1,c2,c3 = img_t.shape
        if c3 in [3,4]:
            img_t = torch.permute(img_t, (2, 0, 1)) # from (W, H, 3) to (3, W, H)
            assert(c1 == c2)
        elif c2 in [3,4]:
            img_t = torch.permute(img_t, (1, 2, 0))
        else:
            assert(c2 == c3)
        img_t = img_t[:3,:,:] # remove alpha component
        return img_t
     
    def tokenize_prompt(self, text):
        return clip.tokenize(["This is " + text])

    def get_feature_from_tokenized_prompt(self, tokenized_prompt):
        return self.model.encode_text(tokenized_prompt)#.float()
    
    def get_feature_from_prompt(self, prompt):
        tokenized_prompt = self.tokenize_prompt(prompt).cuda()
        prompt_feature = self.get_feature_from_tokenized_prompt(tokenized_prompt)
        prompt_feature /= prompt_feature.norm(dim=-1, keepdim=True)
        return prompt_feature
        
    def preprocess_img_t(self, img_t):
        img_t = CLIPwrapper._argb2rgb_tensor(img_t.squeeze())
        prep_img_t = CLIPwrapper.preprocess(img_t)
        return prep_img_t.cuda()
    
    def get_feature_img_from_preprocessed_img(self, img_t):
        img_t = torch.unsqueeze(img_t, 0)
        #print(img_t.shape)
        return self.model.encode_image(img_t)#.float()
    
    def get_feature_img_from_t(self, img_t):
        prep_img_t = self.preprocess_img_t(img_t)
        img_feature = self.get_feature_img_from_preprocessed_img(prep_img_t)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        return img_feature
        
    # CALLABLE FUNCTIONS
    def change_prompt_to(self, prompt):
        self.prompt_feature = self.get_feature_from_prompt(prompt) 
    
    # one tensor image
    def get_cos_sim(self, img_t, eps=1e-8): 
        img_feature = self.get_feature_img_from_t(img_t)
        similarity = nn.CosineSimilarity(dim=1, eps=eps)(img_feature, self.prompt_feature)
        return similarity
    
    def get_cos_diff(self, img_t, eps=1e-8):
        return 1 - self.get_cos_sim(img_t, eps=eps)
    
    def train(self):
        self.model.train()  
 
