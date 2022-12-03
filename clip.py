import torch
import clip

from torchvision import Compose, Normalize

class CLIPwrapper:
    __DEFAULT_MODEL_NAME = "ViT-B/32"
    __IMAGE_TRANSFORM = Compose(
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    )
        
    def __init__(self, model_name, device):
        # save device
        self.__device = device
        # use default if name is None
        model_name = model_name if model_name is not None else CLIPwrapper.__DEFAULT_MODEL_NAME
        # load model
        self.__model, _ = clip.load(model_name)
        self.__model.to(self.__device)
        # switch to evaluation mode
        self.__model.eval()
        
    def eval(self):
        self.__model.eval()
    
    def train(self):
        self.__model.train()

    ###################
    # IMAGE FUNCTIONS
    ###################
    
    @staticmethod
    def _rgb_channels_t(image_t):
        # remove the alpha component: from (W, H, 4) to (W, H, 3) 
        image_t = image_t[:,:,:3] 
        # create RGB channels: from (W, H, 3) to (3, W, H)
        image_t = torch.permute(image_t, (2, 0, 1))
        return image_t
    
    def proc_image_t(self, img_t):
        # from (1, W, H, 4) to (W, H, 4)
        img_t = img_t.squeeze()
        # get rgb channels: result is (3, W, H)
        img_t = CLIPwrapper._rgb_channels_t(img_t)
        # apply custom image preprocessing
        return CLIPwrapper.__IMAGE_TRANSFORM(img_t)
    
    def proc_image_embedding(self, proc_img_t):
        img_emb = self.__model.encode_image(proc_img_t)
        return img_emb
    
    def image_embedding(self, img_t):
        proc_img_t = self.proc_image_t(img_t)
        return self.proc_image_embedding(proc_img_t)
      
    ###################
    # PROMPT FUNCTIONS
    ###################
    
    def tokenize_prompt(self, prompt):
        return clip.tokenize(prompt).to(self.__device)

    def prompt_tk_embedding(self, prompt_tk):
        prompt_emb = self.__model.encode_text(prompt_tk)
        return prompt_emb
        
    def prompt_embedding(self, prompt):
        prompt_tk = self.tokenize_prompt(prompt)
        return self.prompt_tk_embedding(prompt_tk)
    
    ###################
    # JOINT FUNCTIONS
    ###################    
        
    def joint_embedding(self, img_t, prompt):
        return self.image_embedding(img_t), self.prompt_embedding(prompt)
    
    def joint_img_embedding(self, img_t1, img_t2):
        return self.image_embedding(img_t1), self.image_embedding(img_t2)
    
    def joint_prompt_embedding(self, prompt1, prompt2):
        return self.prompt_embedding(prompt1), self.prompt_embedding(prompt2)