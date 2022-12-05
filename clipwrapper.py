import torch
import clip

from torchvision.transforms import Normalize

class CLIPwrapper:
    __DEFAULT_MODEL_NAME = "ViT-B/32"
    __IMAGE_TRANSFORM = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        
    def __init__(self, model_name, device):
        # save device
        self.__device = device
        # use default if name is None
        model_name = model_name if model_name is not None else CLIPwrapper.__DEFAULT_MODEL_NAME
        # load model
        self.__model, _ = clip.load(model_name)
        self.__model.to(self.__device)
        # switch to evaluation mode
        self.__model.train()
        
    def eval(self):
        self.__model.eval()
    
    def train(self):
        self.__model.train()

    ###################
    # IMAGE FUNCTIONS
    ###################
    
    @staticmethod
    def _rgb_channels(img_t):
        # remove the alpha component: from (W, H, 4) to (W, H, 3) 
        img_t = img_t[:,:,:3] 
        # create RGB channels: from (W, H, 3) to (3, W, H)
        img_t = torch.permute(img_t, (2, 0, 1))
        return img_t
    
    def proc_img(self, img_t):
        # from (1, W, H, 4) to (W, H, 4)
        img_t = img_t.squeeze()
        # get rgb channels: result is (3, W, H)
        img_t = CLIPwrapper._rgb_channels(img_t)
        # apply custom image preprocessing
        transformed_img_t = CLIPwrapper.__IMAGE_TRANSFORM(img_t)
        return transformed_img_t
    
    def procimg_emb(self, proc_img_t):
        proc_img_t = torch.unsqueeze(proc_img_t, 0)
        img_emb = self.__model.encode_image(proc_img_t)
        return img_emb
    
    def img_emb(self, img_t):
        proc_img_t = self.proc_img(img_t)
        return self.procimg_emb(proc_img_t)
      
    ###################
    # PROMPT FUNCTIONS
    ###################
    
    def tokens(self, prompt):
        return clip.tokenize(prompt).to(self.__device)

    def tk_emb(self, prompt_tk):
        return self.__model.encode_text(prompt_tk)
        
    def pmt_emb(self, prompt):
        prompt_tk = self.tokens(prompt)
        return self.tk_emb(prompt_tk)
    
    #####################################################
    # IMAGE/IMAGE, PROMPT/PROMPT, IMAGE/PROMPT FUNCTIONS
    #####################################################    
        
    def embs(self, img_t, prompt):
        return self.img_emb(img_t), self.pmt_emb(prompt)
    
    def img_embs(self, img_t1, img_t2):
        return self.img_emb(img_t1), self.img_emb(img_t2)
    
    def pmt_embs(self, prompt1, prompt2):
        return self.pmt_emb(prompt1), self.pmt_emb(prompt2)