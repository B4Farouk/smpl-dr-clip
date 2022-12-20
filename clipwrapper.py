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
        self.__model.eval()

    ###################
    # IMAGE FUNCTIONS
    ###################
    
    @staticmethod
    def _rgb_channels(img_t):
        """
        Extracts the rgb channels from an image tensor of shape (W, H, 4).
        
        Args:
            img_t: the image tensor to be processed.
        
        Returns:
            An image tensor of shape (3, W, H).
        """
        # remove the alpha component: from (W, H, 4) to (W, H, 3) 
        img_t = img_t[:,:,:3] 
        # create RGB channels: from (W, H, 3) to (3, W, H)
        img_t = torch.permute(img_t, (2, 0, 1))
        return img_t
    
    def proc_img(self, img_t):
        """
        Preprocesses an image tensor of shape (1, W, H, 4) for CLIP.
        
        Args:
            img_t: the image tensor to be preprocessed.
        
        Returns:
            A preprocessed image tensor of shape (3, W, H).
        """
        # from (1, W, H, 4) to (W, H, 4)
        img_t = img_t.squeeze()
        # get rgb channels: result is (3, W, H)
        img_t = CLIPwrapper._rgb_channels(img_t)
        # apply custom image preprocessing
        transformed_img_t = CLIPwrapper.__IMAGE_TRANSFORM(img_t)
        return transformed_img_t
    
    def procimg_emb(self, proc_img_t):
        """
        Computes the image embedding of a preprocessed image tensor.

        Args:
            proc_img_t: the preprocessed image tensor.

        Returns:
            the image embedding.
        """
        proc_img_t = torch.unsqueeze(proc_img_t, 0)
        img_emb = self.__model.encode_image(proc_img_t)
        return img_emb
    
    def img_emb(self, img_t):
        """
        Computes the image embedding of a raw image tensor.

        Args:
            img_t: the raw image tensor.

        Returns:
            the image embedding.
        """
        proc_img_t = self.proc_img(img_t)
        return self.procimg_emb(proc_img_t)
    
    ############################
    # MULTIPLE IMAGES FUNCTIONS
    ############################
    
    @staticmethod
    def _n_rgb_channels(imgs_t):
        """
        Extracts the rgb channels from a collection of image tensors of shape (N, W, H, 4).
        
        args:
            imgs_t: the image tensors to be processed.
        
        returns:
            An image tensor of shape (N, 3, W, H).
        """
        # remove the alpha component: from (N, W, H, 4) to (N, W, H, 3)
        imgs_t = imgs_t[...,:3]
        # create RGB channels: from (N, W, H, 3) to (N, 3, W, H)
        return torch.permute(imgs_t, (0, 3, 1, 2))
    
    def proc_imgs(self, imgs_t):
        """
        Preprocesses a collection of image tensors of shape (N, W, H, 4) for CLIP.
        
        args:
            img_t: the image tensors to be preprocessed.
        
        returns:
            A collection of preprocessed image tensors of shape (N, 3, W, H).
        """
        # get rgb channels: result is (N, 3, W, H)
        imgs_t = CLIPwrapper._n_rgb_channels(imgs_t)
        # apply custom image preprocessing
        return CLIPwrapper.__IMAGE_TRANSFORM(imgs_t)       
    
    def procimgs_emb(self, proc_imgs_t):
        """
        Computes the image embeddings of a collection of preprocessed image tensors in parallel.

        Args:
            proc_img_t: the preprocessed image tensors.

        Returns:
            the image embeddings.
        """
        # encode_image can encode multiple images simultaneously
        imgs_embs = self.__model.encode_image(proc_imgs_t)
        return imgs_embs
    
    def imgs_embs(self, imgs_t):
        """
        Computes the image embeddings of a collection of preprocessed image tensors in parallel.

        Args:
            proc_img_t: the preprocessed image tensors.

        Returns:
            the image embeddings.
        """
        proc_imgs_t = self.proc_imgs(imgs_t)
        return self.procimgs_emb(proc_imgs_t)
    
    ###################
    # PROMPT FUNCTIONS
    ###################
    
    def tokenize(self, prompt):
        """
        Tokenizes a prompt.

        Args:
            prompt: the prompt

        Returns:
            a list of tokens
        """
        return clip.tokenize(prompt).to(self.__device)

    def tk_emb(self, prompt_tk):
        """
        Computes the prompt tokens' embedding.

        Args:
            prompt_tk: the prompt tokens.
            
        Returns:
            the prompt embedding.
        """
        return self.__model.encode_text(prompt_tk)
        
    def pmt_emb(self, prompt):
        """Computes the embedding of a prompt.

        Args:
            prompt: the prompt.

        Returns:
            the prompt embedding.
        """
        prompt_tk = self.tokenize(prompt)
        return self.tk_emb(prompt_tk)
    
    ###################
    # MULTIPLE PROMPTS FUNCTIONS
    ###################    
    
    def pmts_emb(self, prompts):
        prompts_tk = []
        for prompt in prompts:
            prompts_tk.append(self.tokenize(prompt))
        features = []
        for prompt_tk in prompts_tk:
            features.append(self.tk_emb(prompt_tk))
        return features
    
