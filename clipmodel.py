import torch
import torchvision
import torchvision.transforms as T

import clip

import numpy as np
from matplotlib import pyplot as plt

class CLIPmodel:

    def __init__(self, model = "ViT-B/32"):
        self.model, self.preprocess = clip.load(model)
        self.model.cuda().eval()

    def preprocess_images(self, images):
        prep_images = (self.preprocess(img) for img in images)
        return torch.tensor(np.stack(prep_images)).cuda()
    
    def preprocess_tensor(self, tensor_image):
        return self.preprocess(tensor_image).unsqueeze(0)

    def tokenize_texts(self, texts):
        return clip.tokenize(["This is " + desc for desc in texts]).cuda()

    def get_features(self, image_tensor, tokenized_texts):
        return (self.model.encode_image(image_tensor).float(), self.model.encode_text(tokenized_texts).float())

    
    # multiple images and texts (images type Pil)
    def get_cosine_similarities(self, images, texts): 
        # Preprocess the images and tokenize the texts
        prep_images_tensor = self.preprocess_images(images)
        tokenized_texts = self.tokenize_texts(texts)

        # Extract features
        with torch.no_grad():
            image_features, text_features = self.get_features(prep_images_tensor, tokenized_texts)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the cosine similarity
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        return similarity
    
    
    def get_image_text_features(self, image, text):
        prep_images_tensor = torch.tensor(self.preprocess(image)).cuda().unsqueeze(0)
        tokenized_texts = clip.tokenize(["This is " +text]).cuda()

        # Extract features
        with torch.no_grad():
            image_features, text_features = self.get_features(prep_images_tensor, tokenized_texts)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


    # Get cosine sim for one image (type Pil) and one text (str)
    def get_cosine_similarity(self, image, text):
        image_features, text_features = self.get_image_text_features(image, text)

        # Compute the cosine similarity
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        return similarity[0][0]
    
    def get_cosine_difference(self, image, text):
        return 1 - self.get_cosine_similarity(image, text)
    
    """
    def get_cosine_similarity(self, image, text):# Get cosine sim for one image and one text
        if torch.is_tensor(image):
            image = T.ToPILImage()(image)
        prep_images_tensor = torch.tensor(np.stack(self.preprocess(image))).cuda()
        tokenized_texts = clip.tokenize(["This is " +text]).cuda()

        # Extract features
        with torch.no_grad():
            image_features, text_features = self.get_features(prep_images_tensor, tokenized_texts)

        # Normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute the cosine similarity
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        return similarity
"""

    def plot_cosine_similarity(similarity, images, texts, fig_size = (20, 14)):
        count = len(texts)

        plt.figure(figsize=fig_size)
        plt.imshow(similarity, vmin=0.1, vmax=0.3)

        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])
        for i, image in enumerate(images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)


"""
    __DEFAULT_MODEL_NAME = "ViT-B/32"
    def __init__(self, model_name, device):
        # save device
        self.__device = device
        # use default if name is None
        model_name = model_name if model_name is not None else CLIPwrapper.__DEFAULT_MODEL_NAME
        # load model
        self.__model, self.__preprocess = clip.load(model_name)
        self.__model.to(self.__device)
        # switch to training mode
        self.__model.train()
        
    def eval(self):
        self.__model.eval()
    
    def train(self):
        self.__model.train()

    def preproc_image(self, image):
        preproc_img = self.__preprocess(image)
        return torch.tensor(np.stack(preproc_img)).to(self.__device)

    def tokenize_prompt(self, prompt):
        return clip.tokenize(prompt).to(self.__device)

    def get_image_embedding(self, image):
        return self.__model.encode_image(image).float()

    def get_prompt_embedding(self, tokenized_prompt):
        prompt_feat_vect = self.__model.encode_text(tokenized_prompt).float()
        return prompt_feat_vect

    def cos_dist(self, image, prompt):
        # preprocess the images and tokenize the prompts
        preproc_images    = self.preproc_image(image)
        tokenized_prompts = self.tokenize_prompt(prompt)

        # extract image and prompt embeddings (feature vectors)
        with torch.no_grad():
            image_embedding = self.get_image_embedding(preproc_images)
            prompt_embedding = self.get_prompt_embedding(tokenized_prompts)
        
        return cosine(image_embedding, prompt_embedding)

    @staticmethod
    def plot_cosine_similarity(similarity, images, texts, fig_size = (20, 14)):
        count = len(texts)

        plt.figure(figsize=fig_size)
        plt.imshow(similarity, vmin=0.1, vmax=0.3)

        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])
        for i, image in enumerate(images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)
"""
