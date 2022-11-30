import torch
import clip

import numpy as np

from scipy.spatial.distance import cosine

from matplotlib import pyplot as plt

class CLIPwrapper:
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
        preproc_img = self.preprocess(image)
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
