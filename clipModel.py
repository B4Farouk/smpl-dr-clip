import numpy as np
from matplotlib import pyplot as plt
import torch

import clip


class ClipModel:

    def __init__(self, model = "ViT-B/32"):
        self.model, self.preprocess = clip.load(model)
        self.model.cuda().eval()

    def preprocess_images(self, images):
        prep_images = (self.preprocess(img) for img in images)
        return torch.tensor(np.stack(prep_images)).cuda()

    def tokenize_texts(self, texts):
        return clip.tokenize(["This is " + desc for desc in texts]).cuda()

    def get_features(self, image_tensor, tokenized_texts):
        return (self.model.encode_image(image_tensor).float(), self.model.encode_text(tokenized_texts).float())

    def get_cosine_similarity(self, images, texts):
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
