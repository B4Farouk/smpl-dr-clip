import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
#import torch.nn.functional as functional
import torch
#from pkg_resources import packaging

import Juliette_clip

def get_cosine_similarity(img_path, texts):

    model, preprocess = Juliette_clip.load("ViT-B/32") #choose the model ViT-B/32
    model.cuda().eval()

    # Load and preprocess the images
    images = []
    image_names = os.listdir(img_path)
    for name in image_names:
        images.append(Image.open(name))
    preprocessed_images = (preprocess(im) for im in images)

    image_input = torch.tensor(np.stack(preprocessed_images)).cuda()

    # Tokenize the texts
    text_tokens = Juliette_clip.tokenize(["This is " + desc for desc in texts]).cuda()

    # Building features
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    # Normalize the features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Get the cosine similarity
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    return similarity, images


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