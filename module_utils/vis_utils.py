import abc
import math
import os
import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List, Dict, Sequence, Optional
from diffusers.models.attention_processor import Attention
from torch.nn import functional as F
from diffusers.models.embeddings import apply_rotary_emb

def get_token_indices(tokenizer, prompt, mask_prompt):
    # 对于 T5，通常需要 add_special_tokens=False
    full_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(mask_prompt, add_special_tokens=False)
    
    indices = []
    target_len = len(target_ids)
    
    for i in range(len(full_ids) - target_len + 1):
        if full_ids[i : i + target_len] == target_ids:
            # 记录匹配成功的起始到结束的所有索引
            indices.extend(list(range(i, i + target_len)))
            
    if not indices:
        print(f"警告: 在 prompt 中未找到匹配的 '{mask_prompt}' token！")
    
    return indices

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


def show_cross_attention(prompt: str,
                         attention_store,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         orig_image=None,
                         save_dir: str = None
                         ):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = attention_store.aggregate(from_where).to(torch.float32).detach().cpu()
    images = []

    direction = from_where[0] if from_where else "unknown_dir"

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            print(decoder(int(tokens[i])))
            image = show_image_relevance(image_relevance=image, image=orig_image, relevnace_res=res, save_dir=save_dir, token_name=decoder(int(tokens[i])),direction=direction)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = text_under_image(image, decoder(int(tokens[i]))) #T5没有BoS
            images.append(image)

    image_all = view_images(np.stack(images, axis=0))
    return image_all


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16, save_dir: str = None, token_name: str = "unknown",direction: str = "unknown_dir"):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)

    if save_dir is not None:
        # 净化 token_name，防止带有影响文件系统的特殊字符（如空格、</w> 等）
        clean_token = "".join(x for x in token_name if x.isalnum() or x in ["_", "-"]).strip()
        if not clean_token:
            clean_token = "unknown"
            
        np.save(os.path.join(save_dir, f"raw_attn-{direction}-{clean_token}.npy"), image_relevance)
        
        # 原图存一次就够了，判断一下如果不存在再存
        orig_save_path = os.path.join(save_dir, "resized_orig.jpg")
        if not os.path.exists(orig_save_path):
            Image.fromarray(image).save(orig_save_path)

    image_relevance = np.where(image_relevance > 0.10, image_relevance, 0) #0.4 for other

    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
