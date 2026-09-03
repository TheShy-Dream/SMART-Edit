from ast import List
import math
import random
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from PIL import Image
from torchvision.utils import save_image
from misc.frequency_utils import freq_com,mask_freq_augment_fusion,mask_freq_augment_fusion_v2,mask_freq_augment_fusion_v3,mask_freq_augment_fusion_v4
import os
import torch
import torch.nn.functional as F
from module_utils.guidance_function import get_centroid
from module_utils.attentionstore import AttentionStoreFlux, AttentionStoreDiT


def region_growing_torch(
    heatmap, 
    patch_size=1, 
    threshold=0.1, 
    fix_margin=0.0, 
    fix_adjacent_only=False,
    max_iter=100,
    use_centroid_seed=True,
    soft_coefficient=10.0  # 新增：用于缩放 Soft Mask 强度的系数
):
    """
    基于 PyTorch 实现的热力图区域生长算法
    heatmap: [H, W] 或 [1, H, W] 的 torch.Tensor
    """
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

    orig_dtype = heatmap.dtype
    heatmap = heatmap.float()
    
    device = heatmap.device
    _, _, h, w = heatmap.shape

    def get_gaussian_kernel(kernel_size=3, sigma=1.0):
        coords = torch.arange(kernel_size).to(device) - (kernel_size - 1) / 2.
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

    kernel = get_gaussian_kernel(3, 0.8)
    blurred = F.conv2d(heatmap, kernel, padding=1)

    if patch_size > 1:
        patch_means = F.avg_pool2d(blurred, kernel_size=patch_size, stride=patch_size)
    else:
        patch_means = blurred
    
    H, W = patch_means.shape[-2:]

    if use_centroid_seed:
        attn_for_centroid = patch_means.squeeze(0).squeeze(0)  # [H, W]
        if attn_for_centroid.ndim == 1:
            attn_for_centroid = attn_for_centroid.view(H, W)
        
        # 假设外部定义了 get_centroid
        centroid_coords = get_centroid(attn_for_centroid)  
        seed_w, seed_h = centroid_coords[0].round().long(), centroid_coords[1].round().long()
        
        seed_h = torch.clamp(seed_h, 0, H - 1)
        seed_w = torch.clamp(seed_w, 0, W - 1)
        
        seed_mask = torch.zeros((1, 1, H, W), device=device)
        seed_mask[0, 0, seed_h, seed_w] = 1.0
    else:
        global_mean = patch_means.mean()
        global_std = patch_means.std()
        seed_mask = (patch_means > (global_mean + 0.5 * global_std)).float()

        if seed_mask.sum() == 0:
            flat_idx = torch.argmax(patch_means)
            seed_mask.view(-1)[flat_idx] = 1.0

    current_mask = seed_mask.clone()
    kernel_8 = torch.ones((1, 1, 3, 3), device=device)
    
    for _ in range(max_iter):
        region_vals = patch_means[current_mask > 0]
        region_mean = region_vals.mean() if region_vals.numel() > 0 else 0.0
        
        dilated = (F.conv2d(current_mask, kernel_8, padding=1) > 0).float()
        neighbors = (dilated - current_mask) > 0
        diff_cond = torch.abs(patch_means - region_mean) < threshold
        new_mask = current_mask + (neighbors & diff_cond).float()
        
        if torch.equal(new_mask, current_mask):
            break
        current_mask = new_mask

    region_mean = patch_means[current_mask > 0].mean()
    if fix_adjacent_only:
        dilated = (F.conv2d(current_mask, kernel_8, padding=1) > 0).float()
        to_add = (dilated > current_mask) & (patch_means > (region_mean + fix_margin))
    else:
        to_add = (current_mask == 0) & (patch_means > (region_mean + fix_margin))
    
    current_mask[to_add] = 1.0

    if patch_size > 1:
        full_mask = F.interpolate(current_mask, size=(h, w), mode='nearest')
    else:
        full_mask = current_mask

    def morph_op(m, op='dilate', iters=1):
        k = torch.ones((1, 1, 3, 3), device=device)
        for _ in range(iters):
            if op == 'dilate':
                m = (F.conv2d(m, k, padding=1) > 0).float()
            elif op == 'erode':
                m = (F.conv2d(m, k, padding=1) >= 9).float()
        return m

    full_mask = morph_op(full_mask, 'dilate', 1)
    full_mask = morph_op(full_mask, 'erode', 1)
    full_mask = morph_op(full_mask, 'erode', 1)
    full_mask = morph_op(full_mask, 'dilate', 3)

    # 利用区域生长的边界约束原有的 Sigmoid 缩放响应
    scaled_heatmap = torch.sigmoid(soft_coefficient * (heatmap - 0.5))
    soft_mask = full_mask * scaled_heatmap
    
    # 额外高斯模糊：消除由于二值 mask 截断带来的边缘锯齿，形成真正平滑的 Soft 边缘
    smooth_kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0)
    soft_mask = F.conv2d(soft_mask, smooth_kernel, padding=2)
    soft_mask = torch.clamp(soft_mask, 0.0, 1.0)

    return soft_mask.squeeze().to(orig_dtype)  # 输出 Soft Mask [H, W]

def precise_semantic_localization_v2t(A_v_to_t, A_v_to_v, A_t_to_t, target_indices):

    """
    Precise Semantic Localization (PSL) 核心函数实现
    
    参数:
    - A_v_to_t: 融合后的 cross-attention 图, 形状为 [M, N] (M: 视觉 token 数, N: 文本 token 数)
    - A_v_to_v: 融合后的视觉 self-attention 矩阵, 形状为 [M, M]
    - A_t_to_t: 融合后的文本 self-attention 矩阵, 形状为 [N, N]
    - target_indices: 目标语义对应的文本 token 索引列表 (例如 ["a", "dog"] 对应的索引)
    
    返回:
    - M_refined: 精准定位后的语义掩码 [M]
    """
    
    # linalg.inv 不支持低精度（如 bfloat16），这里统一在 float32 上做 PSL，
    # 最后再转回原 dtype，避免运行时报错。
    out_dtype = A_v_to_t.dtype
    A_v_to_t = A_v_to_t.float()
    A_v_to_v = A_v_to_v.float()
    A_t_to_t = A_t_to_t.float()

    # 将视觉自注意力矩阵进行行归一化，使其成为行随机矩阵
    M_v = F.softmax(A_v_to_v, dim=-1) # 捕捉图像 token 间的相似性，用于修复内部空洞和边界
    
    # A_t_to_t 记录了文本语义间的纠缠，通过求逆操作来抵消这种耦合
    M_t = F.softmax(A_t_to_t, dim=-1)
    # 对角正则化提升数值稳定性（避免极端情况下不可逆）
    eye = torch.eye(M_t.shape[-1], device=M_t.device, dtype=M_t.dtype)
    M_t_inv = torch.inverse(M_t + 1e-6 * eye)
    
    # 消除 cross-attention 中由于文本关联导致的错误激活
    decoupled_attn = torch.matmul(A_v_to_t, M_t_inv)
    
    # 提取用户想要编辑的具体语义列
    selected_attn = decoupled_attn[:, target_indices]
    # 如果对应多个 token（如 "pink harness"），通常取平均或最大值
    selected_attn = selected_attn.mean(dim=-1, keepdim=True)
    
    # 利用视觉特征的相似性，将激活区域扩散到完整的物体上
    refined_map = torch.matmul(M_v, selected_attn)
    
    map_min = refined_map.min()
    map_max = refined_map.max()
    M_refined = (refined_map - map_min) / (map_max - map_min + 1e-8)
    
    return M_refined.squeeze().to(out_dtype)

def precise_semantic_localization_t2v(A_t_to_v, A_v_to_v, A_t_to_t, target_indices):
    """
    针对 Text-to-Vision 注意力图的 PSL 实现
    参数:
    - A_t_to_v: [N, M] (N: 文本, M: 视觉) -> 每一行是一个词在图上的投影
    - A_v_to_v: [M, M] -> 视觉自注意力
    - A_t_to_t: [N, N] -> 文本自注意力

    返回:
    - M_refined: 精准定位后的语义掩码 [M]
    """
    out_dtype = A_t_to_v.dtype
    A_t_to_v = A_t_to_v.float()
    A_v_to_v = A_v_to_v.float()
    A_t_to_t = A_t_to_t.float()

    # A_t_to_v 的查询向量是文本，因此文本间的纠缠会影响“哪一行”被激活
    M_t = F.softmax(A_t_to_t, dim=-1)
    eye = torch.eye(M_t.shape[-1], device=M_t.device)
    M_t_inv = torch.inverse(M_t + 1e-6 * eye)
    
    # 核心步骤 I：左乘逆矩阵，消除文本 Query 侧的语义纠缠
    # [N, N] * [N, M] -> [N, M]
    decoupled_attn = torch.matmul(M_t_inv, A_t_to_v)

    # 提取目标单词对应的“投影图”
    selected_attn = decoupled_attn[target_indices, :] # [len(target_indices), M]
    selected_attn = selected_attn.mean(dim=0, keepdim=True) # [1, M]

    # 这一步将词的投影利用视觉相似性进行扩散
    M_v = F.softmax(A_v_to_v, dim=-1) # [M, M]
    
    # 核心步骤 II：利用视觉关联矩阵细化投影
    # [M, M] * [M, 1] -> [M, 1]
    refined_map = torch.matmul(M_v, selected_attn.transpose(-2, -1))
    
    map_min, map_max = refined_map.min(), refined_map.max()
    M_refined = (refined_map - map_min) / (map_max - map_min + 1e-8)

    return M_refined.squeeze().to(out_dtype)


def scaled_dot_product_attention_new(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    txt_length: int,
    is_causal: bool = False,
    token_indices_clip: int | list[int] =2,
    token_indices_t5: int | list[int] =2,
    attn_mask: torch.FloatTensor | None = None,
    scale: float | None = None,
    coefficient: float = 10.0,
    thresh: float = 0.10,
    alpha: float = 1.2,
    d_s: float = 0.3,
    weight: float = 0.7,
    model_type: str = 'flux',
    hard_mask_thresh: float = 0.5
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if model_type == "flux":
        txt_txt_self = attn_weight[:, :, :txt_length, :txt_length].mean(dim=(0,1)) #upper left part 左上角 [512,512]
        img_txt_cross = attn_weight[:, :, txt_length:, :txt_length].mean(dim=(0,1))  # lower left part 左下角 [512,1024]
        txt_img_cross = attn_weight[:, :, :txt_length, txt_length:].mean(dim=(0,1)) # upper right part 右上角 [1024,512]
        img_img_self = attn_weight[:, :, txt_length:, txt_length:].mean(dim=(0,1)) # lower right part 右下角 [1024,1024]

        img_token_count = img_img_self.shape[0]
        H = W = int(math.sqrt(img_token_count))

        # v2t: [img_tokens, 1], t2v: [img_tokens]
        norm_heatmap_v2t = precise_semantic_localization_v2t(A_v_to_t=img_txt_cross, A_v_to_v=img_img_self,A_t_to_t=txt_txt_self, target_indices=token_indices_t5)
        norm_heatmap_t2v = precise_semantic_localization_t2v(A_t_to_v=txt_img_cross, A_v_to_v=img_img_self,A_t_to_t=txt_txt_self, target_indices=token_indices_t5)

        heatmap_v2t = norm_heatmap_v2t.view(H, W)
        heatmap_t2v = norm_heatmap_t2v.view(H, W)
        # First: fuse heatmaps with mask_freq_augment_fusion_v4
        norm_heatmap = mask_freq_augment_fusion_v4(mask_1=heatmap_v2t, mask_2=heatmap_t2v, alpha=alpha, d_s=d_s, weight=weight)
        # Then: apply region_growing_torch on the fused heatmap
        soft_mask_img = region_growing_torch(norm_heatmap, soft_coefficient=coefficient)
        hard_mask_img = (soft_mask_img > hard_mask_thresh).float()

    if model_type == "sd35":
        if attn_weight.shape[0] > 1:
            attn_uncond, attn_cond = attn_weight.chunk(2, dim=0)
        else:
            attn_cond = attn_weight
        # SD3.5 joint sequence order inside attention is [img | txt].
        # txt = CLIP(77) + T5(256), i.e. CLIP first then T5.
        clip_len = 77
        t5_len = 256
        txt_clip_len = min(clip_len, txt_length)
        txt_t5_start = txt_clip_len
        txt_t5_end = min(txt_t5_start + t5_len, txt_length)

        # 修复：直接在截取时进行 mean(0,1) 降维，消除 Batch 和 Heads 维度，变成标准的 2D 张量
        
        # (text query, text key)
        txt_txt_self = attn_cond[:, :, -txt_length:, -txt_length:].mean(dim=(0,1))  # lower right part 右下角
        clip_clip_self = txt_txt_self[:txt_clip_len, :txt_clip_len]
        t5_t5_self = txt_txt_self[txt_t5_start:txt_t5_end, txt_t5_start:txt_t5_end]

        # (img query, text key)
        img_txt_cross = attn_cond[:, :, :-txt_length, -txt_length:].mean(dim=(0,1))  # upper right part 右上角
        img_clip_cross = img_txt_cross[:, :txt_clip_len]
        img_t5_cross = img_txt_cross[:, txt_t5_start:txt_t5_end]

        # (text query, img key)
        txt_img_cross = attn_cond[:, :, -txt_length:, :-txt_length].mean(dim=(0,1))  # lower left part 左下角
        clip_img_cross = txt_img_cross[:txt_clip_len, :]
        t5_img_cross = txt_img_cross[txt_t5_start:txt_t5_end, :]
        # Backward-compat alias for your placeholder name.

        # (img query, img key)
        img_img_self = attn_cond[:, :, :-txt_length, :-txt_length].mean(dim=(0,1))  # upper left part 左上角
        
        norm_heatmap_clip_v2t = precise_semantic_localization_v2t(
            A_v_to_t=img_clip_cross, 
            A_v_to_v=img_img_self, 
            A_t_to_t=clip_clip_self, 
            target_indices=token_indices_clip # Pass relevant CLIP indices
        )

        norm_heatmap_t5_v2t = precise_semantic_localization_v2t(
            A_v_to_t=img_t5_cross, 
            A_v_to_v=img_img_self, 
            A_t_to_t=t5_t5_self, 
            target_indices=token_indices_t5 # Pass relevant T5 indices
        )

        # t2v: text-to-visual attention heatmap (text token to visual position)
        # 这里聚合 img_clip_cross, img_t5_cross 到文本 token, 得到文本 token 在图片中的响应位置
        norm_heatmap_clip_t2v = precise_semantic_localization_t2v(
            A_t_to_v=clip_img_cross,
            A_t_to_t=clip_clip_self,
            A_v_to_v=img_img_self,
            target_indices=token_indices_clip,  # CLIP文本token index
        )

        norm_heatmap_t5_t2v = precise_semantic_localization_t2v(
            A_t_to_v=t5_img_cross,
            A_t_to_t=t5_t5_self,
            A_v_to_v=img_img_self,
            target_indices=token_indices_t5,  # T5文本token index
        )

        img_token_count = img_img_self.shape[0]
        H = W = int(math.sqrt(img_token_count))

        heatmap_clip_v2t = norm_heatmap_clip_v2t.view(H, W)
        heatmap_t5_v2t = norm_heatmap_t5_v2t.view(H, W)
        heatmap_clip_t2v = norm_heatmap_clip_t2v.view(H, W)
        heatmap_t5_t2v = norm_heatmap_t5_t2v.view(H, W)

        # First: fuse heatmaps with mask_freq_augment_fusion_v4
        fused_heatmap_clip = mask_freq_augment_fusion_v4(mask_1=heatmap_clip_v2t, mask_2=heatmap_clip_t2v, alpha=alpha, d_s=d_s, weight=weight)
        fused_heatmap_t5 = mask_freq_augment_fusion_v4(mask_1=heatmap_t5_v2t, mask_2=heatmap_t5_t2v, alpha=alpha, d_s=d_s, weight=weight)

        # Then: apply region_growing_torch on the fused heatmaps
        soft_mask_clip = region_growing_torch(fused_heatmap_clip, soft_coefficient=coefficient)
        soft_mask_t5 = region_growing_torch(fused_heatmap_t5, soft_coefficient=coefficient)

        soft_mask_img = (soft_mask_clip + soft_mask_t5) / 2.0

        hard_mask_img = (soft_mask_img > hard_mask_thresh).float()
    return attn_weight @ value, soft_mask_img, hard_mask_img

def scaled_dot_product_attention_simple(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool,device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight,attn_weight @ value

def auto_mask(
    load_list: list[str],
    mask_accumulator: torch.FloatTensor,
    thresh: float,
    attn_guidance_start_block: int,
    mask_num: int = 4,
):
    mask_list = []
    for img_path in load_list:
        load_mask_img = Image.open(img_path).convert("L")
        transform = transforms.PILToTensor()
        mask_tensor = transform(load_mask_img)
        mask_tensor = mask_tensor.to(device=mask_accumulator.device, dtype=mask_accumulator.dtype)
        mask_tensor /= 255.0
        mask_list.append(mask_tensor) #加载并预处理 mask 图像

    # Sort masks based on their activation levels
    mask_list.sort(key=lambda x: x.sum().item(), reverse=True) # 基于激活水平排序 mask
    # Select the 5 medium activated masks
    num_masks = len(mask_list)
    if num_masks > mask_num:
        # selected_masks = mask_list[num_masks//2 - mask_num : num_masks//2]
        # 选择从 attn_guidance_start_block 开始的 mask_num 个 mask
        attn_guidance_end_block = attn_guidance_start_block + mask_num
        if attn_guidance_end_block > num_masks - 1:
            selected_masks = mask_list[-mask_num:] ## 如果超出范围，取最后 mask_num 个
        else:
            selected_masks = mask_list[attn_guidance_start_block:attn_guidance_end_block]
    else:
        selected_masks = mask_list # 如果数量不足，使用所有 mask

    # Accumulate the selected masks
    for mask in selected_masks:
        mask_accumulator += mask

    mask_tensor = (mask_accumulator / len(selected_masks)).to(
        dtype=mask_accumulator.dtype
    )  # Average the masks and convert back to original dtype
    mask_tensor[mask_tensor >= thresh] = 1
    mask_tensor[mask_tensor < thresh] = 0

    return mask_tensor


def adaptive_attention(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    txt_shape: int,
    img_shape: int,
    cur_step: int,
    cur_block: int,
    attn_guidance_start_block: int,
    layer: list[int] = list(range(19)),  # noqa: B008 B006
    is_causal: bool = False,
    token_index: int = 2,
    attn_mask: torch.FloatTensor | None = None,
    scale: float | None = None,
    coefficient: float = 10.0,
    mask_num: int = 4,
    thresh: float = 0.3,
    highlight_factor: float = 2.0,  # Factor to increase weights in the masked area
    reduce_factor: float = 0.8,  # Factor to decrease weights in the unmasked area
) -> torch.FloatTensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    attn_uncond, attn_cond = attn_weight.chunk(2, dim=0)

    txt_img_cross = attn_cond[:, :, -img_shape:, :txt_shape]  # lower left part 左下角
    # each column maps to a token's heatmap
    token_heatmap = txt_img_cross[:, :, :, token_index]  # Shape: [1, 24, 1024]
    token_heatmap = token_heatmap.mean(dim=1)[0]  # Shape: [1024]
    min_val, max_val = token_heatmap.min(), token_heatmap.max()
    norm_heatmap = (token_heatmap - min_val) / (max_val - min_val)  #normalize 

    mask_img = torch.sigmoid(coefficient * (norm_heatmap - 0.5)) #rescale

    H = W = int(math.sqrt(mask_img.size(0))) 
    mask_img = mask_img.reshape(H, W) #根据self-attention的heatmap生成不同物件的mask

    save_path = f"heatmap/step_{cur_step}_layer_{cur_block}_token{token_index}.png"
    load_path = [f"heatmap/step_{cur_step - 1}_layer_{i}_token{token_index}.png" for i in layer]
    save_image(mask_img.unsqueeze(0), save_path)

    mask_img[mask_img >= thresh] = 1
    mask_img[mask_img < thresh] = 0

    mask_tensor = torch.zeros_like(mask_img)  # Set mask_tensor as a zero tensor
    if cur_step >= mask_num: #mask从哪个时间步开始
        mask_accumulator = torch.zeros_like(
            mask_tensor.unsqueeze(0), dtype=mask_img.dtype
        )  # Accumulator for averaging masks
        mask_tensor = auto_mask(
            load_path, mask_accumulator, thresh, attn_guidance_start_block, mask_num=mask_num
        )
        if cur_block == 1:
            save_image(
                mask_tensor,
                f"heatmap/average_heatmaps/step_{cur_step}_layer_{cur_block}_token{token_index}.png",
            )

    if not torch.all(mask_tensor == 0):
        mask_tensor = mask_tensor.reshape(1, H * W)
        mask_tensor = mask_tensor.unsqueeze(1).unsqueeze(-1)
        # Create a multiplier tensor: 2.0 where mask is active, 0.5 where mask is inactive.
        multiplier = torch.where(
            mask_tensor.bool(), torch.tensor(highlight_factor), torch.tensor(reduce_factor)
        ) #论文假定mask区域是比较重要的，所以对于mask的部分构建一个>1的乘子，对于非mask的部分构建一个<1的乘子
        attn_weight[:, :, -img_shape:, :15] *= multiplier #为什么文本只取前15个进行reweight有些奇怪

    return attn_weight @ value


class FluxAttnProcessorWithMemory:
    def __init__(self, block_idx: int = -1, txt_length=512, attention_store: AttentionStoreFlux = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessorWithMemory requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.memory = {}
        self.block_idx = block_idx
        self.txt_length = txt_length
        self.attention_store = attention_store

    def clear_memory(self):
        self.memory.clear()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        second_order: bool = False,
        timestep: int = -1,
        inject: bool = False,
        editing_strategy: Literal["replace_q", "add_q", "replace_k", "add_k", "replace_v", "add_v", "attn_guidance", "replace_qkv"] | None = None,
        noise_injecting_strategy: Literal["inject_q","inject_k","inject_v","inject_qk","inject_kv","inject_qv","inject_qkv"] | None = None,
        fusion_strategy: Literal["fusion_q","fusion_k","fusion_v","fusion_qk","fusion_kv","fusion_qv","fusion_qkv"] | None = None,
        inverse: bool = False,
        attn_guidance_start_block: int = -1,
        qkv_ratio: list[float] = [1.0, 1.0, 1.0],  # noqa: B006
        noise_coeff: list[float] = [0.2, 0.2 ,0.8],
        scale_coeff: float = 0.3,
        offset_coeff: float = 0.8,

        mask_fusion_strategy: Literal["fusion_q","fusion_k","fusion_v"] | None = None,
        attention_weight_control: bool = False,
        token_indices: int|list[int] = 2,
        coefficient: int = 10,
        thresh: float = 0.1,
        save_attn: bool = False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # For sharing value to insert similar features
        if encoder_hidden_states is None and inject:  
            q_feature_name = f"{timestep}_{second_order}_q"
            k_feature_name = f"{timestep}_{second_order}_k"
            v_feature_name = f"{timestep}_{second_order}_v"
            hard_mask_feature_name = f"{timestep}_{second_order}_hard_mask"
            soft_mask_feature_name = f"{timestep}_{second_order}_soft_mask"

            if inverse:
                # === Inversion Phase: Store Features ===
                if editing_strategy and "replace_qkv" in editing_strategy:
                    # 存储 Q, K, V
                    self.memory[q_feature_name] = (query * qkv_ratio[0]).cpu()
                    self.memory[k_feature_name] = (key * qkv_ratio[1]).cpu()
                    self.memory[v_feature_name] = (value * qkv_ratio[2]).cpu()
                else:
                    # 根据策略分别存储
                    if editing_strategy and "q" in editing_strategy:
                        self.memory[q_feature_name] = (query * qkv_ratio[0]).cpu()
                    if editing_strategy and "k" in editing_strategy:
                        self.memory[k_feature_name] = (key * qkv_ratio[1]).cpu()
                    if editing_strategy and "v" in editing_strategy:
                        self.memory[v_feature_name] = (value * qkv_ratio[2]).cpu()
            else:
                # === Generation/Editing Phase: Inject Features ===
                if editing_strategy == "replace_qkv":
                    if q_feature_name in self.memory:
                        if fusion_strategy and noise_injecting_strategy:
                            query = self.memory[q_feature_name].to(hidden_states.device)
                        elif noise_injecting_strategy and "q" in noise_injecting_strategy:
                            query += freq_com(
                                self.memory[q_feature_name].to(query.device),
                                query,
                                alpha=noise_coeff[0],
                                eta=noise_coeff[1],
                                beta = noise_coeff[2],
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                            )
                        elif fusion_strategy and "q" in fusion_strategy:
                            query = freq_com(
                                self.memory[q_feature_name].to(query.device),
                                query,
                                alpha=noise_coeff[0],
                                offset=offset_coeff,
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                                mode=2
                            )
                        else:
                            pass
                    if k_feature_name in self.memory:
                        if fusion_strategy and noise_injecting_strategy:
                            key = self.memory[k_feature_name].to(hidden_states.device)
                        elif noise_injecting_strategy and "k" in noise_injecting_strategy:
                            key += freq_com(
                                self.memory[k_feature_name].to(key.device),
                                key,
                                alpha=noise_coeff[0],
                                eta=noise_coeff[1],
                                beta=noise_coeff[2],
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                            )
                        elif fusion_strategy and "k" in fusion_strategy:
                            key = freq_com(
                                self.memory[k_feature_name].to(key.device),
                                key,
                                alpha=noise_coeff[0],
                                offset=offset_coeff,
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                                mode=2
                            )
                        else:
                            pass
                    if v_feature_name in self.memory:
                        if fusion_strategy and noise_injecting_strategy:
                            value = self.memory[v_feature_name].to(hidden_states.device)
                        elif noise_injecting_strategy and "v" in noise_injecting_strategy:
                            value += freq_com(
                                self.memory[v_feature_name].to(value.device),
                                value,
                                alpha=noise_coeff[0],
                                eta=noise_coeff[1],
                                beta = noise_coeff[2],
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                            )
                        elif fusion_strategy and "v" in fusion_strategy:
                            value = freq_com(
                                self.memory[v_feature_name].to(value.device),
                                value,
                                alpha=noise_coeff[0],
                                offset=offset_coeff,
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                                mode=2
                            )
                        else:
                            pass

                else:
                    if editing_strategy == "replace_q" and q_feature_name in self.memory:
                        query = self.memory[q_feature_name].to(hidden_states.device)
                    elif editing_strategy == "add_q" and q_feature_name in self.memory:
                        query += self.memory[q_feature_name].to(hidden_states.device)

                    if editing_strategy == "replace_k" and k_feature_name in self.memory:
                        key = self.memory[k_feature_name].to(hidden_states.device)
                    elif editing_strategy == "add_k" and k_feature_name in self.memory:
                        key += self.memory[k_feature_name].to(hidden_states.device)

                    if editing_strategy == "replace_v" and v_feature_name in self.memory:
                        value = self.memory[v_feature_name].to(hidden_states.device)
                    elif editing_strategy == "add_v" and v_feature_name in self.memory:
                        value += self.memory[v_feature_name].to(hidden_states.device)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)


        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if (
            encoder_hidden_states is not None and not inverse and editing_strategy == "attn_guidance"
        ):  # dual-stream block #MultiTurn Edit
            hidden_states = adaptive_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                txt_shape=encoder_hidden_states.shape[1],
                img_shape=hidden_states.shape[1],
                cur_step=timestep,
                cur_block=self.block_idx,
                attn_guidance_start_block=attn_guidance_start_block,
            ) #这里应该输入token index的
        elif attention_weight_control and inject:
            hidden_states, soft_mask, hard_mask = scaled_dot_product_attention_new(
                query=query,
                key=key,
                value=value,
                txt_length=self.txt_length,
                is_causal = False,
                token_indices_t5 = token_indices,
                attn_mask=attention_mask,
                scale=None,
                coefficient = coefficient,
                thresh = thresh,
                alpha=1.2,
                d_s = 0.3,
                weight=0.7,
                model_type = "flux",
            )

            if inverse:
                self.memory[hard_mask_feature_name] = hard_mask.detach().cpu()
                self.memory[soft_mask_feature_name] = soft_mask.detach().cpu()
                if "fusion_q" in mask_fusion_strategy:
                    self.memory[q_feature_name] = (query * qkv_ratio[0]).cpu()
                if "fusion_k" in mask_fusion_strategy:
                    self.memory[k_feature_name] = (key * qkv_ratio[1]).cpu()
                if "fusion_v" in mask_fusion_strategy:
                    self.memory[v_feature_name] = (value * qkv_ratio[2]).cpu()
            elif not inverse and mask_fusion_strategy == "fusion_v":
                if v_feature_name in self.memory and soft_mask_feature_name in self.memory:
                    v_old = self.memory[v_feature_name].to(hidden_states.device)
                    mask_old = self.memory[soft_mask_feature_name].to(hidden_states.device)
                    
                    # mask_old 形状通常是 [H, W]，需要展平并对齐维度
                    # Flux 的 Value 序列是 [Batch, Heads, Txt_len + Img_len, Head_dim]
                    # mask 只作用于 Img_len 部分
                    m = mask_old.reshape(1, 1, -1, 1) # [1, 1, Img_len, 1]
                    
                    # 分离文本和图像部分
                    v_img_current = value[:, :, self.txt_length:, :]
                    v_img_old = v_old[:, :, self.txt_length:, :]

                    eps = 1e-6
                    # 在序列维度 (dim=2) 上计算均值和标准差，保持维度以便广播
                    mu_current = v_img_current.mean(dim=2, keepdim=True)
                    std_current = v_img_current.std(dim=2, keepdim=True) + eps
                    
                    mu_old = v_img_old.mean(dim=2, keepdim=True)
                    std_old = v_img_old.std(dim=2, keepdim=True) + eps
                    
                    # 执行 AdaIN: 将当前特征的分布对齐到 old 特征的分布
                    v_img_adain = std_old * ((v_img_current - mu_current) / std_current) + mu_old
                    
                    # 掩码融合：Mask区域用新生成的(value)，非Mask区域用旧的(v_old)
                    v_img_fused = v_img_adain * m + v_img_old * (1 - m)
                    value = torch.cat([value[:, :, :self.txt_length, :], v_img_fused], dim=2)
                    value = value.to(query.dtype)
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        if save_attn:
            attn_weight, _= scaled_dot_product_attention_simple(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            self.attention_store(attn_weight)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class SD3AttnProcessorWithMemory:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, block_idx: int = -1, block_inter_type: str = "attn,", clip_txt_length=77, t5_txt_length=256, attention_store: AttentionStoreDiT = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SD3AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.memory = {}
        self.block_idx = block_idx
        self.block_inter_type = block_inter_type
        self.clip_txt_length = clip_txt_length
        self.t5_txt_length = t5_txt_length
        self.txt_length = self.clip_txt_length + self.t5_txt_length
        self.attention_store = attention_store

    def clear_memory(self):
        self.memory.clear()


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        second_order: bool = False,
        timestep: int = -1,
        inject: bool = False,
        editing_strategy: Literal["replace_q", "add_q", "replace_k", "add_k", "replace_v", "add_v", "attn_guidance", "replace_qkv"] | None = None,
        noise_injecting_strategy: Literal["inject_q","inject_k","inject_v","inject_qk","inject_kv","inject_qv","inject_qkv"] | None = None,
        inverse: bool = False,
        attn_guidance_start_block: int = -1,
        qkv_ratio: list[float] = [1.0, 1.0, 1.0],
        noise_coeff: list[float] = [0.2, 0.2 ,0.8],
        scale_coeff: float = 0.3,

        mask_fusion_strategy: Literal["fusion_q","fusion_k","fusion_v"] | None = None,
        attention_weight_control: bool = False,
        token_indices_clip: int|list[int] = 2,
        token_indices_t5: int|list[int] = 2,
        coefficient: int = 10,
        thresh: float = 0.1,
        save_attn: bool = False,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        split_idx = batch_size // 2 if batch_size > 1 else 0

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if inject:
            q_name = f"{timestep}_{second_order}_q"
            k_name = f"{timestep}_{second_order}_k"
            v_name = f"{timestep}_{second_order}_v"
            hard_mask_feature_name = f"{timestep}_{second_order}_hard_mask"
            soft_mask_feature_name = f"{timestep}_{second_order}_soft_mask"

            # ========== INVERSION ==========
            if inverse:
                if editing_strategy == "replace_qkv":
                    self.memory[q_name] = (query[split_idx:] * qkv_ratio[0]).cpu()
                    self.memory[k_name] = (key[split_idx:] * qkv_ratio[1]).cpu()
                    self.memory[v_name] = (value[split_idx:] * qkv_ratio[2]).cpu()
                else:
                    if editing_strategy and "q" in editing_strategy:
                        self.memory[q_name] = (query[split_idx:] * qkv_ratio[0]).cpu()
                    if editing_strategy and "k" in editing_strategy:
                        self.memory[k_name] = (key[split_idx:] * qkv_ratio[1]).cpu()
                    if editing_strategy and "v" in editing_strategy:
                        self.memory[v_name] = (value[split_idx:] * qkv_ratio[2]).cpu()

            # ========== EDITING ==========
            else:
                if editing_strategy == "replace_qkv":
                    # ---- Q ----
                    if q_name in self.memory:
                        if noise_injecting_strategy and "q" in noise_injecting_strategy:
                            query[split_idx:] += freq_com(
                                self.memory[q_name].to(query.device),
                                query[split_idx:],
                                alpha=noise_coeff[0],
                                eta=noise_coeff[1],
                                beta=noise_coeff[2],
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                            )
                        else:
                            query[split_idx:] = self.memory[q_name].to(query.device)

                    # ---- K ----
                    if k_name in self.memory:
                        if noise_injecting_strategy and "k" in noise_injecting_strategy:
                            key[split_idx:] += freq_com(
                                self.memory[k_name].to(key.device),
                                key[split_idx:],
                                alpha=noise_coeff[0],
                                eta=noise_coeff[1],
                                beta=noise_coeff[2],
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                            )
                        else:
                            key[split_idx:] = self.memory[k_name].to(key.device)

                    # ---- V ----
                    if v_name in self.memory:
                        if noise_injecting_strategy and "v" in noise_injecting_strategy:
                            value[split_idx:] += freq_com(
                                self.memory[v_name].to(value.device),
                                value[split_idx:],
                                alpha=noise_coeff[0],
                                eta=noise_coeff[1],
                                beta=noise_coeff[2],
                                d_s=scale_coeff,
                                d_t=scale_coeff,
                            )
                        else:
                            if not mask_fusion_strategy:
                                value[split_idx:] = self.memory[v_name].to(value.device)

                # ===== Individual Replace/Add =====
                else:
                    if editing_strategy == "replace_q" and q_name in self.memory:
                        query[split_idx:] = self.memory[q_name].to(query.device)
                    elif editing_strategy == "add_q" and q_name in self.memory:
                        query[split_idx:] += self.memory[q_name].to(query.device)

                    if editing_strategy == "replace_k" and k_name in self.memory:
                        key[split_idx:] = self.memory[k_name].to(key.device)
                    elif editing_strategy == "add_k" and k_name in self.memory:
                        key[split_idx:] += self.memory[k_name].to(key.device)

                    if editing_strategy == "replace_v" and v_name in self.memory:
                        value[split_idx:] = self.memory[v_name].to(value.device)
                    elif editing_strategy == "add_v" and v_name in self.memory:
                        value[split_idx:] += self.memory[v_name].to(value.device)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        if encoder_hidden_states is not None and not inverse and editing_strategy == "attn_guidance":
            # 进入自定义的自适应引导逻辑
            hidden_states = adaptive_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                txt_shape=encoder_hidden_states.shape[1],
                img_shape=residual.shape[1],
                cur_step=timestep,
                cur_block=self.block_idx,
                attn_guidance_start_block=attn_guidance_start_block,
            )

        elif attention_weight_control and inject:
            hidden_states, soft_mask, hard_mask = scaled_dot_product_attention_new(
                query=query,
                key=key,
                value=value,
                txt_length=self.txt_length,
                is_causal = False,
                token_indices_t5 = token_indices_t5,
                token_indices_clip= token_indices_clip,
                attn_mask=attention_mask,
                scale=None,
                coefficient = coefficient,
                thresh = 0.15,
                alpha=1.2,
                d_s = 0.3,
                weight=0.7,
                model_type = "sd35",
            )

            if inverse:
                self.memory[hard_mask_feature_name] = hard_mask.detach().cpu()
                self.memory[soft_mask_feature_name] = soft_mask.detach().cpu()
                if "fusion_q" in mask_fusion_strategy:
                    self.memory[q_name] = (query[split_idx:] * qkv_ratio[0]).cpu()
                if "fusion_k" in mask_fusion_strategy:
                    self.memory[k_name] = (key[split_idx:] * qkv_ratio[1]).cpu()
                if "fusion_v" in mask_fusion_strategy:
                    self.memory[v_name] = (value[split_idx:] * qkv_ratio[1]).cpu()
            
            elif not inverse and mask_fusion_strategy == "fusion_v":
                if v_name in self.memory and soft_mask_feature_name in self.memory:
                    v_old = self.memory[v_name].to(hidden_states.device)
                    mask_old = self.memory[soft_mask_feature_name].to(hidden_states.device)
                    
                    # mask_old 形状通常是 [H, W]，需要展平并对齐维度
                    # SD35 的 Value 序列是 [Batch, Heads, Img_len + Txt_len , Head_dim]
                    # mask 只作用于 Img_len 部分
                    m = mask_old.reshape(1, 1, -1, 1) # [1, 1, Img_len, 1]
                    # 分离文本和图像部分
                    v_img_current = value[:, :, :-self.txt_length, :]
                    v_img_old = v_old[:, :, :-self.txt_length, :]


                    eps = 1e-6
                    # 在序列维度 (dim=2) 上计算均值和标准差，保持维度以便广播
                    mu_current = v_img_current.mean(dim=2, keepdim=True)
                    std_current = v_img_current.std(dim=2, keepdim=True) + eps
                    
                    mu_old = v_img_old.mean(dim=2, keepdim=True)
                    std_old = v_img_old.std(dim=2, keepdim=True) + eps
                    
                    # 执行 AdaIN: 将当前特征的分布对齐到 old 特征的分布
                    v_img_adain = std_old * ((v_img_current - mu_current) / std_current) + mu_old
                    
                    # ==========================================
                    # 掩码融合
                    # ==========================================
                    # Mask区域用新生成的(AdaIN处理后)，非Mask区域用旧的
                    v_img_fused = v_img_adain * m + v_img_old * (1 - m)
                    value = torch.cat([v_img_fused, value[:, :, -self.txt_length:, :]], dim=2)
                    value = value.to(query.dtype)
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
        
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        if save_attn:
            attn_weight, _= scaled_dot_product_attention_simple(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
            self.attention_store(attn_weight)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states