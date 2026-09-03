import torch
from einops import rearrange
from torch import tensor

def get_centroid(attn, top_k: int = 10):
    """
    计算注意力图的重心坐标 (语义种子)

    参数:
        attn: 注意力图，支持两种格式：
            1. [H, W] - 二维 heatmap
            2. [batch, seq_len, dim] - 三维注意力图
        top_k: 仅在响应最高的 top_k 个空间位置上计算语义种子，
                提升对噪声注意力响应的鲁棒性。
    """
    # 处理二维输入 [H, W]
    if attn.ndim == 2:
        h, w = attn.shape
        attn_2d = attn
        hs = torch.arange(h).view(-1, 1).to(attn.device).float()
        ws = torch.arange(w).view(1, -1).to(attn.device).float()
        # 仅在 top_k 个最高响应位置上计算重心
        flat = attn_2d.reshape(-1)
        k = min(top_k, flat.numel())
        if k < flat.numel():
            topk_vals, topk_idx = torch.topk(flat, k)
            sel_h = topk_idx // w
            sel_w = topk_idx % w
            mask = torch.zeros_like(flat)
            mask[topk_idx] = topk_vals
            sel_attn = mask.view(h, w)
        else:
            sel_attn = attn_2d
            sel_h, sel_w = hs.squeeze(-1), ws.squeeze(-1)
        weighted_w = torch.sum(ws * sel_attn, dim=[0, 1])
        weighted_h = torch.sum(hs * sel_attn, dim=[0, 1])
        total = sel_attn.sum()
        return torch.stack([weighted_w, weighted_h]) / (total + 1e-8)

    # 处理三维输入 [batch, seq_len, dim]
    if not len(attn.shape) == 3:
        attn = attn[:, :, None]
    h = w = int(tensor(attn.shape[-2]).sqrt().item())
    attn = rearrange(attn.mean(0), '(h w) d -> h w d', h=h)
    # 取模长作为空间响应强度
    spatial = attn.norm(dim=-1)  # [H, W]
    flat = spatial.reshape(-1)
    k = min(top_k, flat.numel())
    topk_vals, topk_idx = torch.topk(flat, k)
    sel = torch.zeros_like(flat)
    sel[topk_idx] = topk_vals
    sel_attn = sel.view(h, w)
    hs = torch.arange(h).view(-1, 1).to(attn.device).float()
    ws = torch.arange(w).view(1, -1).to(attn.device).float()
    weighted_w = torch.sum(ws * sel_attn, dim=[0, 1])
    weighted_h = torch.sum(hs * sel_attn, dim=[0, 1])
    total = sel_attn.sum()
    return torch.stack([weighted_w, weighted_h]) / (total + 1e-8)


    