from typing import Dict, List, Sequence, Tuple
import torch


class AttentionStoreDiT:
    """Collects the cross‑attention tensors at each U‑Net layer.
    
    For SD35, stores:
        - clip_img_to_txt: CLIP image-to-text cross attention
        - clip_txt_to_img: CLIP text-to-image cross attention
        - t5_img_to_txt: T5 image-to-text cross attention
        - t5_txt_to_img: T5 text-to-image cross attention
    """

    @staticmethod
    def _blank() -> dict:
        return {
            "clip_img_to_txt": [],
            "clip_txt_to_img": [],
            "t5_img_to_txt": [],
            "t5_txt_to_img": []
        }

    def __init__(self, clip_txt_length=77, t5_txt_length=256):
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.attn_res: Tuple[int, int] = (32, 32)  # 注意力分数的分辨率 1024=32*32
        self.clip_txt_length = clip_txt_length
        self.t5_txt_length = t5_txt_length
        self.txt_length = clip_txt_length + t5_txt_length  # 77 + 256 = 333
        self.step_store: Dict[str, List[torch.Tensor]] = self._blank()
        self.attention_store: Dict[str, List[torch.Tensor]] = {}

    # Called by custom AttnProcessor -----------------------------------------
    def __call__(
        self, 
        attn: torch.Tensor, 
        txt_length: int = None,
        img_length: int = None,
        store_attn: bool = True
    ):
        """Store cross-attention matrices for SD35.
        
        Args:
            attn: Attention weight matrix [batch, heads, seq_len, seq_len]
                  SD35 joint sequence order is [img | txt], where txt = [CLIP | T5]
            txt_length: Length of text sequence (CLIP + T5)
            img_length: Length of image sequence
            store_attn: Whether to store this layer's attention
        """
        if txt_length is None:
            txt_length = self.txt_length
        if img_length is None:
            img_length = attn.shape[2] - txt_length
            
        if self.cur_att_layer >= 0 and store_attn:

            if attn.shape[0] > 1:
                attn_uncond, attn_cond = attn.chunk(2, dim=0)
            else:
                attn_cond = attn
            # SD3.5 joint sequence order inside attention is [img | txt].
            # img_length = seq_len - txt_length
            
            # img query, text key -> right top part: [batch, heads, img_len, txt_len]
            img_txt_cross = attn_cond[:, :, :-txt_length, -txt_length:]
            
            # text query, img key -> left bottom part: [batch, heads, txt_len, img_len]
            txt_img_cross = attn_cond[:, :, -txt_length:, :-txt_length]
            
            # Split into CLIP and T5 parts
            clip_len = min(self.clip_txt_length, txt_length)
            t5_start = clip_len
            t5_end = min(t5_start + self.t5_txt_length, txt_length)
            
            # CLIP cross-attention: shape [batch, heads, img_len, clip_len]
            self.step_store["clip_img_to_txt"].append(img_txt_cross[:, :, :, :clip_len])
            # CLIP text-to-img: shape [batch, heads, clip_len, img_len]
            self.step_store["clip_txt_to_img"].append(txt_img_cross[:, :, :clip_len, :].transpose(2, 3))
            
            # T5 cross-attention
            if t5_end > t5_start:
                # T5 img-to-txt: shape [batch, heads, img_len, t5_len]
                self.step_store["t5_img_to_txt"].append(img_txt_cross[:, :, :, t5_start:t5_end])
                # T5 text-to-img: shape [batch, heads, t5_len, img_len]
                self.step_store["t5_txt_to_img"].append(txt_img_cross[:, :, t5_start:t5_end, :].transpose(2, 3))
            
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:  # 走完了一次unet就强制刷新咯
            self.cur_att_layer = 0
            self.attention_store = self.step_store
            self.step_store = self._blank()

    # Queries -----------------------------------------------------------------
    def aggregate(self, what: Sequence[str]) -> torch.Tensor:
        maps = []
        for loc in what: #what in ["text"]
            for m in self.attention_store.get(loc, []):
                maps.append(
                    m.reshape(-1, self.attn_res[0], self.attn_res[1], m.shape[-1])
                )
        if not maps:
            raise ValueError("No attention maps collected; check attn_res.")
        maps = torch.cat(maps, 0)  # 有20个头 维度[20,32,32,77] 一共有60个block 所以是1200个头在平均，根据输入的平均值可以做分类平均，但是默认全都做
        return maps.sum(0) / maps.shape[0]


class AttentionStoreFlux:
    """Collects the cross‑attention tensors at each U‑Net layer.
    
    For Flux, stores:
        - img_to_txt: image-to-text cross attention (left bottom)
        - txt_to_img: text-to-image cross attention (right top)
    """

    @staticmethod
    def _blank() -> dict:
        return {
            "img_to_txt": [],
            "txt_to_img": []
        }

    def __init__(self, txt_length=512):
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.attn_res: Tuple[int, int] = (32, 32)  # 注意力分数的分辨率 1024=32*32
        self.txt_length = txt_length
        self.step_store: Dict[str, List[torch.Tensor]] = self._blank()
        self.attention_store: Dict[str, List[torch.Tensor]] = {}

    # Called by custom AttnProcessor -----------------------------------------
    def __call__(
        self, 
        attn: torch.Tensor, 
        txt_length: int = None,
        store_attn: bool = True
    ):
        """Store cross-attention matrices for Flux.
        
        Args:
            attn: Attention weight matrix [batch, heads, seq_len, seq_len]
                  Flux joint sequence order is [txt | img]
            txt_length: Length of text sequence
            store_attn: Whether to store this layer's attention
        """
        if txt_length is None:
            txt_length = self.txt_length
            
        if self.cur_att_layer >= 0 and store_attn:
            # Flux joint sequence order: [txt | img]
            # img_length = seq_len - txt_length
            
            # img query, text key -> left bottom part: [batch, heads, img_len, txt_len]
            img_to_txt = attn[:, :, txt_length:, :txt_length]
            
            # text query, img key -> right top part: [batch, heads, txt_len, img_len]
            txt_to_img = attn[:, :, :txt_length, txt_length:]
            
            self.step_store["img_to_txt"].append(img_to_txt)
            self.step_store["txt_to_img"].append(txt_to_img.transpose(2, 3))
            
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:  # 走完了一次unet就强制刷新咯
            self.cur_att_layer = 0
            self.attention_store = self.step_store
            self.step_store = self._blank()

    # Queries -----------------------------------------------------------------
    def aggregate(self, what: Sequence[str]) -> torch.Tensor:
        maps = []
        for loc in what: #what in ["text"]
            for m in self.attention_store.get(loc, []):
                maps.append(
                    m.reshape(-1, self.attn_res[0], self.attn_res[1], m.shape[-1])
                )
        if not maps:
            raise ValueError("No attention maps collected; check attn_res.")
        maps = torch.cat(maps, 0)  # 有20个头 维度[20,32,32,77] 一共有60个block 所以是1200个头在平均，根据输入的平均值可以做分类平均，但是默认全都做
        return maps.sum(0) / maps.shape[0]