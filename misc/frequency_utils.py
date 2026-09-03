import torch
import torch.fft as fft

def get_freq_filter(shape, device, d_s, d_t):
    """
    独立的高斯核生成函数 (替代原代码中的 for 循环)
    """
    T, H, W = shape
    
    # 使用 torch.arange 配合网格生成，保持与原代码 (2*t/T-1) 完全一致的数学逻辑
    # 相比 for 循环，这里利用 GPU 并行计算，速度极快
    t = torch.arange(T, device=device, dtype=torch.float32)
    h = torch.arange(H, device=device, dtype=torch.float32)
    w = torch.arange(W, device=device, dtype=torch.float32)
    
    grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing='ij')
    
    # 原始公式复刻：
    # d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
    d_square = (((d_s/d_t) * (2*grid_t/T - 1))**2 + 
                (2*grid_h/H - 1)**2 + 
                (2*grid_w/W - 1)**2)
                
    LPS = torch.exp(-1 / (2 * d_s**2) * d_square)
    return LPS

def get_freq_filter_2d(shape, device, d_s):
    """
    生成 2D 高斯频率滤波器
    shape: (H, W)
    d_s: 标准差，控制低频范围 (越小越模糊)
    """
    H, W = shape
    h = torch.arange(H, device=device, dtype=torch.float32)
    w = torch.arange(W, device=device, dtype=torch.float32)
    
    grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')
    
    # 归一化坐标到 [-1, 1]
    d_square = (2*grid_h/H - 1)**2 + (2*grid_w/W - 1)**2
    
    # 低通滤波器 (LPS)
    LPS = torch.exp(-1 / (2 * d_s**2) * d_square)
    return LPS

@torch.no_grad()
def mask_freq_augment_fusion(
    mask_1, mask_2, 
    alpha=0.5, d_s=0.3, 
    fusion_mode='weighted_sum',
    weight=0.8
):
    """
    mask_1, mask_2: [H, W] 的二维张量 (0-1)
    alpha: 增强因子 (控制低频成分权重)
    d_s: 频率切分阈值
    fusion_mode: 'sum' (相加再截断), 'max' (取最强区域), 'blend' (平均)
    """
    device = mask_1.device
    H, W = mask_1.shape
    
    LPF = get_freq_filter_2d((H, W), device, d_s)
    HPF = 1 - LPF

    def augment_mask(m):
        # 转到频域
        m_f = fft.fft2(m.to(torch.float32))
        m_f = fft.fftshift(m_f)
        m_aug_f = 1.0 * (m_f * LPF) + alpha * (m_f * HPF)
        
        # 转回空间域
        m_aug = fft.ifftshift(m_aug_f)
        m_aug = fft.ifft2(m_aug).real
        return m_aug

    m1_aug = augment_mask(mask_1)
    m2_aug = augment_mask(mask_2)

    if fusion_mode == 'sum':
        combined = m1_aug + m2_aug
    elif fusion_mode == 'max':
        combined = torch.max(m1_aug, m2_aug)
    elif fusion_mode == 'blend':
        combined = (m1_aug + m2_aug) / 2
    elif fusion_mode == "weighted_sum":
        combined = weight * m1_aug + (1-weight) * m2_aug
    else:
        combined = m1_aug # 默认
        
    if combined.max() > 0:
        combined = combined / combined.max() # 线性拉伸到 0-1
    
    return combined

@torch.no_grad()
def mask_freq_augment_fusion_v4(
    mask_1, mask_2,
    alpha=1.2, d_s=0.3,
    fusion_mode='weighted_sum', # 空间域先决定共识
    weight=0.7
):
    device = mask_1.device
    H, W = mask_1.shape

    if fusion_mode == 'max':
        combined = torch.max(mask_1, mask_2)
    elif fusion_mode == 'weighted_sum':
        combined = weight * mask_1 + (1 - weight) * mask_2
    else:
        combined = (mask_1 + mask_2) / 2

    LPF = get_freq_filter_2d((H, W), device, d_s)
    f_combined = torch.fft.fftshift(torch.fft.fft2(combined.to(torch.float32)))
    
    # 提取低频，通过减法获得绝对对齐的高频
    m_low = torch.fft.ifft2(torch.fft.ifftshift(f_combined * LPF)).real
    m_high = combined - m_low 

    m_aug = m_low + alpha * m_high
    
    return torch.clamp(m_aug, 0.0, 1.0)

@torch.no_grad()
def mask_freq_augment_fusion_v3(
    mask_1, mask_2, 
    alpha=0.5, d_s=0.3, 
    fusion_mode_low='weighted_sum',
    fusion_mode_high='max',
    weight_low=0.5,
    weight_high=0.5
):
    device = mask_1.device
    H, W = mask_1.shape

    LPF = get_freq_filter_2d((H, W), device, d_s)
    HPF = 1 - LPF

    def get_components(m):
        f = torch.fft.fftshift(torch.fft.fft2(m.to(torch.float32)))
        m_low = torch.fft.ifft2(torch.fft.ifftshift(f * LPF)).real
        m_high = torch.fft.ifft2(torch.fft.ifftshift(f * HPF)).real
        return m_low, m_high

    def fuse_op(m1, m2, mode, w):
        if mode == 'sum':
            return m1 + m2
        elif mode == 'max':
            return torch.max(m1, m2)
        elif mode == 'weighted_sum':
            return w * m1 + (1 - w) * m2
        elif mode == 'blend':
            return (m1 + m2) / 2
        else:
            return m1

    low_1, high_1 = get_components(mask_1)
    low_2, high_2 = get_components(mask_2)

    fused_low = fuse_op(low_1, low_2, fusion_mode_low, weight_low)
    
    fused_high = fuse_op(high_1, high_2, fusion_mode_high, weight_high)

    m_aug = fused_low + alpha * fused_high

    m_aug = torch.clamp(m_aug, 0.0, 1.0)
    
    return m_aug

@torch.no_grad()
def mask_freq_augment_fusion_v2(
    mask_1, mask_2, 
    alpha=0.5, d_s=0.3, 
    fusion_mode='weighted_sum',
    weight=0.8
):
    device = mask_1.device
    H, W = mask_1.shape

    if fusion_mode == 'sum':
        combined = mask_1 + mask_2
    elif fusion_mode == 'max':
        combined = torch.max(mask_1, mask_2)
    elif fusion_mode == 'blend':
        combined = (mask_1 + mask_2) / 2
    elif fusion_mode == "weighted_sum":
        combined = weight * mask_1 + (1 - weight) * mask_2
    else:
        combined = mask_1

    LPF = get_freq_filter_2d((H, W), device, d_s)
    HPF = 1 - LPF

    m_f = torch.fft.fft2(combined.to(torch.float32))
    m_f = torch.fft.fftshift(m_f)
    
    # 核心逻辑：对整体进行频率重分配
    # 这里也可以理解为：保留原始边缘(HPF)，并根据alpha调整整体的丰满度(LPF)
    m_aug_f = 1.0 * (m_f * LPF) + alpha * (m_f * HPF)
    
    m_aug = torch.fft.ifftshift(m_aug_f)
    m_aug = torch.fft.ifft2(m_aug).real

    m_aug = torch.clamp(m_aug, 0.0, 1.0)
    
    return m_aug


@torch.no_grad()
def freq_com(high, low, alpha=0.2, beta=None, eta=None, offset=0.8, d_s=0.3, d_t=0.3, mode=0):
    """ Frequency manipulation for latent space. """
    if alpha < 1e-5:
        return torch.randn_like(high).to(high.device)
    if alpha > 0.9999:
        return high
    
    try:
        B, C, H, W = high.shape
        Three = False
    except:
        high = high[None, ...]
        low = low[None, ...]
        B, C, H, W = high.shape
        Three = True
    
    dtype = high.dtype
    high = high.view(C,B,H,W).to(torch.float32)
    low = low.view(C,B,H,W).to(torch.float32)
    f_shape = high.shape 
    f_dtype = high.dtype
    
    # 这里 B 对应原代码中的 T (Time/Batch维度)
    # 动态生成 LPS，不再依赖全局变量
    LPS = get_freq_filter((B, H, W), high.device, d_s, d_t)
    
    LPF = LPS.to(high.device) # 保持变量名 LPF

    
    # High FFT
    HPF = 1 - LPF
    high_freq = fft.fftn(high, dim=(-3, -2, -1))
    high_freq = fft.fftshift(high_freq, dim=(-3, -2, -1))

    
    # Low FFT
    low_freq = fft.fftn(low, dim=(-3, -2, -1))
    low_freq = fft.fftshift(low_freq, dim=(-3, -2, -1))


    if mode==0:
        high_freq_high = high_freq * HPF
        low_freq_low = low_freq * LPF

        # Combine
        x_freq_sum = high_freq_high + alpha * low_freq_low 
        _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
        x_sum = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
    
    elif mode==1:
        high_freq_high = high_freq * HPF
        high_freq_low = high_freq * LPF
        low_freq_low = low_freq * LPF
        low_freq_high = low_freq * HPF
        
        x_freq_sum = high_freq_high + alpha * low_freq_low
        _x_freq_sum = fft.ifftshift(x_freq_sum, dim=(-3, -2, -1))
        x_sum1 = fft.ifftn(_x_freq_sum, dim=(-3, -2, -1)).real
        
        x_freq_sum2 = high_freq_low + alpha * low_freq_high
        _x_freq_sum2 = fft.ifftshift(x_freq_sum2, dim=(-3, -2, -1))
        x_sum2 = fft.ifftn(_x_freq_sum2, dim=(-3, -2, -1)).real
    
        x_sum = 0.5 * (x_sum1 + x_sum2)
    
    elif mode==2:
        # SD3 Mode 2 逻辑
        high_freq_high = high_freq * HPF
        high_freq_low = high_freq * LPF
        low_freq_low = low_freq * LPF
        low_freq_high = low_freq * HPF
        
        x_freq_sum = high_freq_high + alpha * low_freq_low
        x_freq_sum2 = high_freq_low + alpha * low_freq_high
        x_freq_sum_total = offset * x_freq_sum + (1-offset) * x_freq_sum2
        
        _x_freq_sum_total = fft.ifftshift(x_freq_sum_total, dim=(-3, -2, -1))
        x_sum = fft.ifftn(_x_freq_sum_total, dim=(-3, -2, -1)).real
    
    x_sum = x_sum.to(f_dtype)
    x_sum = x_sum.view(B,C,H,W)
    
    if Three:
        x_sum = x_sum[0]
    
    # 兼容原始逻辑中的 beta/eta 设置
    if beta is None:
        beta = 1 - alpha if mode == 0 else 0.0
    if eta is None:
        eta = alpha if mode == 0 else 1.0
        
    return (eta*x_sum + beta*torch.randn_like(x_sum).to(x_sum.device)).to(dtype)
