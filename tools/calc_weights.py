import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def calculate_weights(mask_dir, num_classes, ignore_index=255, method='log_smooth'):
    print(f"正在扫描目录: {mask_dir}")
    mask_paths = list(Path(mask_dir).rglob('*_labelTrainIds.png'))
    
    if not mask_paths:
        raise ValueError("未找到任何掩码图片，请检查路径！")

    # 初始化每个类别的像素计数器
    total_pixels_per_class = np.zeros(num_classes, dtype=np.int64)
    
    print("开始统计像素...")
    for path in tqdm(mask_paths):
        # 读取图片并转为 numpy 数组
        mask = np.array(Image.open(path))
        
        # 过滤掉 ignore_index
        valid_mask = mask[mask != ignore_index]
        
        # 统计该图中的像素，累加到全局计数器
        counts = np.bincount(valid_mask, minlength=num_classes)
        total_pixels_per_class += counts

    # 计算每个类别的频率 p_c
    total_valid_pixels = np.sum(total_pixels_per_class)
    p_c = total_pixels_per_class / total_valid_pixels
    
    print("\n[像素频率统计]:")
    for i, p in enumerate(p_c):
        print(f"类别 {i:2d}: {p*100:.4f}%")

    weights = np.zeros(num_classes)
    
    # -----------------------
    # 计算权重
    # -----------------------
    if method == 'median_freq':
        # 策略 A: 中值频率平衡
        median_freq = np.median(p_c)
        # 避免除以 0，加一个极小值 eps
        weights = median_freq / (p_c + 1e-8)
        
    elif method == 'log_smooth':
        # 策略 B: 对数平滑 (ENet 论文常数 c=1.02)
        c = 1.02
        weights = 1 / np.log(c + p_c)
    
    # ==========================================
    # 非常重要的一步：权重归一化！
    # 必须让权重的平均值接近 1，否则会大幅改变全局的 Learning Rate
    # ==========================================
    weights = weights / np.mean(weights)
    
    # 将结果转换为可以复制到 Config 中的格式
    weights_list = [round(float(w), 4) for w in weights]
    
    print(f"\n[{method} 最终计算出的 class_weight]:")
    print(weights_list)
    
    return weights_list

if __name__ == '__main__':
    # ================= 你的配置 =================
    MASK_DIRECTORY = '/cache/gtFine/train' # 替换为你的【训练集】Mask文件夹路径
    NUM_CLASSES = 19 # 你的类别数量 (不含 ignore_index)
    IGNORE_INDEX = 255
    # ============================================
    
    # 推荐使用 log_smooth
    final_weights = calculate_weights(
        mask_dir=MASK_DIRECTORY, 
        num_classes=NUM_CLASSES, 
        ignore_index=IGNORE_INDEX, 
        method='log_smooth' 
    )