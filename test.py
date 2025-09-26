import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def evaluate_watermarking(image_dir, mode):
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist.")
        return
    
    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the directory.")
        return
    
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        watermarked = apply_watermark(original, mode)
        
        if original is None or watermarked is None:
            print(f"Skipping {img_name}, unable to load image.")
            continue
        
        psnr_value = psnr(original, watermarked)
        ssim_value = ssim(original, watermarked)
        
        total_psnr += psnr_value
        total_ssim += ssim_value
        count += 1
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
    else:
        print("No valid images processed.")

def apply_watermark(image, mode):
    if mode == 'cnn':
        return cnn_watermark(image)
    elif mode == 'vit':
        return vit_watermark(image)
    else:
        print(f"Unknown mode: {mode}")
        return None

def cnn_watermark(image):
    return np.clip(image + np.random.normal(0, 5, image.shape), 0, 255).astype(np.uint8)

def vit_watermark(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Set image directory and mode
image_dir = r"E:\\digital_watermarking_project-20250325T155517Z-001\\digital_watermarking_project\\data\\flickr8k\\Images"
mode = "cnn"  # Change this to 'vit' or other modes if needed
evaluate_watermarking(image_dir, mode)