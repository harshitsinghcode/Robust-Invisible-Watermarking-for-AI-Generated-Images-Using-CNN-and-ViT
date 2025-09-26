from test import evaluate_watermarking

if __name__ == "__main__":
    image_dir = "data/flickr8k/images"
    mode = "cnn" 

    psnr_val, ssim_val = evaluate_watermarking(image_dir, mode)

    print(f"{'Metric':<16}Value")
    print(f"{'PSNR (dB)':<16}{psnr_val:.2f}")
    print(f"{'SSIM (0–1 scale)':<16}{ssim_val:.4f}")
