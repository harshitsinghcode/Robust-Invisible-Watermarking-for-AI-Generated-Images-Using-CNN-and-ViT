# **Robust Invisible Watermarking for AI-Generated Images Using CNN \& ViT**

## 📖 **Overview**

This project presents a state-of-the-art **deep learning-based watermarking system** that embeds and extracts invisible watermarks in AI-generated images. By combining **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)**, it ensures watermark robustness against tampering, compression, and other distortions while maintaining imperceptibility.

Developed under the prestigious **Samsung PRISM Program**, this project represents an innovative approach to intellectual property protection for AI-generated content.

---
![image](https://github.ecodesamsung.com/harshitkumar-singh2022/Robust-Invisible-Watermarking-for-AI-Generated-Images-Using-CNN-ViT/assets/33519/e50cf406-5ba5-403a-885e-16e06a013872)
---
## ✨ **Key Features**

- **CNN-Based Watermarking**: A CNN encoder-decoder model for embedding and extracting invisible watermarks.
- **ViT-Based Tampering Detection**: A Vision Transformer (ViT) to analyze image features and detect tampering with high accuracy.
- **Robustness Against Distortions**: Handles noise, compression, blurring, and more without compromising watermark integrity.
- **Evaluation Metrics**: Uses quantitative measures like **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)** to assess performance.
- **Custom Dataset Support**: Easily integrates with datasets like Flickr8k or other image repositories.

---



## 🛠️ **Installation**

To set up the project:

1. Clone the repository:

```bash
git clone https://github.com/harshitsinghcode/Robust-Invisible-Watermarking-for-AI-Generated-Images-Using-CNN-and-ViT.git
cd Robust-Invisible-Watermarking-for-AI-Generated-Images-Using-CNN-and-ViT
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
    - Place your image dataset in the `data/flickr8k/images/` directory.
4. Ensure GPU support (optional):
    - Install CUDA if you plan to train models on a GPU.

---
![WhatsApp Image 2025-05-30 at 13 28 50_99d448ca](https://github.ecodesamsung.com/SRIB-PRISM/Robust-Invisible-Watermarking-for-AI-Generated-Images-Using-CNN-ViT/assets/33519/23f8cf69-9fd2-4574-88ee-4d61ef95244b)


## 🚀 **How to Use**

### Training

1. Train both CNN and ViT models by running:

```bash
python train.py
```

2. Modify hyperparameters such as learning rate, batch size, or number of epochs directly in `train.py` for better control over training.


## Requirements (If not using Docker)
- Python 3.10
- torch, torchvision, Pillow, opencv-python, etc.

## Docker Instructions
1. Build:
   docker build -t prism-watermark:v1 .
2. Run:
   docker run --rm prism-watermark:v1

## Dataset
`data/flickr8k/images/`.

---

### Testing and Evaluation

1. Evaluate the watermarking system by running:

```bash
python test.py
```

2. The script outputs key metrics like PSNR and SSIM for performance evaluation.

---

## 📊 **Results**

The following table summarizes the performance of our watermarking system based on testing:


| Metric | Value |
| :-- | :-- |
| **PSNR (dB)** | 34.21 |
| **SSIM (0–1 scale)** | 0.8951 |

These results demonstrate that our system achieves robust watermarking while maintaining high image quality.

---

## 📂 **Project Structure**

Here’s an organized view of the project directory:

```plaintext
Robust-Invisible-Watermarking/
├── models/
│   ├── cnn_model.py          # CNN-based watermarking model
│   ├── vit_model.py          # ViT-based feature extraction model
│   └── watermarking_model.py # Combined CNN + ViT model
├── utils/
│   ├── data_loader.py        # Data loading utilities
│   └── flickr8k_dataset.py   # Dataset processing scripts
├── data/
│   └── flickr8k/images/      # Image dataset directory
├── train.py                  # Training script for CNN &amp; ViT models
├── test.py                   # Testing &amp; evaluation script
└── README.md                 # Project documentation (this file)
```

---


## 🏆 **Acknowledgments**

We extend our gratitude to Samsung PRISM Program for providing us with this opportunity to explore advanced concepts in deep learning and computer vision.

Special thanks to our mentors for their invaluable guidance throughout this journey.

