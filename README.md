# Pix2Pix Implementation for Semantic-to-Real Image Translation

This repository contains an implementation of the Pix2Pix model using PyTorch. The Pix2Pix model leverages Conditional Generative Adversarial Networks (CGANs) to learn a mapping from an input image (e.g., a semantic segmentation map) to a corresponding output image (e.g., a photorealistic cityscape). This approach was first introduced in the paper:

> *Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros (2017). [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). CVPR.*

## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
  - [Conditional GANs](#conditional-gans)
  - [U-Net Generator](#u-net-generator)
  - [PatchGAN Discriminator](#patchgan-discriminator)
  - [Loss Functions](#loss-functions)
- [Dataset](#dataset)
  - [Download and Preparation](#download-and-preparation)
  - [Structure](#structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Hyperparameters and Settings](#hyperparameters-and-settings)
- [Results](#results)
- [Loss Plots](#loss-plots)
- [References](#references)

## Introduction

This project implements Pix2Pix to translate segmented images, as prepared by self-driving cars (e.g., Cityscapes label maps), into realistic images. Such a model can aid in tasks like data augmentation, improved visualization, and sim-to-real adaptation in autonomous driving pipelines.

## Theoretical Background

### Conditional GANs
Unlike traditional GANs that generate data from random noise, Conditional GANs (CGANs) incorporate conditional inputs. The model takes a given input (such as a segmentation map) and tries to produce an output that looks realistic and aligns with the provided condition. The discriminator thus evaluates pairs of (input, output) to determine if they are "real" or "fake."

### U-Net Generator
The generator architecture is based on a U-Net: an encoder-decoder network with skip connections. 
- **Encoder:** Extracts features and reduces spatial resolution.  
- **Decoder:** Reconstructs the image from latent features back to the original spatial size.  
- **Skip Connections:** Preserve fine-grained details from early layers, improving the quality and sharpness of generated images.

### PatchGAN Discriminator
Instead of evaluating the entire image holistically, the PatchGAN discriminator classifies each N×N patch of the image as real or fake. This helps the model focus on local texture details, leading to sharper and more realistic outputs.

### Loss Functions
- **Adversarial Loss (GAN Loss):** Encourages the generator to produce outputs indistinguishable from real images.
- **L1 Loss:** Ensures that the generated image is closely aligned with the target image at a pixel level, improving structural fidelity.

## Dataset

We use a Cityscapes-based Pix2Pix dataset, which contains pairs of:
- **Input (Segmented) Images:** Semantic label maps.
- **Target (Real) Images:** Corresponding realistic cityscape photographs.

### Download and Preparation
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/cityscapes-pix2pix-dataset/data?select=val). Extract it into a directory like:
```
dataset/
  train/
    *_input.jpg
    *_target.jpg
  val/
    *_input.jpg
    *_target.jpg
```

### Structure
- **`train/`**: Training pairs of images (segmented and real).
- **`val/`**: Validation pairs of images.

## Installation & Requirements

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pix2pix-implementation.git
   cd pix2pix-implementation
   ```

2. Install dependencies (Python 3.7+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
   **Key Dependencies:**
   - PyTorch
   - Torchvision
   - PIL (Pillow)
   - Matplotlib
   - NumPy

3. Ensure you have GPU support for training, as it will be significantly faster.

## Usage

### Training
1. Adjust hyperparameters and paths in `train.py` as needed.
2. Run the training:
   ```bash
   python train.py --data_dir ./dataset --epochs 50 --batch_size 4 --lr 2e-4
   ```
   The training script will periodically display generated samples and save model checkpoints.

### Inference
1. After training, use the trained generator to translate new segmented images:
   ```bash
   python inference.py --input_dir ./test_inputs --output_dir ./outputs --model_path ./checkpoints/generator.pth
   ```
   This will generate real-like images corresponding to your segmented inputs.

## Hyperparameters and Settings

- **Learning Rate:** `2e-4`
- **Batch Size:** `4`
- **Epochs:** `50`
- **Lambda_L1 (L1 Loss Weight):** `100`
- **Optimizer:** Adam (β1=0.5, β2=0.999)

These values follow recommendations from the Pix2Pix paper and are known to produce stable training dynamics and realistic outputs.

## Results

During training, after a number of steps, generated images are displayed alongside the input segmented map and the real target image. Over time, the generated outputs should gain detail and more closely resemble the target distributions.

You can expect results where:
- Early epochs: Blurry and less detailed outputs.
- Later epochs: Increasingly realistic images with sharper boundaries and textures.

## Loss Plots

After training completes, the loss functions for both the Generator and Discriminator can be plotted using:
```bash
python plot_loss.py --log_file ./training_log.json
```
You should see the Discriminator loss stabilizing and the Generator loss converging.

## References

- **Pix2Pix Paper:** [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Isola et al.

- **Related Works:**
  - [CycleGAN: Unpaired Image-to-Image Translation](https://arxiv.org/abs/1703.10593)
  - [GAN Zoo (PyTorch)](https://github.com/hindupuravinash/the-gan-zoo)

If you find this repository helpful or use it in your research, consider citing the original Pix2Pix paper.

---

**Enjoy experimenting with Pix2Pix!**
