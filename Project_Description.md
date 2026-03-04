# Project Description: CNN vs Vision Transformer vs Vision Mamba

## Course Project Overview

We will study three major visual model families in one unified course project: CNN, Vision Transformer, and Vision Mamba. The main goal of our project will be to compare their strengths and weaknesses in a fair setting, instead of only reporting one final accuracy number. Through this comparison, we will understand which architecture is more suitable under different practical requirements, such as performance, speed, memory usage, and robustness.

We will use CIFAR-100 as the common benchmark dataset and will select one representative model for each family: ResNet-18 for CNN, DeiT-Tiny for ViT, and Vim-Tiny for Mamba. To make the comparison reliable, we will use the same data preprocessing pipeline, similar training schedules, and consistent evaluation standards. We will evaluate each model with Top-1/Top-5 accuracy, FLOPs, inference latency, and peak memory usage. We will also include FGSM adversarial testing to measure robustness and will use Grad-CAM plus t-SNE visualization to inspect what each model learns.

In this project, we will build a complete workflow from data preparation to training, evaluation, and visualization. This will help us not only compare final results, but also analyze model behavior from different perspectives. For example, CNN will be expected to perform strongly on local feature extraction, ViT will be expected to capture global dependencies effectively, and Mamba will be expected to offer better efficiency when modeling long-range information. By putting these models into the same pipeline, we will provide a clearer and more objective comparison.

We will also develop a simple Gradio demo so users can upload an image and see predictions from all three models at the same time, together with side-by-side Grad-CAM heatmaps. This part will make our project easier to present and will help us explain model differences in an intuitive way during course demonstration.

As an innovation extension, we will explore a hybrid CNN-Mamba design. Our idea will be to combine CNN's strong local feature extraction ability with Mamba's efficient long-range sequence modeling. We will test several fusion strategies, such as parallel branches and staged feature fusion, to see whether the combined model can outperform single architectures in both accuracy and efficiency. We hope this "CNN + Mamba" direction will generate new insights and will potentially lead to better practical performance.
