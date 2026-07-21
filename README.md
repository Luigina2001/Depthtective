<div align="center">
    <img width="350" alt="Depthtective Logo" src="https://github.com/user-attachments/assets/12e098f2-8ce0-47d9-8b22-4ba55be05c91" />
    <p>Official implementation of the method described in: <br>
    <em>“Depthtective: A Depth-Aware Framework for Spatio-Temporal Deepfake Detection”</em></p>

![](https://img.shields.io/badge/Status-Under_Review-yellow)
</div>

<br>

> [!WARNING]
> **Code Release Status:**
> This paper is currently **under review**. The full source code and pre-trained models will be released publicly **after the first review notification**.
>
> The documentation below serves as a preview of the framework's usage.
> 
---

## Overview

Depthtective is a data-efficient framework for the detection of manipulated facial videos based on the analysis of spatio-temporal inconsistencies in estimated depth. As the deepfake detection community races toward data-hungry architectures, Depthtective moves away from exhaustive temporal sequence processing. 

Instead of relying on heavy temporal models such as 3D CNNs or Transformers, the method exposes manipulations using only **two adjacent frames**, a minimal temporal unit abundantly available in any standard video stream. The absolute differences in both RGB and depth domains are fused into a four-channel tensor that exposes motion-related inconsistencies and geometric distortions introduced by manipulation. This representation enables highly accurate video-level classification across diverse datasets (FaceForensics++, Celeb-DF, and DFDC) without the need for extended temporal sequences.

---

## Method

### Residual Representation  
For each pair of consecutive frames, facial landmarks are extracted (MediaPipe FaceMesh) to geometrically align the faces. A depth map is then estimated for both frames using the strong zero-shot capabilities of MiDaS (DPT-Large).  
The temporal variation in appearance and geometry is quantified through the absolute inter-frame residuals in RGB and depth. Their fusion forms a compact **four-channel tensor (RGBD residual)** that serves as the sole input to the classifier.

### Classification Pipeline  
The 4C residual tensor is processed by an adapted Xception or ResNet50 architecture. To retain ImageNet pretraining, the weights corresponding to the RGB channels are preserved, whereas the weights for the depth channel are initialized as the average of the three RGB kernels. The network is fine-tuned to discriminate between authentic and manipulated videos. Despite its simplicity and low computational cost (~23M parameters, ~8.4 GFLOPs), this formulation successfully captures the core temporal inconsistencies typical of deepfakes.

### Contrastive Variant  
To handle high-quality manipulations and severe compression artifacts (e.g., DFDC, Celeb-DF, NeuralTextures), a second formulation adopts a contrastive representation learning approach.  
The CNN backbone is optimized using a Triplet Loss, encouraging compact intra-class clusters and maximizing inter-class separation in the latent space. A lightweight MLP head is then trained on top of the frozen encoder.

<p align="center"><img width="600" alt="pipelineContrastiveLearning_en" src="https://github.com/user-attachments/assets/bfed617a-9963-4ff9-9729-34d4c96dc054" /></p>

---

## Performance Highlights

Depthtective has been evaluated on three major benchmarks: **FaceForensics++ (FF++)** (C23 compression), **Celeb-DF (v2)**, and **Deepfake Detection Challenge (DFDC)**. 

### Quantitative Results
The proposed approach competes with much heavier spatio-temporal architectures:
*   **Celeb-DF (v2):** The contrastive learning variant achieves an **AUC of 97.26%** and an Accuracy of 92.31%, demonstrating remarkable generalization on high-quality, smoothed deepfakes.
*   **DFDC:** Under highly variable, in-the-wild conditions and heavy compression, the contrastive module provides a massive boost, reaching an **AUC of 98.35%** and an Accuracy of 97.03%.
*   **FaceForensics++:** The baseline Xception achieves ~96.88% accuracy on Deepfakes. For the notoriously subtle NeuralTextures manipulation, the contrastive variant pushes accuracy to 89.01%.

### Explainability
Grad-CAM analysis confirms that Depthtective bases its decisions on valid spatio-temporal anomalies rather than contextual biases. The network consistently concentrates its attention on internal facial features—particularly the highly dynamic **nose and mouth regions**, where generative models frequently struggle to preserve geometric consistency.

---

## Installation (Preview)

```bash
git clone https://github.com/Luigina2001/Depthtective.git
cd Depthtective
````

Using Conda:

```bash
conda env create -f environment.yml
conda activate Depthtective
```

Using pip:

```bash
pip install -r requirements.txt
```

---

## Usage (Preview)

Depthtective provides a unified script for classifying a video.
The script performs frame extraction, depth estimation, residual construction, and prediction.

```bash
python main.py \
    --video_path path/to/video.mp4 \
    --contrastive_encoder_path models/best_contrastive_model.pth \
    --classifier_head_path models/best_classifier_head.pth \
    --hidden_features 256
```

Example output:

```
Video: test_video.mp4
Prediction: Deepfake
Confidence: 98.45%
```
