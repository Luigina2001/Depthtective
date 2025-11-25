<div align="center">
    <img width="350" alt="Depthtective Logo" src="https://github.com/user-attachments/assets/12e098f2-8ce0-47d9-8b22-4ba55be05c91" />
    <p>Official implementation of the method described in: <br>
    <em>“Depthtective: A Depth-Aware Framework for Spatio-Temporal Deepfake Detection”</em></p>

![](https://img.shields.io/badge/Status-Under_Review-yellow)
![](https://img.shields.io/badge/Submission-ICPR_2026-blue)
</div>

<br>

> [!WARNING]
> **Code Release Status:**
> This paper has been submitted to the **28th International Conference on Pattern Recognition (ICPR 2026)**. The full source code and pre-trained models will be released publicly **after the first review notification**.
>
> The documentation below serves as a preview of the framework's usage.
> 
---

## Overview

Depthtective is a data-efficient framework for the detection of manipulated facial videos based on the analysis of spatio-temporal inconsistencies in estimated depth. The method draws on the observation that modern deepfake generation techniques, while photorealistic, exhibit subtle violations of geometric coherence that become evident when comparing depth estimates between temporally adjacent frames.

Instead of relying on heavy temporal models such as 3D CNNs or Transformers, Depthtective focuses on the temporal residuals between two consecutive frames. The absolute differences in both RGB and depth domains are fused into a compact four-channel tensor that exposes motion-related inconsistencies and geometric distortions introduced by manipulation. This representation enables accurate video-level classification without the need for extended temporal sequences.

---

## Method

### Residual Representation  
For each pair of aligned frames, a depth map is estimated through MiDaS (DPT-Large).  
The temporal variation in appearance and geometry is quantified through the absolute inter-frame residuals in RGB and depth. Their fusion forms a four-channel tensor (RGBD residual) that serves as the sole input to the classifier.

### Classification Pipeline  
The residual tensor is processed by an adapted Xception or ResNet50 architecture supporting four-channel input while retaining ImageNet pretraining. The network is fine-tuned to discriminate between authentic and manipulated videos using a standard binary classification objective.  
Despite its simplicity, this formulation captures the core temporal inconsistencies typical of deepfake generation.

### Contrastive Variant  
A second formulation adopts a contrastive representation learning approach.  
The CNN is trained using a Triplet Loss to produce embeddings in which real and fake samples occupy well-separated regions of the latent space. A lightweight MLP head is then trained on top of the frozen encoder.  
This strategy enhances separability especially for challenging manipulations such as NeuralTextures, where the artifacts are subtle and stochastic.

<p align="center"><img width="600" alt="pipelineContrastiveLearning_en" src="https://github.com/user-attachments/assets/bfed617a-9963-4ff9-9729-34d4c96dc054" /></p>

---

## Performance Highlights

The effectiveness of Depthtective has been validated through experiments on the **FaceForensics++ (FF++)** benchmark (C23 compression) and the **Celeb-DF (v2)** dataset. We report the performance of our method implemented with standard CNN backbones (Xception, ResNet50) and the Contrastive Learning variant. The radar charts below illustrate the Accuracy, F1-Score, and Area Under the Curve (AUC) across all manipulation types.

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <img src="https://github.com/user-attachments/assets/97458109-c12d-44bd-ac22-f4e2fa0b7137" width="100%" />
        <br><b>Xception</b>
      </td>
      <td align="center" width="33%">
        <img src="https://github.com/user-attachments/assets/a2f096bf-a781-4916-9a4c-9b78078e0702" alt="ResNet Performance" width="100%" />
        <br><b>ResNet50</b>
      </td>
      <td align="center" width="33%">
        <img src="https://github.com/user-attachments/assets/be047832-8b26-4a48-80c2-1ae9a4d1bfb6" alt="Contrastive Learning Performance" width="100%" />
        <br><b>Contrastive Learning</b>
      </td>
    </tr>
  </table>
</div>

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

---

## Citation

If you find this framework useful for your research, please consider citing our work (BibTeX will be updated upon publication):

```
@article{depthtective2025,
  title={Depthtective: A Depth-Aware Framework for Spatio-Temporal Deepfake Detection},
  author={Bisogni, Carmen and Costante, Luigina and Nappi, Michele and Pero, Chiara},
  journal={Submitted to ICPR},
  year={2026}
}
```
