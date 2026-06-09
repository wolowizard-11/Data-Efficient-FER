# Data-Efficient Facial Micro-Expression Recognition (FER2013)

This repository contains a comparative study and implementation of state-of-the-art (SOTA) computer vision architectures designed to overcome severe data scarcity in facial emotion recognition. 

By leveraging targeted Vision Transformer (ViT) fine-tuning and probabilistic ensembling, this project achieves high-accuracy micro-expression classification, evaluated against the full ~28,700 image FER2013 dataset.

##  Key Results
* **Probabilistic Ensemble (SOTA):** Achieved a peak **67.48%** overall accuracy by fusing global ViT attention maps with local EfficientNet-B0 convolutions via discrete softmax probability averaging.
* **Targeted ViT-B/16:** Reached **66.69%** accuracy, outperforming standard transfer learning by selectively unfreezing deep attention blocks to capture localized facial geometry.
* **EfficientNet-B0 Baseline:** Established a **56.25%** accuracy baseline, demonstrating the limitations of strictly localized convolutions on micro-expressions.
* **Virtual Data Expansion:** Scaled a constrained 4,000-image training split into **80,000 unique spatial tensors** via on-the-fly probabilistic augmentations to bypass data scarcity.

## Ensemble Classification Report

The probabilistic ensemble demonstrated exceptional performance on dominant positive emotions while maintaining robust precision across complex minority classes.

| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Happy** | 0.85 | 0.87 | **0.86** | 7,215 |
| **Surprise** | 0.72 | 0.79 | **0.75** | 3,171 |
| **Neutral** | 0.60 | 0.71 | **0.65** | 4,965 |
| **Angry** | 0.62 | 0.58 | **0.60** | 3,995 |
| **Sad** | 0.56 | 0.59 | **0.57** | 4,830 |
| **Disgust** | 0.86 | 0.35 | **0.49** | 436 |
| **Fear** | 0.58 | 0.43 | **0.49** | 4,097 |
| *Macro Avg* | *0.68* | *0.62* | *0.63* | *28,709* |
| *Weighted Avg* | *0.67* | *0.67* | *0.67* | *28,709* |

## 📂 Repository Structure
The project is structured sequentially for easy reproducibility:

1. **`01_EDA_and_Augmentation`**: Exploratory data analysis, spatial `torchvision` augmentations, and inversion-based `WeightedRandomSampler` logic to neutralize extreme class imbalances.
2. **`02_CNN_Transfer_Learning_Base`**: Baseline performance metrics using standard VGG19 transfer learning paradigms.
3. **`03_Targeted_ViT_Finetuning`**: Implementation of selective layer unfreezing, Label Smoothing (0.1), and Cosine Annealing on a ViT-Base architecture.
4. **`04_EfficientNet_SOTA`**: A lightweight, highly optimized CNN branch mimicking modern HSEmotion pipelines.
5. **`05_Probabilistic_Ensemble`**: The final evaluation suite testing the ensemble architecture against raw, hierarchical Kaggle image directories.

## ⚙️ Setup & Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/fer-micro-expressions.git](https://github.com/yourusername/fer-micro-expressions.git)
cd fer-micro-expressions
pip install -r requirements.txt
