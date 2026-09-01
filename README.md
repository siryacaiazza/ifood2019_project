# Supervised and Self-Supervised Approaches for Food Image Classification

Comparative study of **supervised learning** and **self-supervised learning (SSL)** approaches for fine-grained food image classification using Convolutional Neural Networks (CNNs).

The project investigates whether pretraining an encoder through a **denoising autoencoder** can improve downstream classification performance when compared with a fully supervised model, while keeping both models below **5 million trainable parameters**.

## 📌 Project Overview

Food image classification is a challenging computer vision task due to:

- **251 fine-grained food classes**
- High **inter-class similarity**, where different foods can look very similar
- High **intra-class variability**, caused by differences in preparation and presentation
- Limited availability of large-scale annotated datasets

Two approaches are compared:

1. **Supervised CNN** — the model is trained directly on labeled images.
2. **Self-Supervised CNN** — the encoder is first pretrained using a denoising autoencoder and is subsequently used for supervised classification.

Both approaches use the **same encoder and classifier architecture**, allowing a more direct comparison between the learning strategies.

## 📊 Dataset

The project uses the **iFood 2019** dataset, containing approximately **158,000 food images across 251 classes**.

The original dataset consists of:

| Split | Images | Usage in this project |
|---|---:|---|
| Labeled training | ~118,000 | Reduced and split into train/validation |
| Labeled validation | ~11,000 | Test set |
| Unlabeled | ~29,000 | Not used |

Due to computational limitations, the training dataset was reduced by sampling a maximum of **150 images per class**.

The resulting dataset was split using an **80/20 stratified split**:

- **30,027 training images**
- **7,507 validation images**
- **11,994 test images**

## 🧹 Preprocessing

The following preprocessing steps were applied:

- Resize images to **224 × 224 pixels**
- Normalize pixel values to **[0, 1]**
- Convert class labels to **one-hot encoded vectors**
- Use a **batch size of 64**
- For SSL, add **Gaussian noise** to the input images

The noisy images are used as input to the denoising autoencoder, while the original clean images are used as reconstruction targets.

## 🧠 Model Architectures

### Supervised Model

The supervised CNN consists of an encoder followed by a classifier.

The encoder contains four convolutional blocks, each composed of:

```text
Conv2D
↓
BatchNormalization
↓
ReLU
↓
MaxPooling2D
```

The encoder is followed by:

```text
GlobalAveragePooling2D
↓
Dense
↓
Dropout (50%)
↓
Dense classifier
```

The model contains approximately **2.34 million trainable parameters**, remaining well below the 5-million-parameter constraint.

### Self-Supervised Model

The SSL model uses the **same encoder and classifier** as the supervised model, with an additional decoder for the pretext task.

The training pipeline consists of two stages:

```text
Noisy image
     ↓
   Encoder
     ↓
   Decoder
     ↓
Reconstructed image
```

### Stage 1 — Denoising Autoencoder

The encoder-decoder network is trained to reconstruct clean images from noisy inputs.

**Loss:** Mean Squared Error (MSE)

This pretext task encourages the encoder to learn general visual representations such as shapes and textures without using class labels.

### Stage 2 — Downstream Classification

After pretraining:

```text
Pretrained Encoder
        ↓
   Classifier Head
        ↓
  Food Class Prediction
```

The pretrained encoder is connected to the classifier and trained using the labeled dataset.

The SSL architecture, including encoder and classifier, contains approximately **3.89 million trainable parameters**.

## ⚙️ Training

### Supervised Training

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Initial learning rate | `1e-3` |
| LR scheduling | ExponentialDecay |
| Early stopping | Patience = 8 |
| Maximum epochs | 50 |
| Training time | Just under 2 hours |

The supervised model achieved a maximum **validation Top-5 accuracy of 47%** during training.

### SSL — Autoencoder Pretraining

| Parameter | Value |
|---|---|
| Initial learning rate | `5e-4` |
| Loss | MSE |
| Early stopping | Patience = 8 |
| Maximum epochs | 15 |
| Training time | A little over 2.5 hours |

The autoencoder achieved a best validation accuracy of **84%** and a lowest validation loss of **0.070** during pretraining.

### SSL — Downstream Classification

The downstream classifier used the same hyperparameters as the supervised model.

- **29 epochs**
- Early stopping triggered
- Training time: just under **1 hour 10 minutes**
- Validation Top-5 accuracy: **39%**

## 📈 Final Results

Both models were evaluated on the same test set of approximately 11,000 labeled images.

| Metric | Supervised | Self-Supervised |
|---|---:|---:|
| Accuracy | **0.2489** | 0.1867 |
| Top-5 Accuracy | **0.5163** | 0.4277 |
| Weighted F1 | **0.2318** | 0.1641 |
| Weighted Recall | **0.2489** | 0.1867 |
| Weighted Precision | **0.2444** | 0.1748 |
| Macro F1 | **0.2211** | 0.1552 |
| Macro Recall | **0.2388** | 0.1782 |
| Macro Precision | **0.2338** | 0.1661 |

### 🔎 Main Findings

The **supervised model outperformed the SSL model across all evaluated metrics**.

The observed differences include approximately:

- **6 percentage points** improvement on standard and weighted scores
- **9 percentage points** improvement in Top-5 accuracy

However, the gap between the models is not excessively large. This suggests that the denoising autoencoder was able to learn useful visual representations despite not using class labels during pretraining.

Inference speed was also very similar:

- Supervised: **97.99 s**
- SSL: **97.21 s**

## 🧪 Limitations

The main limitation of the project was the availability of computational resources.

Training was performed using **Google Colab with a Tesla T4 GPU**, which limited:

- Dataset size
- Number of training epochs
- Hyperparameter exploration
- Architecture experimentation
- Exhaustive model optimization

The SSL model was also trained for fewer epochs during pretraining than the supervised model. Therefore, the performance difference may partly be explained by **undertraining rather than an inherent limitation of the SSL approach**.

## 🚀 Future Work

Several improvements could be investigated:

### Longer SSL pretraining

The denoising autoencoder was trained for only 15 epochs. Increasing the number of epochs could allow the encoder to learn more robust representations.

### Learning-rate optimization

The validation curves showed instability during training. A different learning-rate schedule or decay configuration could potentially improve training stability.

### Smaller SSL encoder

A possible optimization would be to use only the first three convolutional blocks during the autoencoder pretraining stage:

```text
Input
  ↓
Block 1
  ↓
Block 2
  ↓
Block 3
  ↓
Lightweight Decoder
```

The remaining encoder block and classifier could then be added during downstream training.

This could reduce pretraining time and computational requirements while encouraging the encoder to focus on low-level features such as **edges and textures**.

## 🗂️ Suggested Repository Structure

```text
.
├── README.md
├── notebooks/
│   └── food_classification.ipynb
├── models/
│   ├── supervised/
│   └── self_supervised/
├── figures/
│   ├── class_distribution.png
│   ├── noisy_images.png
│   ├── supervised_training.png
│   ├── autoencoder_training.png
│   ├── ssl_training.png
│   └── confusion_matrices.png
├── requirements.txt
└── LICENSE
```

> The exact repository structure depends on the files included in the implementation.

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- Convolutional Neural Networks
- Supervised Learning
- Self-Supervised Learning
- Denoising Autoencoders
- Google Colab
- Tesla T4 GPU

## 📚 References

- **iFood 2019 Dataset** — [Karan Sikka, GitHub](https://github.com/karansikka1/iFood_2019)
- **Top-1 and Top-5 Accuracy** — Anh T. Dang (2021)
- **Machine Learning Evaluation Metrics** — GeeksforGeeks

## 👤 Author

**Sirya Caiazza**  
MSc Artificial Intelligence for Science and Technology  
Università degli Studi di Milano-Bicocca

---

### 📌 Summary

This project compares supervised and self-supervised approaches for food image classification under a strict **5M-parameter constraint**.

Although the **supervised model achieved the best overall performance**, the SSL approach demonstrated that a denoising autoencoder can learn useful representations without class labels. The results therefore highlight both the current effectiveness of supervised learning and the potential of self-supervised learning in scenarios where annotated data is limited.