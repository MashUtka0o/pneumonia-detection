# Model Card: ResNet-18 Pneumonia Classifier (Mobile)

## Model Details

- **Model Date:** June 2025
- **Model Version:** 1.0
- **Model Type:** Deep Convolutional Neural Network (ResNet-18, transfer learning)
- **Training Algorithms & Parameters:**  
  - Optimizer: Adam (lr=1e-4, weight_decay=1e-4)  
  - Loss: CrossEntropyLoss  
  - Scheduler: StepLR (step_size=7, gamma=0.1)  
  - Epochs: 20  
  - Batch size: 32  
  - Data Augmentation: RandomResizedCrop, RandomHorizontalFlip (train only)
- **Fairness Constraints:** None explicitly applied
- **Features:** 3-class classification: NORMAL, PNEUMONIA_BACTERIA, PNEUMONIA_VIRUS
- **Paper/Resource:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Citation:**  
  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. arXiv:1512.03385

---

## Intended Use

- **Primary Intended Uses:**  
  - Automated classification of chest X-ray images into normal, bacterial pneumonia, or viral pneumonia for clinical decision support or research.
- **Primary Intended Users:**  
  - Medical professionals, researchers, and developers in healthcare AI.
- **Out-of-Scope Use Cases:**  
  - Direct clinical diagnosis without human oversight.
  - Use on non-chest X-ray images or populations not represented in the training data.

---

## Factors

- **Relevant Factors:**  
  - Patient demographics (age, sex, ethnicity)
  - X-ray image quality and acquisition device
- **Evaluation Factors:**  
  - Accuracy per class
  - Performance across different demographic groups (if available)

---

## Metrics

- **Model Performance Measures:**  
  - Overall accuracy, per-class accuracy
  - Confusion matrix
- **Decision Thresholds:**  
  - Standard argmax for 3-class classification
- **Variation Approaches:**  
  - Data augmentation during training

---
## Training Data

- **Source:**  
  - Same as evaluation data (Kaggle Chest X-Ray Pneumonia Dataset)
- **Distribution:**  
  - Balanced across three classes after preprocessing
- **Preprocessing:**  
  - Data augmentation (random crop, flip) for training set
  - Normalization to ImageNet statistics

---

## Evaluation Data

- **Datasets:**  
  - [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Motivation:**  
  - Widely used benchmark for pneumonia detection
- **Preprocessing:**  
  - Images combined and relabeled into three classes
  - Split into train/val/test (70/15/15) with stratified randomization

---

## Quantitative Analyses

## Quantitative Analyses

- **Binary classification (Normal/Pneumonia)**
    - **Pneumonia Detection Accuracy:** 96.94%  
    - **Precision:** 0.9920  
    - **Recall (Sensitivity):** 0.9658  
    - **Specificity:** 0.9790  
    - **F1 Score:** 0.9787  
    - **Matthews Correlation Coefficient (MCC):** 0.9251  

- **Intersectional Results:**  
    - Performance across demographic subgroups not reported (data unavailable).

---

## Ethical Considerations

- Model is not a substitute for professional medical advice or diagnosis.
- Potential for bias if training data is not representative of target population.
- Risk of misclassification; human oversight is required.

---

## Caveats and Recommendations

- Evaluate model performance on local data before deployment.
- Monitor for distribution shift and retrain as needed.
- Do not use as the sole basis for clinical decisions.
- Further validation is recommended for populations not present in the training data.