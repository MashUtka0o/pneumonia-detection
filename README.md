# pneumonia-detection with X-rays
Deep Learning with Prof. Wahl

# Task

Transfer learning with MobileNet or ResNet to distinguish between normal lungs
and pneumonia.  

# Submission

**Model Card Submission: Friday, June 6, 2025, 11:59 PM CEST**

Model Card (max. 3 pages) with: objective, data, model, training, evaluation, limitations 
[Model Card](https://docs.google.com/document/d/1begDbBezvR3kWWZCJEFZHgpqFSggf6WnROFrXBSmaO0/edit?usp=sharing)   
[Model Card Template](https://huggingface.co/docs/hub/en/model-cards)  
Code (e.g., via GitHub or Colab)  

5min group presentation  23.06.- 07.07.

# Data
[Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

# Contributor
Christopher Reinard Kohar 22210239   
Harsh Amit Doshi 22203459   
Maria Matyukhina 22210692   
Marlis Wagner 22301867  

# Model Card

## Model Details

* Developed in 2015 by researchers at Microsoft Research
* Convolutional Neural Network
* Introduces residual connections to mitigate vanishing gradients in deep networks

## Intended Use

* Typically uses ResNet-50 or ResNet-101 for medical imaging tasks
* Assist radiologists in detecting pneumonia from chest X-ray images
* Intended to be used by healthcare professionals, researchers, and medical imaging developers

## Factors

* Relevant factors for patients: age groups, genders, and ethnicities
* Potential relevant factors from technical site: on X-ray machine settings, resolution, and patient positioning
* Designed for diagnostic support, not standalone diagnosis

## Metrics

Tuned to balance false positives

## Training Data

* Chest X-RAY Images (Pneumonia) [1]
* Data is splitted in normal (no pneumonia), pneumonia virus, and pneumonia bacteria images
* Size has been adjusted from original dataset for better performance

## Evaluation Data

* Chest X-RAY Images (Pneumonia) [1]
* Data is splitted in normal (no pneumonia) and pneumonia images
* Size has been adjusted from original dataset for better performance

## Ethical Considerations

The diagnoses for the images were graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Caveats and Recommendations

## Quantitative Analyses
