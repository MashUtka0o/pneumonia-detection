# Model Card: Resnet18

## Model Overview
**ResNet-18** is a deep convolutional neural network architecture with 18 layers, introduced in the paper ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by He et al. It utilizes residual connections to ease the training of deep networks.

## Intended Use
- 3-class Pneumonia detection
    - Normal
    - Pneumonia Bacteria
    - Pneumonia Virus

## Architecture
- 18 layers with residual (skip) connections
- Basic building block: two 3x3 convolutional layers per block
- Fewer parameters compared to deeper ResNet variants

## Training Data
- Pretrained on ImageNet-1K
- Finetuned on [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


## Performance
- Accuracy on Pneumonia Detection:
- Accuracy on Class Detection: 

## Limitations
- May underperform on very complex tasks compared to deeper models
- Sensitive to data quality and preprocessing

## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
