# YOLOv1 From Scratch

This project is an implementation of the YOLOv1 (You Only Look Once) object detection algorithm from scratch in Python. It is designed for educational purposes to help understand the inner workings of YOLOv1, including model architecture, loss calculation, and training pipeline.

By default, this implementation is designed to work with the Pascal VOC dataset for object detection tasks. You can adapt the code to other datasets with similar annotation formats.

## Project Structure

- `model.py` — Defines the YOLOv1 model architecture.
- `loss.py` — Contains the custom loss function for YOLOv1.
- `train.py` — Training script for the model.
- `dataset.py` — Handles dataset loading and preprocessing.
- `utils.py` — Utility functions used throughout the project.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- (Optional) Matplotlib for visualization

Install dependencies:
```bash
pip install torch numpy matplotlib
```

### Dataset
This project is set up to use the Pascal VOC dataset (VOC2007 or VOC2012) by default. Please download and extract the VOC dataset, and update the dataset path in `dataset.py` as needed.

### Usage
1. Prepare your VOC dataset and update `dataset.py` if your path or annotation format differs.
2. Train the model:
   ```bash
   python train.py
   ```
3. Checkpoints and logs will be saved as configured in the scripts.

## References
- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License
This project is for educational purposes only.
