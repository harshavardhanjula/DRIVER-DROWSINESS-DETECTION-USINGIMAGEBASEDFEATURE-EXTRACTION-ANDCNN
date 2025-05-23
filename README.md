# Image Classification Project

This project implements deep learning models for image classification using both Inception V3 and ResNet architectures. It includes training, evaluation, and prediction capabilities for image classification tasks.

## Project Structure

```
.
├── Dataset/                  # Contains training and validation data
├── Inception_V3.ipynb        # Jupyter notebook for Inception V3 implementation
├── ResNet (2).ipynb          # Jupyter notebook for ResNet implementation
├── check.py                  # Verification/utility script
├── final.py                  # Main Python script for the application
├── inspection_v3.h5          # Pre-trained model weights
└── image.jpg                 # Sample image file
```

## Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Keras
- Jupyter Notebook (for running .ipynb files)
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd major-project
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
Run the Jupyter notebooks `Inception_V3.ipynb` or `ResNet (2).ipynb` to train the models.

### Prediction
Use the `final.py` script for making predictions:
```bash
python final.py --image path/to/image.jpg
```

## Models

1. **Inception V3**
   - Pre-trained on ImageNet
   - Fine-tuned on custom dataset
   - Weights saved in `inspection_v3.h5`

2. **ResNet**
   - Implementation of ResNet architecture
   - Customizable depth

## Dataset

The dataset should be organized in the following structure:
```
Dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── validation/
    ├── class1/
    ├── class2/
    └── ...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

