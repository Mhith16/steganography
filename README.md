# Enhanced Steganography for X-Ray Images

An advanced steganography system designed to hide patient data within X-ray images using deep learning.

## Features

- Hide text data in medical X-ray images with minimal visual impact
- Enhanced neural network architecture inspired by SteganoGAN
- Multiple capacity levels with progressive training
- Adversarial training for higher quality results
- Consistent metrics calculation for PSNR and SSIM
- Comprehensive visualization tools
- Support for X-ray images paired with patient text data

## Project Structure

```
steganography/
│
├── data/                  # Data directory
│   ├── xrays/             # X-ray images (.jpg files)
│   └── labels/            # Patient data text files (.txt files)
│
├── models/                # Model architectures
│   ├── encoder.py         # Enhanced encoder models
│   ├── decoder.py         # Enhanced decoder models
│   ├── critic.py          # Critic for adversarial training
│   └── layers.py          # Custom layer implementations
│
├── utils/                 # Utility functions
│   ├── dataset.py         # Dataset handling
│   ├── metrics.py         # Metrics calculation
│   ├── text_processor.py  # Text processing tools
│   └── visualization.py   # Visualization tools
│
├── trainer/               # Training implementations
│   ├── basic_trainer.py   # Basic training loop
│   └── adversarial_trainer.py # Adversarial training
│
├── saved_models/          # Saved model weights
│
├── requirements.txt       # Project dependencies
├── main.py                # Main entry point
└── README.md              # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xray-steganography.git
cd xray-steganography

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a New Model

```bash
# Basic training
python main.py --train --xray_dir data/xrays --label_dir data/labels --model_dir saved_models/basic --epochs 30 --batch_size 16

# Adversarial training
python main.py --train --adversarial --xray_dir data/xrays --label_dir data/labels --model_dir saved_models/adversarial --epochs 30 --batch_size 16

# Progressive training (multi-capacity)
python main.py --train --progressive --xray_dir data/xrays --label_dir data/labels --model_dir saved_models/progressive --epochs 30 --batch_size 16
```

### Encoding a Message

```bash
# Encode a simple message
python main.py --encode --model_dir saved_models/basic --input data/xrays/sample.jpg --output stego_output.png --message "Patient: John Doe, 45, Male, Pneumonia"

# Encode from a text file
python main.py --encode --model_dir saved_models/basic --input data/xrays/sample.jpg --output stego_output.png --data_file data/labels/patient123.txt

# Random image selection (will use matching text file if it exists)
python main.py --encode --model_dir saved_models/basic --xray_dir data/xrays --label_dir data/labels --output stego_output.png

# Encode with high capacity (for progressive models)
python main.py --encode --progressive --level 2 --model_dir saved_models/progressive --input data/xrays/sample.jpg --output stego_output.png --data_file data/labels/patient123.txt
```

### Decoding a Message

```bash
# Decode a message
python main.py --decode --model_dir saved_models/basic --output stego_output.png

# Decode with high capacity (for progressive models)
python main.py --decode --progressive --level 2 --model_dir saved_models/progressive --output stego_output.png

# Decode and compare with original
python main.py --decode --model_dir saved_models/basic --output stego_output.png --data_file data/labels/patient123.txt
```

### Combined Operations

```bash
# Train, encode, and decode in one command
python main.py --train --encode --decode --model_dir saved_models/new_model --xray_dir data/xrays --label_dir data/labels --output stego_output.png --message "Test message" --epochs 10
```

## Model Configuration Options

- `--data_depth`: Number of bits per pixel (default: 1)
- `--img_size`: Image size (default: 256)
- `--hidden_blocks`: Number of residual blocks (default: 4)
- `--hidden_channels`: Number of hidden channels (default: 32)

## Training Options

- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 16)
- `--save_interval`: Epoch interval for saving models (default: 5)
- `--adversarial`: Use adversarial training
- `--critic_iterations`: Number of critic iterations per generator iteration (default: 5)
- `--progressive`: Use progressive training
- `--level`: Capacity level for progressive models (0=low, 2=high) (default: 0)
- `--cuda`: Use GPU acceleration if available

## Metrics

The system calculates and reports the following metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality (higher is better)
- **SSIM (Structural Similarity Index)**: Measures perceptual image quality (higher is better)
- **Bit Accuracy**: Measures message recovery accuracy (higher is better)

## License

This project is licensed under the MIT License - see the LICENSE file for details.