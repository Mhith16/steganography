#!/usr/bin/env python3
"""
Enhanced steganography for X-ray images with patient data.
"""
import os
import sys
import glob
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from models.encoder import SteganoEncoder, ProgressiveSteganoEncoder
from models.decoder import SteganoDecoder, ProgressiveSteganoDecoder
from models.critic import Critic
from utils.dataset import (
    XrayDataset, PatientDataset, get_default_transforms,
    create_train_val_split
)
from utils.text_processor import (
    prepare_patient_data_for_encoding,
    text_to_binary_data,
    binary_data_to_text
)
from utils.metrics import calculate_psnr, calculate_ssim
from utils.visualization import visualize_result, visualize_difference
from trainer.basic_trainer import BasicTrainer
from trainer.adversarial_trainer import AdversarialTrainer


def train_model(args):
    """Train a new steganography model."""
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Get transforms for training and validation
    train_transform, val_transform = get_default_transforms()
    
    # Create full dataset first
    full_dataset = XrayDataset(args.xray_dir, transform=None, size=(args.img_size, args.img_size))
    
    # Split into train and validation sets
    train_dataset, val_dataset = create_train_val_split(full_dataset, val_ratio=args.val_ratio)
    
    # Set transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    
    # Initialize models
    if args.progressive:
        data_depths = [1, 2, 4]
        encoder = ProgressiveSteganoEncoder(data_depths, args.hidden_blocks, args.hidden_channels, args.img_size)
        decoder = ProgressiveSteganoDecoder(data_depths, args.hidden_blocks, args.hidden_channels)
        # Set initial level
        encoder.set_level(0)  # Start with data_depth=1
        decoder.set_level(0)
    else:
        encoder = SteganoEncoder(args.data_depth, args.hidden_blocks, args.hidden_channels, args.img_size)
        decoder = SteganoDecoder(args.data_depth, args.hidden_blocks, args.hidden_channels)
    
    # Choose training method
    if args.adversarial:
        # Create critic model
        critic = Critic(args.hidden_channels)
        
        # Initialize adversarial trainer
        trainer = AdversarialTrainer(
            encoder=encoder,
            decoder=decoder,
            critic=critic,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            output_dir=args.model_dir,
            data_depth=args.data_depth,
            image_size=args.img_size
        )
        
        # Train with adversarial loss
        history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            critic_iterations=args.critic_iterations
        )
    else:
        # Initialize basic trainer
        trainer = BasicTrainer(
            encoder=encoder,
            decoder=decoder,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            output_dir=args.model_dir,
            data_depth=args.data_depth,
            image_size=args.img_size
        )
        
        # Train with basic loss
        history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_interval=args.save_interval
        )
    
    # Plot training history
    plot_training_history(history, args.model_dir)
    
    print(f"Training completed. Models saved to {args.model_dir}")


def encode_message(args):
    """Encode a message into an X-ray image."""
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Load encoder model
    if args.progressive:
        encoder = ProgressiveSteganoEncoder([1, 2, 4], args.hidden_blocks, args.hidden_channels, args.img_size)
        encoder.load_state_dict(torch.load(os.path.join(args.model_dir, 'encoder_final.pt'), map_location='cpu'))
        encoder.set_level(args.level)
        data_depth = encoder.data_depths[args.level]
    else:
        encoder = SteganoEncoder(args.data_depth, args.hidden_blocks, args.hidden_channels, args.img_size)
        # Try to load the best model first, fall back to final model
        encoder_path = os.path.join(args.model_dir, 'encoder_best.pt')
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(args.model_dir, 'encoder_final.pt')
        
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
        data_depth = args.data_depth
    
    encoder.to(device)
    encoder.eval()
    
    # If no input image specified, select a random one
    if args.input is None:
        image_files = sorted(glob(os.path.join(args.xray_dir, "*.jpg")))
        if not image_files:
            raise ValueError(f"No JPEG images found in {args.xray_dir}")
        args.input = np.random.choice(image_files)
        print(f"No input image specified. Randomly selected: {args.input}")
    
    # Load and prepare the image
    image = Image.open(args.input).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prepare message for encoding
    if args.data_file:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            message = f.read()
    elif args.message:
        message = args.message
    else:
        # Try to find a corresponding text file for the image
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        possible_text_file = os.path.join(args.label_dir, f"{base_name}.txt")
        if os.path.exists(possible_text_file):
            with open(possible_text_file, 'r', encoding='utf-8') as f:
                message = f.read()
            print(f"Using corresponding text file: {possible_text_file}")
        else:
            message = f"Sample patient data for {base_name}"
            print(f"No message or text file specified. Using default message.")
    
    # Process message
    message = prepare_patient_data_for_encoding(message)
    print(f"Message to encode: {message}")
    print(f"Message length: {len(message)} characters")
    
    # Convert to binary data
    payload = text_to_binary_data(
        message, 
        data_depth,
        args.img_size,
        args.img_size,
        device
    )
    
    # Encode the message
    with torch.no_grad():
        stego = encoder(image_tensor, payload)
    
    # Save the output image
    stego_img = ((stego[0] + 1) / 2).cpu().permute(1, 2, 0).numpy()
    stego_img = (stego_img * 255).astype(np.uint8)
    stego_pil = Image.fromarray(stego_img)
    stego_pil.save(args.output)
    
    # Calculate metrics
    original_img = ((image_tensor[0] + 1) / 2).cpu().permute(1, 2, 0).numpy()
    
    psnr = calculate_psnr(original_img, stego_img)
    ssim = calculate_ssim(original_img, stego_img)
    
    print(f"Message encoded and saved to {args.output}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # Visualize and save comparison
    fig = visualize_result(
        original_img, 
        stego_img, 
        title="Steganography Result",
        save_path=os.path.join(os.path.dirname(args.output), "comparison.png")
    )
    
    # Visualize and save difference
    diff_fig = visualize_difference(
        original_img, 
        stego_img, 
        amplification=20,
        title="Difference (Amplified 20x)",
        save_path=os.path.join(os.path.dirname(args.output), "difference.png")
    )
    
    try:
        plt.show()
    except:
        pass


def decode_message(args):
    """Decode a message from an X-ray image."""
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Load decoder model
    if args.progressive:
        decoder = ProgressiveSteganoDecoder([1, 2, 4], args.hidden_blocks, args.hidden_channels)
        decoder.load_state_dict(torch.load(os.path.join(args.model_dir, 'decoder_final.pt'), map_location='cpu'))
        decoder.set_level(args.level)
    else:
        decoder = SteganoDecoder(args.data_depth, args.hidden_blocks, args.hidden_channels)
        # Try to load the best model first, fall back to final model
        decoder_path = os.path.join(args.model_dir, 'decoder_best.pt')
        if not os.path.exists(decoder_path):
            decoder_path = os.path.join(args.model_dir, 'decoder_final.pt')
        
        decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    
    decoder.to(device)
    decoder.eval()
    
    # Load and prepare the image
    image = Image.open(args.output).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Decode the message
    with torch.no_grad():
        decoded = decoder(image_tensor)
    
    # Convert decoded data to text
    message = binary_data_to_text(decoded)
    
    # Output the message
    print(f"Decoded message: {message}")
    
    # If original message was provided, compare
    if args.data_file:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            original = f.read()
        
        original = prepare_patient_data_for_encoding(original)
        
        # Check for match
        if message == original:
            print("✅ Success! The decoded message matches the original.")
        else:
            # Find the first difference
            for i, (orig_char, decoded_char) in enumerate(zip(original, message)):
                if orig_char != decoded_char:
                    print(f"First difference at position {i}:")
                    print(f"  Original: '{orig_char}' ({ord(orig_char)})")
                    print(f"  Decoded: '{decoded_char}' ({ord(decoded_char)})")
                    break
            
            # Calculate character accuracy
            min_len = min(len(original), len(message))
            matches = sum(1 for i in range(min_len) if original[i] == message[i])
            accuracy = matches / min_len
            
            print(f"⚠️ The decoded message differs from the original.")
            print(f"  Original length: {len(original)}")
            print(f"  Decoded length: {len(message)}")
            print(f"  Character accuracy: {accuracy:.2%}")
    
    return message


def plot_training_history(history, output_dir):
    """Plot training history and save figures."""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_encoder_loss'], label='Train')
    plt.plot(history['val_encoder_loss'], label='Validation')
    plt.title('Encoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_decoder_loss'], label='Train')
    plt.plot(history['val_decoder_loss'], label='Validation')
    plt.title('Decoder Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(2, 2, 3)
    plt.plot(history['val_psnr'])
    plt.title('PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('dB')
    
    plt.subplot(2, 2, 4)
    plt.plot(history['val_ssim'])
    plt.title('SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'))
    
    # Plot bit accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['val_bit_accuracy'])
    plt.title('Bit Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'bit_accuracy.png'))
    
    # If adversarial training was used, plot additional metrics
    if 'train_adversarial_loss' in history and 'train_critic_loss' in history:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_adversarial_loss'])
        plt.title('Adversarial Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_critic_loss'])
        plt.title('Critic Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'adversarial_metrics.png'))
    
    try:
        plt.close('all')
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Enhanced steganography for X-ray images")
    
    # Operation modes
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--encode', action='store_true', help='Encode a message')
    parser.add_argument('--decode', action='store_true', help='Decode a message')
    
    # General options
    parser.add_argument('--xray_dir', type=str, default='data/xrays',
                        help='Directory with X-ray images')
    parser.add_argument('--label_dir', type=str, default='data/labels',
                        help='Directory with label/text files')
    parser.add_argument('--model_dir', type=str, default='saved_models',
                        help='Directory to save/load models')
    parser.add_argument('--input', type=str, default=None,
                        help='Input image for encoding (if not specified, a random validation image will be used)')
    parser.add_argument('--output', type=str, default='stego_xray.png',
                        help='Output path for steganographic image')
    parser.add_argument('--message', type=str, default='Patient data: John Doe, 45, Male, Pneumonia',
                        help='Message to encode')
    parser.add_argument('--data_file', type=str, default=None,
                        help='File containing data to encode')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation split (default: 0.2)')
    
    # Model configuration
    parser.add_argument('--data_depth', type=int, default=1,
                        help='Data depth (bits per pixel)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--hidden_blocks', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--hidden_channels', type=int, default=32,
                        help='Number of hidden channels')
    parser.add_argument('--progressive', action='store_true',
                        help='Use progressive training')
    parser.add_argument('--level', type=int, default=0,
                        help='Level for progressive model (0=low capacity, 2=high capacity)')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Epoch interval for saving models')
    parser.add_argument('--adversarial', action='store_true',
                        help='Use adversarial training')
    parser.add_argument('--critic_iterations', type=int, default=5,
                        help='Number of critic iterations per generator iteration')
    parser.add_argument('--target_accuracy', type=float, default=0.98,
                        help='Target bit accuracy for early stopping (default: 0.98)')
    parser.add_argument('--min_psnr', type=float, default=35.0,
                        help='Minimum acceptable PSNR (default: 35.0)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
                        
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Modified and improved versions
    if args.train:
        print("Training new model...")
        train_model(args)
    
    if args.encode:
        print("Encoding message...")
        encode_message(args)
    
    if args.decode:
        print("Decoding message...")
        decode_message(args)
    
    if not (args.train or args.encode or args.decode):
        parser.print_help()
        print("\nNote: The current setup expects your data to be organized as follows:")
        print("  - X-ray images in the directory specified by --xray_dir")
        print("  - Text files (if any) in the directory specified by --label_dir")
        print("When training, a portion of the images will be automatically set aside for validation.")
        print("For encoding, you can specify an input image or let the program select one randomly.")
        print("If a text file with the same base name exists, it will be used as the message to encode.")


if __name__ == "__main__":
    main()