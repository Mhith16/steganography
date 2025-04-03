"""
Utilities for processing text data for steganography.
"""
import torch
import numpy as np
import re


def extract_patient_info(text):
    """
    Extract structured information from patient data text.
    
    Args:
        text: Raw patient data text
        
    Returns:
        Dictionary with structured patient information
    """
    info = {}
    
    # Extract name (assuming format like "Name: John Doe")
    name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', text)
    if name_match:
        info['name'] = name_match.group(1).strip()
    
    # Extract age
    age_match = re.search(r'Age:\s*(\d+)', text)
    if age_match:
        info['age'] = int(age_match.group(1))
    
    # Extract gender
    gender_match = re.search(r'Gender:\s*(\w+)', text)
    if gender_match:
        info['gender'] = gender_match.group(1).strip()
    
    # Extract diagnosis
    diagnosis_match = re.search(r'Diagnosis:\s*(.+?)(?:\n|$)', text)
    if diagnosis_match:
        info['diagnosis'] = diagnosis_match.group(1).strip()
    
    # Extract patient ID
    id_match = re.search(r'ID:\s*(.+?)(?:\n|$)', text)
    if id_match:
        info['id'] = id_match.group(1).strip()
    
    return info


def prepare_patient_data_for_encoding(text, max_length=200):
    """
    Prepare patient text data for encoding, limiting length and normalizing.
    
    Args:
        text: Raw patient data text
        max_length: Maximum text length to encode
        
    Returns:
        Processed text ready for encoding
    """
    # Truncate to max_length
    text = text[:max_length]
    
    # Normalize text (remove extra whitespace, etc.)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure text ends with a special marker for better detection
    if not text.endswith('###'):
        text += '###'
    
    return text


def text_to_binary_data(text, data_depth, height, width):
    """
    Convert text to binary data suitable for encoding.
    
    Args:
        text: Text to encode
        data_depth: Number of binary channels
        height, width: Dimensions of the data tensor
        
    Returns:
        Binary tensor [1, data_depth, height, width]
    """
    # Convert text to binary string
    binary = ''.join(format(ord(char), '08b') for char in text)
    
    # Create tensor
    data = np.zeros((1, data_depth, height, width), dtype=np.float32)
    
    # Fill tensor with binary data
    idx = 0
    for d in range(data_depth):
        for h in range(height):
            for w in range(width):
                if idx < len(binary):
                    data[0, d, h, w] = float(binary[idx])
                    idx = (idx + 1) % len(binary)  # Loop if we run out of data
    
    return torch.from_numpy(data)


def binary_data_to_text(binary_data, max_length=1000):
    """
    Convert binary data back to text.
    
    Args:
        binary_data: Binary tensor from decoder [1, data_depth, height, width]
        max_length: Maximum number of characters to decode
        
    Returns:
        Decoded text
    """
    # Convert tensor to numpy if needed
    if isinstance(binary_data, torch.Tensor):
        # Convert logits to binary
        binary = (binary_data.detach().cpu() >= 0).numpy().flatten()
    else:
        binary = binary_data.flatten()
    
    # Convert binary to text
    text = ""
    for i in range(0, min(len(binary), max_length * 8), 8):
        if i + 8 <= len(binary):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(binary[i + j])
            
            # Only add printable ASCII
            if 32 <= byte <= 126 or byte in [10, 13]:  # Include newlines
                text += chr(byte)
    
    # Try to find the end marker
    end_marker = '###'
    if end_marker in text:
        text = text[:text.index(end_marker) + len(end_marker)]
    
    # Look for repeating patterns (common in steganography decoding)
    if len(text) > 10:
        for pattern_length in range(5, min(len(text) // 2, 50)):
            pattern = text[:pattern_length]
            if text[:pattern_length * 2] == pattern * 2:
                return pattern
    
    return text


def add_error_correction(text):
    """
    Add simple error correction by repeating the text.
    
    Args:
        text: Original text
        
    Returns:
        Text with error correction
    """
    # Simple repetition code (3x repetition)
    return text * 3


def decode_with_error_correction(text):
    """
    Decode text with error correction by finding the most common substring.
    
    Args:
        text: Text with potential errors
        
    Returns:
        Corrected text
    """
    # For simple repetition code, find the first third
    text_length = len(text)
    if text_length < 3:
        return text
    
    third_length = text_length // 3
    first_part = text[:third_length]
    
    # Check if it's repeated
    if text.startswith(first_part * 2):
        return first_part
    
    return text