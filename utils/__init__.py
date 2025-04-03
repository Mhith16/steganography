"""
Utility functions for steganography.
"""
from .metrics import calculate_psnr, calculate_ssim, calculate_bit_accuracy
from .dataset import XrayDataset, PatientDataset, get_default_transforms
from .text_processor import (
    extract_patient_info, 
    prepare_patient_data_for_encoding,
    text_to_binary_data,
    binary_data_to_text,
    add_error_correction,
    decode_with_error_correction
)
from .visualization import (
    visualize_result,
    visualize_difference,
    visualize_binary_data,
    denormalize_image
)