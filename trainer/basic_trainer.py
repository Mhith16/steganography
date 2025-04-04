"""
Basic trainer for steganography models.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils.metrics import calculate_psnr, calculate_ssim, calculate_bit_accuracy
from utils.visualization import visualize_result, visualize_binary_data


class BasicTrainer:
    """
    Basic trainer for encoder-decoder steganography models.
    """
    def __init__(
        self, 
        encoder, 
        decoder, 
        train_dataset, 
        val_dataset, 
        device='cuda',
        output_dir='saved_models',
        data_depth=1,
        image_size=256,
        target_accuracy=0.98,
        min_psnr=35.0
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.output_dir = output_dir
        self.data_depth = data_depth
        self.image_size = image_size
        self.target_accuracy = target_accuracy
        self.min_psnr = min_psnr
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize losses and optimizers
        self.encoder_criterion = nn.MSELoss()
        self.decoder_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
        
        # Combined parameters
        self.params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = optim.Adam(self.params, lr=0.001)
        
        # Training history
        self.history = {
            'train_encoder_loss': [],
            'train_decoder_loss': [],
            'val_encoder_loss': [],
            'val_decoder_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_bit_accuracy': [],
        }
    
    def train(self, epochs=10, batch_size=8, save_interval=1):
        """
        Train the models.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: How often to save model checkpoints
            
        Returns:
            Training history
        """
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        print(f"Training on {len(self.train_dataset)} images")
        print(f"Validating on {len(self.val_dataset)} images")
        
        best_bit_accuracy = 0.0
        best_epoch = 0
        early_stop = False
        patience = 10  # Number of epochs to wait for improvement
        no_improve_count = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_encoder_loss, train_decoder_loss = self._train_epoch(train_loader)
            
            # Validation
            val_metrics = self._validate(val_loader)
            
            # Update history
            self.history['train_encoder_loss'].append(train_encoder_loss)
            self.history['train_decoder_loss'].append(train_decoder_loss)
            self.history['val_encoder_loss'].append(val_metrics['encoder_loss'])
            self.history['val_decoder_loss'].append(val_metrics['decoder_loss'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['val_bit_accuracy'].append(val_metrics['bit_accuracy'])
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {elapsed:.1f}s")
            print(f"  Train: Encoder Loss: {train_encoder_loss:.4f}, Decoder Loss: {train_decoder_loss:.4f}")
            print(f"  Val: Encoder Loss: {val_metrics['encoder_loss']:.4f}, Decoder Loss: {val_metrics['decoder_loss']:.4f}")
            print(f"  Val: PSNR: {val_metrics['psnr']:.2f} dB, SSIM: {val_metrics['ssim']:.4f}")
            print(f"  Val: Bit Accuracy: {val_metrics['bit_accuracy']:.4f}")
            
            # Check for best model
            current_bit_accuracy = val_metrics['bit_accuracy']
            if current_bit_accuracy > best_bit_accuracy and val_metrics['psnr'] >= self.min_psnr:
                best_bit_accuracy = current_bit_accuracy
                best_epoch = epoch
                print(f"  New best model! Bit Accuracy: {best_bit_accuracy:.4f}")
                
                # Save best model
                torch.save(self.encoder.state_dict(), os.path.join(self.output_dir, 'encoder_best.pt'))
                torch.save(self.decoder.state_dict(), os.path.join(self.output_dir, 'decoder_best.pt'))
                
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Save models at interval
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch)
            
            # Check for early stopping criteria
            if val_metrics['bit_accuracy'] >= self.target_accuracy and val_metrics['psnr'] >= self.min_psnr:
                print(f"\nEarly stopping: Target bit accuracy of {self.target_accuracy:.2f} reached!")
                early_stop = True
                break
            
            # Check for no improvement early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping: No improvement for {patience} epochs")
                early_stop = True
                break
        
        # Always save final models
        torch.save(self.encoder.state_dict(), os.path.join(self.output_dir, 'encoder_final.pt'))
        torch.save(self.decoder.state_dict(), os.path.join(self.output_dir, 'decoder_final.pt'))
        
        if early_stop:
            print(f"Training stopped early at epoch {epoch+1}.")
        else:
            print("Training completed through all epochs.")
            
        print(f"Best model from epoch {best_epoch+1} with bit accuracy {best_bit_accuracy:.4f}")
        print(f"Models saved to {self.output_dir}")
        
        return self.history
    
    def _train_epoch(self, train_loader):
        """Run one epoch of training."""
        self.encoder.train()
        self.decoder.train()
        
        total_encoder_loss = 0
        total_decoder_loss = 0
        
        for cover in tqdm(train_loader, desc="Training"):
            cover = cover.to(self.device)
            batch_size = cover.size(0)
            
            # Generate random binary data
            payload = self._generate_random_data(batch_size)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            stego = self.encoder(cover, payload)
            decoded = self.decoder(stego)
            
            # Calculate losses
            encoder_loss = self.encoder_criterion(stego, cover)
            decoder_loss = self.decoder_criterion(decoded, payload)
            
            # Combined loss (you can adjust weights if needed)
            loss = encoder_loss + 5.0 * decoder_loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_encoder_loss += encoder_loss.item()
            total_decoder_loss += decoder_loss.item()
        
        # Calculate average losses
        avg_encoder_loss = total_encoder_loss / len(train_loader)
        avg_decoder_loss = total_decoder_loss / len(train_loader)
        
        return avg_encoder_loss, avg_decoder_loss
    
    def _validate(self, val_loader):
        """Run validation."""
        self.encoder.eval()
        self.decoder.eval()
        
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_bit_accuracy = 0
        
        with torch.no_grad():
            for cover in tqdm(val_loader, desc="Validating"):
                cover = cover.to(self.device)
                batch_size = cover.size(0)
                
                # Generate random binary data
                payload = self._generate_random_data(batch_size)
                
                # Forward pass
                stego = self.encoder(cover, payload)
                decoded = self.decoder(stego)
                
                # Calculate losses
                encoder_loss = self.encoder_criterion(stego, cover)
                decoder_loss = self.decoder_criterion(decoded, payload)
                
                # Calculate metrics for the batch
                psnr = calculate_psnr(cover.cpu(), stego.cpu())
                ssim = calculate_ssim(cover.cpu(), stego.cpu())
                bit_accuracy = calculate_bit_accuracy(payload.cpu(), decoded.cpu())
                
                # Update totals
                total_encoder_loss += encoder_loss.item()
                total_decoder_loss += decoder_loss.item()
                total_psnr += psnr
                total_ssim += ssim
                total_bit_accuracy += bit_accuracy
                
                # Visualize the first batch
                if total_encoder_loss == encoder_loss.item():
                    # Save sample images
                    visualize_result(
                        cover[0].cpu(), 
                        stego[0].cpu(),
                        "Validation Sample",
                        save_path=os.path.join(self.output_dir, f"val_sample.png")
                    )
                    
                    # Save sample data visualization
                    visualize_binary_data(
                        payload[0].cpu(),
                        "Original Data",
                        save_path=os.path.join(self.output_dir, f"val_data_original.png")
                    )
                    visualize_binary_data(
                        decoded[0].cpu(),
                        "Decoded Data",
                        save_path=os.path.join(self.output_dir, f"val_data_decoded.png")
                    )
        
        # Calculate averages
        metrics = {
            'encoder_loss': total_encoder_loss / len(val_loader),
            'decoder_loss': total_decoder_loss / len(val_loader),
            'psnr': total_psnr / len(val_loader),
            'ssim': total_ssim / len(val_loader),
            'bit_accuracy': total_bit_accuracy / len(val_loader),
        }
        
        return metrics
    
    def _generate_random_data(self, batch_size):
        """Generate random binary data for training."""
        return torch.randint(
            0, 2, 
            (batch_size, self.data_depth, self.image_size, self.image_size), 
            device=self.device
        ).float()
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        torch.save(
            self.encoder.state_dict(), 
            os.path.join(self.output_dir, f'encoder_epoch_{epoch+1}.pt')
        )
        torch.save(
            self.decoder.state_dict(), 
            os.path.join(self.output_dir, f'decoder_epoch_{epoch+1}.pt')
        )