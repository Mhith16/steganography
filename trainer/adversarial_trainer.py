"""
Adversarial trainer for steganography models with critic.
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


class AdversarialTrainer:
    """
    Adversarial trainer for encoder-decoder-critic steganography models.
    """
    def __init__(
        self, 
        encoder, 
        decoder, 
        critic,
        train_dataset, 
        val_dataset, 
        device='cuda',
        output_dir='saved_models',
        data_depth=1,
        image_size=256,
        target_accuracy=0.98,
        min_psnr=35.0,
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.critic = critic.to(device)
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
        
        # Initialize losses
        self.reconstruction_criterion = nn.MSELoss()
        self.decoder_criterion = nn.BCEWithLogitsLoss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        
        # Separate optimizers
        self.encoder_decoder_optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=0.001, 
            betas=(0.5, 0.999)
        )
        self.critic_optimizer = optim.Adam(
            critic.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        # For scheduling learning rates
        self.encoder_decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_decoder_optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_encoder_loss': [],
            'train_decoder_loss': [],
            'train_adversarial_loss': [],
            'train_critic_loss': [],
            'val_encoder_loss': [],
            'val_decoder_loss': [],
            'val_psnr': [],
            'val_ssim': [],
            'val_bit_accuracy': [],
        }
        
        # Loss weights
        self.lambda_reconstruction = 1.0
        self.lambda_adversarial = 0.001
        self.lambda_decoder = 1.0
    
    def train(self, epochs=10, batch_size=8, save_interval=1, critic_iterations=5):
        """
        Train the models.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: How often to save model checkpoints
            critic_iterations: Number of critic training iterations per encoder step
            
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
        
        # Helper tensors for labels
        self.real_label = 1.0
        self.fake_label = 0.0
        
        print(f"Training on {len(self.train_dataset)} images")
        print(f"Validating on {len(self.val_dataset)} images")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self._train_epoch(train_loader, critic_iterations)
            
            # Validation
            val_metrics = self._validate(val_loader)
            
            # Update history
            self.history['train_encoder_loss'].append(train_metrics['encoder_loss'])
            self.history['train_decoder_loss'].append(train_metrics['decoder_loss'])
            self.history['train_adversarial_loss'].append(train_metrics['adversarial_loss'])
            self.history['train_critic_loss'].append(train_metrics['critic_loss'])
            self.history['val_encoder_loss'].append(val_metrics['encoder_loss'])
            self.history['val_decoder_loss'].append(val_metrics['decoder_loss'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_ssim'].append(val_metrics['ssim'])
            self.history['val_bit_accuracy'].append(val_metrics['bit_accuracy'])
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {elapsed:.1f}s")
            print(f"  Train: Encoder Loss: {train_metrics['encoder_loss']:.4f}, "
                  f"Decoder Loss: {train_metrics['decoder_loss']:.4f}")
            print(f"  Train: Adversarial Loss: {train_metrics['adversarial_loss']:.4f}, "
                  f"Critic Loss: {train_metrics['critic_loss']:.4f}")
            print(f"  Val: Encoder Loss: {val_metrics['encoder_loss']:.4f}, "
                  f"Decoder Loss: {val_metrics['decoder_loss']:.4f}")
            print(f"  Val: PSNR: {val_metrics['psnr']:.2f} dB, "
                  f"SSIM: {val_metrics['ssim']:.4f}")
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
                torch.save(self.critic.state_dict(), os.path.join(self.output_dir, 'critic_best.pt'))
                
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Save models
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                self._save_checkpoint(epoch)
            
            # Update learning rate schedulers
            self.encoder_decoder_scheduler.step(val_metrics['encoder_loss'] + val_metrics['decoder_loss'])
            self.critic_scheduler.step(train_metrics['critic_loss'])
            
            # Check for early stopping criteria
            if val_metrics['bit_accuracy'] >= self.target_accuracy and val_metrics['psnr'] >= self.min_psnr:
                print(f"\nEarly stopping: Target bit accuracy of {self.target_accuracy:.2f} reached!")
                early_stop = True
                break
            patience = 5
            # Check for no improvement early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping: No improvement for {patience} epochs")
                early_stop = True
                break
        
        # Save final models
        torch.save(self.encoder.state_dict(), os.path.join(self.output_dir, 'encoder_final.pt'))
        torch.save(self.decoder.state_dict(), os.path.join(self.output_dir, 'decoder_final.pt'))
        torch.save(self.critic.state_dict(), os.path.join(self.output_dir, 'critic_final.pt'))
        
        if early_stop:
            print(f"Training stopped early at epoch {epoch+1}.")
        else:
            print("Training completed through all epochs.")
            
        print(f"Best model from epoch {best_epoch+1} with bit accuracy {best_bit_accuracy:.4f}")
        print(f"Models saved to {self.output_dir}")
        
        return self.history
    
    def _train_epoch(self, train_loader, critic_iterations=5):
        """Run one epoch of adversarial training."""
        self.encoder.train()
        self.decoder.train()
        self.critic.train()
        
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_adversarial_loss = 0
        total_critic_loss = 0
        
        for cover in tqdm(train_loader, desc="Training"):
            cover = cover.to(self.device)
            batch_size = cover.size(0)
            
            # Generate random binary data
            payload = self._generate_random_data(batch_size)
            
            # ---------------------
            # Train Critic
            # ---------------------
            for _ in range(critic_iterations):
                self.critic_optimizer.zero_grad()
                
                # Generate stego images
                with torch.no_grad():
                    stego = self.encoder(cover, payload)
                
                # Real images -> critic should output 0 (real)
                pred_real = self.critic(cover)
                real_labels = torch.full((batch_size, 1), self.real_label, device=self.device)
                loss_real = self.adversarial_criterion(pred_real, real_labels)
                
                # Stego images -> critic should output 1 (fake)
                pred_fake = self.critic(stego.detach())
                fake_labels = torch.full((batch_size, 1), self.fake_label, device=self.device)
                loss_fake = self.adversarial_criterion(pred_fake, fake_labels)
                
                # Total critic loss
                critic_loss = (loss_real + loss_fake) * 0.5
                critic_loss.backward()
                self.critic_optimizer.step()
                
                total_critic_loss += critic_loss.item()
            
            # ---------------------
            # Train Encoder & Decoder
            # ---------------------
            self.encoder_decoder_optimizer.zero_grad()
            
            # Generate stego images
            stego = self.encoder(cover, payload)
            
            # Decode message
            decoded = self.decoder(stego)
            
            # Calculate losses
            # 1. Reconstruction loss (stego should look like cover)
            encoder_loss = self.reconstruction_criterion(stego, cover) * self.lambda_reconstruction
            
            # 2. Decoder loss (decoded should match payload)
            decoder_loss = self.decoder_criterion(decoded, payload) * self.lambda_decoder
            
            # 3. Adversarial loss (critic should think stego is real)
            pred_stego = self.critic(stego)
            # Use real labels because we want critic to think stego is real
            adv_loss = self.adversarial_criterion(pred_stego, real_labels) * self.lambda_adversarial
            
            # Combined loss
            combined_loss = encoder_loss + decoder_loss + adv_loss
            combined_loss.backward()
            self.encoder_decoder_optimizer.step()
            
            # Update running metrics
            total_encoder_loss += encoder_loss.item()
            total_decoder_loss += decoder_loss.item()
            total_adversarial_loss += adv_loss.item()
        
        # Calculate average metrics
        num_batches = len(train_loader)
        metrics = {
            'encoder_loss': total_encoder_loss / num_batches,
            'decoder_loss': total_decoder_loss / num_batches,
            'adversarial_loss': total_adversarial_loss / num_batches,
            'critic_loss': total_critic_loss / (num_batches * critic_iterations),
        }
        
        return metrics
    
    def _validate(self, val_loader):
        """Run validation."""
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()
        
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
                encoder_loss = self.reconstruction_criterion(stego, cover)
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
        num_batches = len(val_loader)
        metrics = {
            'encoder_loss': total_encoder_loss / num_batches,
            'decoder_loss': total_decoder_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'ssim': total_ssim / num_batches,
            'bit_accuracy': total_bit_accuracy / num_batches,
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
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(
            self.encoder.state_dict(), 
            os.path.join(checkpoint_dir, 'encoder.pt')
        )
        torch.save(
            self.decoder.state_dict(), 
            os.path.join(checkpoint_dir, 'decoder.pt')
        )
        torch.save(
            self.critic.state_dict(), 
            os.path.join(checkpoint_dir, 'critic.pt')
        )