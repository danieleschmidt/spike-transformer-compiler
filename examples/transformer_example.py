#!/usr/bin/env python3
"""Transformer model example for Spike-Transformer-Compiler."""

import sys
sys.path.append('src')

import torch
import torch.nn as nn
from spike_transformer_compiler import SpikeCompiler

class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""
    
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, num_classes=10):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Patch embedding (simple linear projection)
        self.patch_embed = nn.Linear(28*28, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.view(B, -1)  # (B, 784)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim)
        x = x.unsqueeze(1)  # (B, 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        
        return x

def create_spike_transformer():
    """Create a SpikeFormer-like model."""
    class SpikeTransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            
        def forward(self, x):
            # Attention with residual
            norm_x = self.norm1(x)
            attn_out, _ = self.attention(norm_x, norm_x, norm_x)
            x = x + attn_out
            
            # MLP with residual
            norm_x = self.norm2(x)
            mlp_out = self.mlp(norm_x)
            x = x + mlp_out
            
            return x
    
    class SpikeFormer(nn.Module):
        def __init__(self, embed_dim=192, num_heads=6, num_layers=4, num_classes=10):
            super().__init__()
            
            # Patch embedding
            self.patch_embed = nn.Linear(28*28, embed_dim)
            
            # Transformer blocks
            self.blocks = nn.ModuleList([
                SpikeTransformerBlock(embed_dim, num_heads) 
                for _ in range(num_layers)
            ])
            
            # Classification head
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
            
        def forward(self, x):
            B, C, H, W = x.shape
            x = x.view(B, -1)  # Flatten
            
            x = self.patch_embed(x)  # Embed
            x = x.unsqueeze(1)  # Add sequence dimension
            
            for block in self.blocks:
                x = block(x)
            
            x = self.norm(x)
            x = x.mean(dim=1)  # Global pooling
            x = self.head(x)
            
            return x
    
    return SpikeFormer()

def main():
    """Demonstrate transformer compilation."""
    print("=== Spike-Transformer-Compiler Transformer Example ===\n")
    
    # Create transformer models
    print("1. Creating transformer models...")
    simple_transformer = SimpleTransformer(embed_dim=128, num_heads=4, num_layers=2)
    spike_transformer = create_spike_transformer()
    
    models = [
        ("Simple Transformer", simple_transformer),
        ("SpikeFormer", spike_transformer)
    ]
    
    for model_name, model in models:
        print(f"\n=== {model_name} ===")
        model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Initialize compiler
        compiler = SpikeCompiler(
            target="simulation",
            optimization_level=2,
            time_steps=8,  # More time steps for transformers
            verbose=False
        )
        
        # Compile model
        input_shape = (1, 1, 28, 28)
        
        try:
            print("Compiling...")
            compiled_model = compiler.compile(
                model,
                input_shape=input_shape,
                profile_energy=True
            )
            
            print("✓ Compilation successful!")
            print(f"   Energy per inference: {compiled_model.energy_per_inference:.3f} nJ")
            print(f"   Hardware utilization: {compiled_model.utilization:.1%}")
            
            # Test inference
            test_input = torch.randn(1, 1, 28, 28)
            output = compiled_model.run(
                test_input,
                time_steps=8,
                return_spike_trains=False
            )
            
            print("✓ Inference successful!")
            print(f"   Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("\n=== Transformer example completed! ===")

if __name__ == "__main__":
    main()