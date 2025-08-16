# Basic Spike-Transformer-Compiler Usage Demo
from spike_transformer_compiler import SpikeCompiler
import torch
import torch.nn as nn

# Simple SpikeFormer-like model for demo
class SimpleSpikeFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(64 * 7 * 7, 1000)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Autonomous compilation demo
def autonomous_compilation_demo():
    model = SimpleSpikeFormer()
    compiler = SpikeCompiler(target="simulation", optimization_level=2)
    
    compiled_model = compiler.compile(
        model,
        input_shape=(1, 3, 224, 224),
        secure_mode=True
    )
    
    # Simulate inference
    dummy_input = torch.randn(1, 3, 224, 224)
    output = compiled_model.run(dummy_input)
    
    return output

if __name__ == "__main__":
    result = autonomous_compilation_demo()
    print(f"Demo completed successfully: {result is not None}")