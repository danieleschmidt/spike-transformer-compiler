"""Mock models for testing when PyTorch is not available."""

class MockModule:
    """Mock PyTorch module for testing."""
    
    def __init__(self):
        self.training = False
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        return self
        
    def parameters(self):
        """Return empty parameters."""
        return []
        
    def state_dict(self):
        """Return empty state dict."""
        return {}
        

class MockTensor:
    """Mock tensor for testing."""
    
    def __init__(self, data, shape=None):
        self.data = data
        self.shape = shape or (1,)
        self.dtype = "float32"
        
    def size(self):
        return self.shape
        
    def dim(self):
        return len(self.shape)


class SimpleTestModel(MockModule):
    """Simple model for compilation testing."""
    
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x):
        """Mock forward pass."""
        return MockTensor([0.0] * self.output_size, (x.shape[0], self.output_size))


def create_test_model(model_type="simple", **kwargs):
    """Factory for creating test models."""
    if model_type == "simple":
        return SimpleTestModel(**kwargs)
    else:
        return MockModule()


def get_test_input(shape=(1, 10)):
    """Create test input tensor."""
    return MockTensor([0.0] * shape[1], shape)