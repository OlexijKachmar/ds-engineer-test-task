import torch
import numpy as np
from torch import nn
from torchvision import transforms
from interfaces import DigitClassificationInterface

class CNNClassifier(DigitClassificationInterface):

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.resize_transform = transforms.Resize((32, 32))

    def predict(self, image: np.ndarray) -> int:
        """
        Args:
            image: numpy array of shape (28, 28, 1)
 
        Returns:
            Predicted digit as int in [0, 9]
        """
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected image shape (28, 28, 1), got {image.shape}")
        
        tensor = torch.from_numpy(image).float().permute(2, 0, 1)
        resized = self.resize_transform(tensor).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(resized)
            _, class_idx = torch.max(logits, 1)

        return int(class_idx.item())
