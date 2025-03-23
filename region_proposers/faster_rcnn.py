import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from typing import List

from .base import RegionProposer

class FasterRCNNProposer(RegionProposer):
    def __init__(self, confidence_threshold: float = 0.5, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        self.transform = T.ToTensor()
    
    def get_proposals(self, image: Image.Image) -> List[List[float]]:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # 篩選高置信度的框
        mask = scores > self.confidence_threshold
        filtered_boxes = boxes[mask]
        
        return filtered_boxes.tolist()