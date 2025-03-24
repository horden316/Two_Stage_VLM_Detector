import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
from typing import List, Optional
import matplotlib.pyplot as plt
import time

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
        start_time = time.time()
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        print(f"Transform time: {time.time() - start_time:.2f}s")
        inference_start = time.time()

        with torch.no_grad():
            predictions = self.model(img_tensor)
        print(f"Inference time: {time.time() - inference_start:.2f}s")

        post_start = time.time()
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # 篩選高置信度的框
        mask = scores > self.confidence_threshold
        filtered_boxes = boxes[mask]

        print(f"Post-processing time: {time.time() - post_start:.2f}s")
        print(f"Total time: {time.time() - start_time:.2f}s")
        
        return filtered_boxes.tolist()
    

    def visualize_proposals(self, image: Image.Image, boxes: List[List[float]], 
                        save_path: Optional[str] = None) -> Image.Image:
        """
        在圖像上繪製檢測框並顯示或保存結果。

        Args:
            image: 原始圖像
            boxes: 邊界框列表，格式為 [x_min, y_min, x_max, y_max]
            save_path: 保存圖像的路徑，如果為None則不保存
            
        Returns:
            繪製了邊界框的圖像
        """
        # 創建圖像的副本以避免修改原始圖像
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        # 為每個框選擇不同顏色
        colors = ["red", "blue", "green", "yellow", "purple", "cyan", "magenta", "orange"]

        # 在圖像上繪製每個框
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            x_min, y_min, x_max, y_max = box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)
            
        # 如果提供了保存路徑，則保存圖像
        if save_path:
            img_with_boxes.save(save_path)

        # 顯示圖像
        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_boxes)
        plt.axis('off')
        plt.show()
    
        return img_with_boxes