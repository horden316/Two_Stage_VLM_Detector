import torch
from PIL import Image
from typing import List

from .base import VLMModel
import json

class CLIPModel(VLMModel):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None , labels_file: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        if labels_file:
            with open(labels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.candidate_labels = data.get("candidate_labels", [])
                if not self.candidate_labels:
                    raise ValueError("candidate_labels is required")
        else:
            raise ValueError("labels_file is required")
        
    def label_region(self, image: Image.Image, query_text: str = None) -> str:
        """使用CLIP進行零樣本圖像分類"""
        inputs = self.processor(
            text=self.candidate_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad(): # 關閉梯度計算，因為我們不需要進行反向傳播
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
        # 獲取最高分數的標籤
        max_prob_idx = probs[0].argmax().item()
        return self.candidate_labels[max_prob_idx]
    
    def batch_label_regions(self, images: List[Image.Image], query_text: str = None) -> List[str]:
        """批處理多個圖像區域"""
        results = []
        for image in images:
            results.append(self.label_region(image, query_text))
        return results