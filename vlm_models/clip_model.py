import torch
from PIL import Image
from typing import List

from .base import VLMModel
import json

class CLIPModel(VLMModel):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None , candidate_labels: list = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.candidate_labels = candidate_labels
        
        
    def label_region(self, image: Image.Image, query_text: str = None) -> str:
        """使用CLIP進行零樣本圖像分類"""
        inputs = self.processor(
            text=self.candidate_labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad(): # 關閉梯度計算，因為我們不需要進行反向傳播
            outputs = self.model(**inputs) #在 Python 中，** 用於將字典解包成關鍵字參數。這意味著字典中的每個鍵值對都會作為單獨的關鍵字參數傳遞給函數
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