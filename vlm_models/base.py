from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class VLMModel(ABC):
    @abstractmethod
    def __init__(self, model_name: str, device: str):
        pass
    
    @abstractmethod
    def label_region(self, image: Image.Image, query_text: str) -> str:
        """為一個圖像區域生成標籤"""
        pass
    
    @abstractmethod
    def batch_label_regions(self, images: List[Image.Image], query_text: str) -> List[str]:
        """批量處理多個圖像區域"""
        pass