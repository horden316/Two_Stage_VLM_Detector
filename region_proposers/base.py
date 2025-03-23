from abc import ABC, abstractmethod
from typing import List
from PIL import Image

class RegionProposer(ABC):
    @abstractmethod
    def get_proposals(self, image: Image.Image) -> List[List[float]]:
        """獲取圖像中的區域提案"""
        pass