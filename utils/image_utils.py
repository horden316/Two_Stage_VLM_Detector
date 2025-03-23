import requests
from io import BytesIO
from PIL import Image
from typing import List

def load_image(image_path: str) -> Image.Image:
    """從路徑或URL載入圖像"""
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

def crop_regions(image: Image.Image, boxes: List[List[float]]) -> List[Image.Image]:
    """從圖像中裁剪出區域"""
    regions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        region = image.crop((x1, y1, x2, y2))
        regions.append(region)
    return regions