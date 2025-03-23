from typing import Optional # Optional是一個泛型，表示可以是任意類型，也可以是None
from ..vlm_models.base import VLMModel
from ..vlm_models.clip_model import CLIPModel

class VLMFactory:
    @staticmethod
    def create_vlm(vlm_type: str, model_name: Optional[str] = None, device: Optional[str] = None) -> VLMModel:
        """創建VLM實例的工廠方法"""
        if vlm_type.lower() == "clip":
            return CLIPModel(model_name if model_name else "openai/clip-vit-base-patch32", device)
        else:
            raise ValueError(f"不支援的VLM類型: {vlm_type}")