from typing import Optional # Optional是一個泛型，表示可以是任意類型，也可以是None
from vlm_models.base import VLMModel
from vlm_models.clip_model import CLIPModel

class VLMFactory:
    @staticmethod
    def create_vlm(vlm_type: str, candidate_labels: list, model_name: Optional[str] = None, device: Optional[str] = None) -> VLMModel:
        """創建VLM實例的工廠方法"""

        if not candidate_labels:
            raise ValueError("candidate_labels is required")
        if vlm_type.lower() == "clip":
            return CLIPModel(model_name if model_name else "openai/clip-vit-base-patch32", device, candidate_labels)
        else:
            raise ValueError(f"不支援的VLM類型: {vlm_type}")