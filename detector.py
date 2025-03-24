import json
from factories.vlm_factory import VLMFactory
from region_proposers.faster_rcnn import FasterRCNNProposer
import utils.image_utils as UT
def getLabel(labels_file : str) -> list:
    if labels_file:
        with open(labels_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(data)
            candidate_labels = data["candidate_labels"]
        if not candidate_labels:
            raise ValueError("candidate_labels is required")

        
    return candidate_labels


if __name__ == '__main__':
    candidate_labels = getLabel("./labels.json")
    rcnn = FasterRCNNProposer(device="cpu")
    vlm_model = VLMFactory.create_vlm(vlm_type = "clip", candidate_labels = candidate_labels, device = "mps")

    image = UT.load_image("/Users/horden/Desktop/cat1.jpg")
    proposal_boxes = rcnn.get_proposals(image)
    print(proposal_boxes)

    rcnn.visualize_proposals(image, proposal_boxes)

    # 裁剪區域
    regions = UT.crop_regions(image, proposal_boxes)
    
    # 第二階段：使用VLM標記
    labels = vlm_model.batch_label_regions(regions)

    print(labels)

