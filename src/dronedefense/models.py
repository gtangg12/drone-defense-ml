# import torch
# from transformers import AutoImageProcessor, AutoModel
# from transformers.image_utils import load_image

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = load_image(url)

# pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
# processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(
#     pretrained_model_name, 
#     device_map="auto", 
# )

# inputs = processor(images=image, return_tensors="pt").to(model.device)
# with torch.inference_mode():
#     outputs = model(**inputs)

# pooled_output = outputs.pooler_output
# print("Pooled output shape:", pooled_output.shape)

from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks

import sys
sys.path.append('./third_party/FastSAM')
from fastsam import FastSAM, FastSAMPrompt
sys.path.pop()


class ModelFastSAM:
    def __init__(
        self,
        checkpoint: str,
        max_regions=128,
        min_area=128,
        device=None,
    ):
        self.model = FastSAM(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_regions = max_regions
        self.min_area = min_area

    def __call__(self, image: np.ndarray, prompt: Optional[dict] = None):
        H, W = image.shape[:2]
        prompt = prompt or dict(func="everything")

        results = self.model(image, device=self.device, retina_masks=True, verbose=False)
        engine = FastSAMPrompt(image, device=self.device, results=results)
        masks = getattr(engine, f"{prompt.pop('func')}_prompt")(**prompt)

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

        annotations = [{'mask': m, 'area': m.sum()} for m in masks if m.sum() >= self.min_area]
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)[:self.max_regions]

        if not annotations:
            return torch.zeros((H, W, 1), dtype=torch.bool), torch.zeros((1, 4), dtype=torch.int32)

        bmasks = torch.stack([torch.from_numpy(a['mask']) for a in annotations])
        bboxes = masks_to_boxes(bmasks)
        return bmasks.permute(1, 2, 0).to(torch.bool), bboxes.to(torch.int32)

    @classmethod
    def visualize_masks(cls, image: np.ndarray, bmasks: torch.Tensor) -> Image.Image:
        bmasks = bmasks.permute(2, 0, 1).cpu().numpy()
        output = image.copy().astype(np.float32)
        colors = np.random.randint(0, 255, size=(len(bmasks), 3))
        for mask, color in zip(bmasks, colors):
            output[mask] = output[mask] * 0.5 + color * 0.5
        return Image.fromarray(output.astype(np.uint8))
        

if __name__ == '__main__':
    model = ModelFastSAM("/Users/gtangg12/Desktop/drone-defense-ml/checkpoints/FastSAM-x.pt")
    image = np.array(Image.open("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones.png").convert("RGB"))

    import time
    start_time = time.time()
    bmasks, bboxes = model(image)
    print(f"Inference time: {time.time() - start_time:.2f} seconds")

    output = ModelFastSAM.visualize_masks(image, bmasks)
    output.save("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones_masks.png")