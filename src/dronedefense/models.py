from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks

import sys
sys.path.append('./third_party/FastSAM')
from fastsam import FastSAM, FastSAMPrompt
sys.path.pop()


class ModelDinoV2:
    PATCH_SIZE = 14

    def __init__(
        self,
        backbone: str = 'dinov2_vits14',
        downsample_factor: float = 2,
        device=None,
    ):
        self.model = torch.hub.load('facebookresearch/dinov2', backbone)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.downsample_factor = downsample_factor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: np.ndarray):
        H, W = image.shape[:2]
        round_dim = lambda x: x // self.PATCH_SIZE * self.PATCH_SIZE
        inH = round_dim(int(H / self.downsample_factor))
        inW = round_dim(int(W / self.downsample_factor))

        image = self.transform(image)
        image = transforms.functional.resize(image, (inH, inW))
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings = self.model.forward_features(image)['x_norm_patchtokens']
        embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        embeddings = embeddings.reshape(
            inH // self.PATCH_SIZE,
            inW // self.PATCH_SIZE,
            -1,
        )
        return embeddings

    @classmethod
    def visualize_pca(cls, features: torch.Tensor, n_components: int = 3):
        H, W, D = features.shape
        features = features.cpu().numpy().reshape(-1, D)
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        pca_features = pca_features.reshape(H, W, n_components)
        pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
        pca_features = (pca_features * 255).astype(np.uint8)
        return Image.fromarray(pca_features)


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
    import time

    image = np.array(Image.open("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones.png").convert("RGB"))

    model = ModelDinoV2(backbone='dinov2_vits14', downsample_factor=2)
    for i in range(5):
        start_time = time.time()
        features = model(image)
        print(f"Inference time: {time.time() - start_time:.2f} seconds")
    output = ModelDinoV2.visualize_pca(features, n_components=3)
    output.save("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones_dino.png")

    model = ModelFastSAM("/Users/gtangg12/Desktop/drone-defense-ml/checkpoints/FastSAM-x.pt")
    for i in range(5):
        start_time = time.time()
        bmasks, bboxes = model(image)
        print(f"Inference time: {time.time() - start_time:.2f} seconds")
    output = ModelFastSAM.visualize_masks(image, bmasks)
    output.save("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones_masks.png")