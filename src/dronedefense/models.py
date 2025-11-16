from typing import Optional

import clip
import torch
import torch.nn as nn
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


class ModelClip(nn.Module):
    EMBEDDING_DIM = 512

    def __init__(
        self,
        name: str = 'ViT-L/14',
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(name, device=self.device)

    def encode_image(self, image: Image.Image | list[Image.Image]):
        if isinstance(image, Image.Image):
            image = [image]
        image_tensor = torch.stack([self.preprocess(img) for img in image]).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    def encode_text(self, text: str | list[str]):
        if isinstance(text, str):
            text = [text]
        tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(tokens)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    def match(self, images: list[Image.Image], texts: list[str]):
        image_embeddings, text_embeddings = self.encode_image(images), self.encode_text(texts)
        logits = (image_embeddings @ text_embeddings.T).squeeze(-1)
        return torch.softmax(100 * logits, dim=0)  # probs


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

    def __call__(self, image: Image.Image):
        image = np.array(image)
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
        checkpoint: str = "./checkpoints/FastSAM-x.pt",
        max_regions: int = 128,
        min_area: int = 1024,
        device: Optional[str] = None,
    ):
        self.model = FastSAM(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_regions = max_regions
        self.min_area = min_area

    def __call__(self, image: Image.Image, prompt: Optional[dict] = None):
        image = np.array(image)
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
        return bmasks.to(torch.bool), bboxes.to(torch.int32)

    @classmethod
    def visualize_masks(cls, image: np.ndarray, bmasks: torch.Tensor) -> Image.Image:
        bmasks = bmasks.cpu().numpy()
        output = image.copy().astype(np.float32)
        colors = np.random.randint(0, 255, size=(len(bmasks), 3))
        for mask, color in zip(bmasks, colors):
            output[mask] = output[mask] * 0.5 + color * 0.5
        return Image.fromarray(output.astype(np.uint8))

    @classmethod
    def crop(cls, image: Image.Image, bmasks: torch.Tensor, expand_ratio=1.0) -> list[Image.Image]:
        crops = []
        for mask in bmasks:
            coords = torch.nonzero(mask)
            if len(coords) == 0:
                continue  # Skip empty masks
            y_min, x_min = coords.min(dim=0).values
            y_max, x_max = coords.max(dim=0).values
            y_center = (y_min + y_max) / 2
            x_center = (x_min + x_max) / 2
            h_half = (y_max - y_min) * expand_ratio / 2
            w_half = (x_max - x_min) * expand_ratio / 2
            y_min = max(0, int(y_center - h_half))
            y_max = min(image.height, int(y_center + h_half))
            x_min = max(0, int(x_center - w_half))
            x_max = min(image.width, int(x_center + w_half))
            crop = image.crop((x_min, y_min, x_max, y_max))
            crops.append(crop)
        return crops


if __name__ == '__main__':
    import time

    image = Image.open("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones.png").convert("RGB")

    # model = ModelDinoV2(backbone='dinov2_vits14', downsample_factor=2)
    # for i in range(5):
    #     start_time = time.time()
    #     features = model(image)
    #     print(f"Inference time: {time.time() - start_time:.2f} seconds")
    # output = ModelDinoV2.visualize_pca(features, n_components=3)
    # output.save("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones_dino.png")

    # model = ModelFastSAM("/Users/gtangg12/Desktop/drone-defense-ml/checkpoints/FastSAM-x.pt")
    # for i in range(5):
    #     start_time = time.time()
    #     bmasks, bboxes = model(image)
    #     print(f"Inference time: {time.time() - start_time:.2f} seconds")
    # output = ModelFastSAM.visualize_masks(image, bmasks)
    # output.save("/Users/gtangg12/Desktop/drone-defense-ml/assets/drones_masks.png")

    model = ModelFastSAM("/Users/gtangg12/Desktop/drone-defense-ml/checkpoints/FastSAM-x.pt")
    bmasks, _ = model(image)
    print(bmasks.shape)
    model_clip = ModelClip(name='ViT-B/16')

    start_time = time.time()
    crops = ModelFastSAM.crop(image, bmasks, expand_ratio=1.2)
    probs = model_clip.match(crops, ["drone"])
    print(f"Clip time: {time.time() - start_time:.2f} seconds")

    for i, crop in enumerate(crops):
        crop.save(f"/Users/gtangg12/Desktop/drone-defense-ml/assets/crop_{i}.png")
    print(probs)