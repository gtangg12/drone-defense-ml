import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dronedefense.models import ModelClip, ModelDinoV2, ModelFastSAM


class Tracker:
    def __init__(
        self,
        clip_threshold: float = 0.075,
        dino_threshold: float = 0.65,
        fastsam_interval: int = 10,
    ):
        self.count = 0
        self.model_clip = ModelClip()
        self.model_dino = ModelDinoV2()
        self.model_fastsam = ModelFastSAM()
        self.clip_threshold = clip_threshold
        self.dino_threshold = dino_threshold
        self.fastsam_interval = fastsam_interval

        self.tracked_dino_feature = None

        # setup fastsam worker thread
        self.fastsam_queue = Queue(maxsize=1)
        self.fastsam_result_queue = Queue(maxsize=1)
        self.fastsam_worker = Thread(target=self._fastsam_worker, daemon=True)
        self.fastsam_worker.start()

    def _fastsam_worker(self):
        while True:
            frame = self.fastsam_queue.get()
            if frame is None:
                break
            bmasks, bboxes = self.model_fastsam(frame)
            dino_features = self.model_dino(frame)

            crops = ModelFastSAM.crop(frame, bmasks, expand_ratio=1.5)
            probs = self.model_clip.match(crops, ['art'])
            print("PROB", torch.max(probs).item())
            bmasks = [mask for mask, prob in zip(bmasks, probs) if prob.item() > self.clip_threshold]
            if len(bmasks) == 0:
                continue
            bmasks = torch.stack(bmasks)
            self.fastsam_result_queue.put((bmasks, bboxes, dino_features, probs))

    def update(self, frame: Image.Image) -> Image.Image:
        if not self.fastsam_queue.full():
            self.fastsam_queue.put(frame)

        if not self.fastsam_result_queue.empty():
            bmasks, bboxes, dino_features, probs = self.fastsam_result_queue.get()
            bmasks = F.interpolate(bmasks.unsqueeze(0).float(), size=dino_features.shape[0:2], mode='nearest').squeeze(0)  # [num_masks, h, w]
            accum = 0
            for mask in bmasks:
                if mask.sum() == 0:
                    continue
                masked_features = dino_features * mask.unsqueeze(-1)  # [h, w, D]
                accum += masked_features.sum(dim=(0, 1)) / mask.sum()  # [D]
            accum = accum / torch.linalg.norm(accum)

            self.tracked_dino_feature = accum / len(bmasks)

        self.count += 1

        if self.tracked_dino_feature is None:
            return frame
        dino_features = self.model_dino(frame) # [h, w, D]

        cosine = (dino_features.reshape(-1, dino_features.shape[-1]) @ self.tracked_dino_feature)
        cosine = cosine.reshape(dino_features.shape[0], dino_features.shape[1])
        print("COSINE", cosine.max().item())
        detect = cosine > self.dino_threshold
        #cv2.imwrite(f"detect_{self.count}.png", np.array(detect.numpy() * 255).astype('uint8'))
        detect = F.interpolate(detect[None, None, ...].float(), size=(frame.height, frame.width), mode='nearest').bool().squeeze().cpu().numpy()
        output = np.array(frame, dtype=float)
        output[detect] = output[detect] * 0.5 + np.array([255, 0, 0]) * 0.5
        return Image.fromarray(output.astype('uint8'))


class StreamVideo:
    def __init__(self, path: str):
        self.video_cap = cv2.VideoCapture(path)

    def stream(self, tracker: Tracker):
        start_time = time.time()
        count = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time_elapsed = time.time() - start_time
            count += 1
            if count < int(time_elapsed * 30):
                continue # skip frame to catch up
            output = tracker.update(frame)
            yield output


class StreamInputCamera:
    def __init__(self, device=0):
        self.video_cap = cv2.VideoCapture(device)

    def stream(self, tracker: Tracker, max_iterations=10000):
        for _ in range(max_iterations):
            ret, frame = self.video_cap.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output = tracker.update(frame)
            yield output


if __name__ == '__main__':
    pass