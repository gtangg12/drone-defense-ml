import time
from queue import Queue
from threading import Thread

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from dronedefense.models import ModelClip, ModelDinoV2, ModelFastSAM


class Tracker:
    def __init__(
        self,
        clip_threshold: float = 0.075,
        dino_threshold: float = 0.9,
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
            masks, bboxes = self.model_fastsam(frame)
            dino_features = self.model_dino(frame)
            self.fastsam_result_queue.put((masks, bboxes, dino_features))

    def update(self, frame: Image.Image) -> Image.Image:
        print(len(self.fastsam_queue.queue), len(self.fastsam_result_queue.queue))
        if self.count % self.fastsam_interval == 0:
            if not self.fastsam_queue.full():
                self.fastsam_queue.put(frame)

        if not self.fastsam_result_queue.empty():
            pass
            # masks, bboxes, dino_features = self.fastsam_result_queue.get()

            # if self.tracked_dino_feature is None:
            #     crops = ModelFastSAM.crop(frame, bboxes)
            #     probs = self.model_clip.match(crops, ['drone'])
            #     masks = [mask for mask, prob in zip(masks, probs) if prob.item() > self.clip_threshold]

            # masks = F.interpolate(masks.unsqueeze(0).float(), size=dino_features.shape[1:3], mode='nearest').squeeze(0)  # [num_masks, h, w]

            # if self.tracked_dino_feature is not None:
            #     filtered_masks = []
            #     for mask in masks:
            #         masked_features = (dino_features * mask.unsqueeze(-1)).sum(dim=(0, 1)) / mask.sum()
            #         cosine = masked_features @ self.tracked_dino_feature
            #         if cosine > self.dino_threshold:
            #             filtered_masks.append(mask)
            #     masks = filtered_masks

            # accum = 0
            # for mask in masks:
            #     masked_features = dino_features * mask.unsqueeze(-1)  # [h, w, D]
            #     accum += masked_features.sum(dim=(0, 1)) / mask.sum()  # [D]

            # self.tracked_dino_feature = accum / len(masks)

        self.count += 1

        return frame

        if self.tracked_dino_feature is None:
            return frame

        dino_features = self.model_dino(frame) # [h, w, D]
        cosine = (dino_features @ self.tracked_dino_feature).max(dim=-1).values  # [h, w]
        detect = cosine > self.dino_threshold
        detect = F.interpolate(detect, size=frame.size[::-1], mode='nearest').squeeze().cpu().numpy()
        output = frame.copy() * 0.5 + (detect[..., None] * [255, 0, 0]) * 0.5  # red overlay
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time_elapsed = time.time() - start_time
            count += 1
            if count < int(time_elapsed * 30):
                continue  # skip frame to catch up
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = tracker.update(frame)
            yield output


if __name__ == '__main__':
    pass