# drone-defense-ml

```
python scripts/run_tracker.py --video assets/sample.mp4 --tracker-clip-threshold 0.6 --tracker-dino-threshold 0.5 --tracker-prompt drone
python scripts/run_tracker.py --tracker-clip-threshold 0.2 --tracker-dino-threshold 0.2 --tracker-prompt human
python scripts/run_tracker.py --tracker-clip-threshold 0.5 --tracker-dino-threshold 0.6 --tracker-prompt "metal bottle"
python scripts/run_tracker.py --tracker-clip-threshold 0.25 --tracker-dino-threshold 0.45 --tracker-prompt art
```