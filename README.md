# drone-defense-ml

Video streaming:
```
python scripts/run_tracker.py --video assets/asset1.mp4 --tracker-clip-threshold 0.65 --tracker-dino-threshold 0.50 --tracker-prompt drone
python scripts/run_tracker.py --video assets/asset2.mp4 --tracker-clip-threshold 0.50 --tracker-dino-threshold 0.50--tracker-prompt drone
python scripts/run_tracker.py --video assets/asset3.mp4 --tracker-clip-threshold 0.50 --tracker-dino-threshold 0.60 --tracker-prompt truck
```

Live camera input:
```
python scripts/run_tracker.py --tracker-clip-threshold 0.2 --tracker-dino-threshold 0.2 --tracker-prompt human
python scripts/run_tracker.py --tracker-clip-threshold 0.5 --tracker-dino-threshold 0.6 --tracker-prompt "metal bottle"
python scripts/run_tracker.py --tracker-clip-threshold 0.25 --tracker-dino-threshold 0.45 --tracker-prompt art
```