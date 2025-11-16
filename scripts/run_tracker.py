import argparse

from dronedefense.tracker import Tracker, StreamInputCamera, StreamVideo
from dronedefense.tracker_gui import DisplayGUI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--tracker-clip-threshold', type=float, default=0.075)
    parser.add_argument('--tracker-dino-threshold', type=float, default=0.9)
    parser.add_argument('--tracker-fastsam-interval', type=int, default=10)
    args = parser.parse_args()

    tracker = Tracker(
        clip_threshold=args.tracker_clip_threshold,
        dino_threshold=args.tracker_dino_threshold,
        fastsam_interval=args.tracker_fastsam_interval,
    )
    if args.video is None:
        stream = StreamInputCamera()
    else:
        stream = StreamVideo(args.video)
    gui = DisplayGUI(stream, tracker)
    gui.run()