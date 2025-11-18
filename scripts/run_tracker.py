import argparse

from dronedefense.tracker import Tracker, StreamInputCamera, StreamVideo
from dronedefense.tracker_gui import DisplayGUI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--tracker-clip-threshold', type=float, default=0.05)
    parser.add_argument('--tracker-dino-threshold', type=float, default=0.50)
    parser.add_argument('--tracker-prompt', type=str, default="drone")
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    print(args)

    tracker = Tracker(
        prompt=args.tracker_prompt,
        clip_threshold=args.tracker_clip_threshold,
        dino_threshold=args.tracker_dino_threshold,
    )
    if args.video is None:
        stream = StreamInputCamera(device=1)
    else:
        stream = StreamVideo(args.video)
    gui = DisplayGUI(stream, tracker, save_path=args.save_path)
    gui.run()