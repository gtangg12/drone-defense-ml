import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class DisplayGUI:
    def __init__(self, stream_source, tracker, save_path=None, fps=10):
        self.root = tk.Tk()
        self.root.title("Video Stream")
        self.root.geometry("1200x1200")

        # Canvas to display video
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Video writer setup
        self.save_path = save_path
        self.fps = fps
        self.video_writer = None
        self.frame_size = None

        # Start streaming
        self.root.after(100, lambda: self.play_stream(stream_source, tracker))

    def play_stream(self, stream_source, tracker):
        for frame in stream_source.stream(tracker):
            # Save frame if save_path is provided
            if self.save_path:
                self.save_frame(frame)
            
            self.display_frame(frame)
            self.root.update()
        
        # Release video writer when done
        if self.video_writer:
            self.video_writer.release()
            print(f"Video saved to {self.save_path}")
        
        # Pauses at the end (window stays open)

    def save_frame(self, frame):
        frame_array = np.array(frame)
        H, W = frame_array.shape[:2]
        
        # Initialize video writer on first frame
        if self.video_writer is None:
            self.frame_size = (W, H)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.save_path, fourcc, self.fps, self.frame_size
            )
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

    def display_frame(self, frame):
        canvasH = self.canvas.winfo_height()
        canvasW = self.canvas.winfo_width()

        H, W = frame.height, frame.width
        frame = np.array(frame)
        scale = min(canvasW / W, canvasH / H)
        newH, newW = int(H * scale), int(W * scale)
        frame_resized = cv2.resize(frame, (newW, newH))

        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.delete("all")
        self.canvas.create_image(canvasW // 2, canvasH // 2, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo
    
    def run(self):
        self.root.mainloop()