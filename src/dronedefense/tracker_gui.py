import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class DisplayGUI:
    def __init__(self, stream_source, tracker):
        self.root = tk.Tk()
        self.root.title("Video Stream")
        self.root.geometry("1200x1200")

        # Canvas to display video
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Start streaming
        self.root.after(100, lambda: self.play_stream(stream_source, tracker))

    def play_stream(self, stream_source, tracker):
        for frame in stream_source.stream(tracker):
            self.display_frame(frame)
            self.root.update()
        # Pauses at the end (window stays open)

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