from queue import Queue
import threading
import numpy as np
import cv2 as cv

import sdl2
import sdl2.ext


class VSLAM:
    def __init__(self):
        # Init SDL2 on the main thread
        sdl2.ext.init()
        print("SDL_WasInit:", hex(sdl2.SDL_WasInit(0)))

        self.W = 1920
        self.H = 800

        self.win = sdl2.ext.Window("SLAM Vid", (self.W, self.H))
        self.win.show()
        print("win:", self.win, "win.window:", self.win.window)

        self.running = True

        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open camera 0")
            self.running = False

        self.run()

        self.cap.release()
        sdl2.ext.quit()

    def run(self):
        while self.running:
            # Read frame
            ret, img = self.cap.read()
            if not ret or img is None:
                continue

            self.run_viewer(self.win, img)

    def run_viewer(self, win, img):
        # Process SDL events
        for event in sdl2.ext.get_events():
            if event.type == sdl2.SDL_QUIT:
                self.running = False
                return

        # ensure video is still initialized
        if not (sdl2.SDL_WasInit(sdl2.SDL_INIT_VIDEO) & sdl2.SDL_INIT_VIDEO):
            print("Video subsystem no longer initialized, exiting viewer")
            self.running = False
            return

        # Get surface via ext helper
        surf = win.get_surface()
        if not surf:
            err = sdl2.SDL_GetError()
            print("get_surface failed:", err)
            self.running = False
            return

        # Clear to black
        sdl2.ext.fill(surf, 0)

        # Get a 2D pixel view (h, w), uint32
        window_array = sdl2.ext.pixels2d(surf)
        h, w = window_array.shape

        # Resize camera frame 
        img_resized = cv.resize(img, (w, h))
        bgra = cv.cvtColor(img_resized, cv.COLOR_BGR2BGRA)

        # Pack BGRA into AARRGGBB
        packed = (bgra[:, :, 3].astype(np.uint32) << 24) | \
                 (bgra[:, :, 2].astype(np.uint32) << 16) | \
                 (bgra[:, :, 1].astype(np.uint32) << 8)  | \
                 (bgra[:, :, 0].astype(np.uint32))

        window_array[:] = packed
        win.refresh()


if __name__ == "__main__":
    VSLAM()
