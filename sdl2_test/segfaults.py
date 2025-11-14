from queue import Queue
import numpy as np
from time import sleep
import cv2 as cv
from threading import Thread

import sdl2
import sdl2.ext

def fake_producer(q):
    while True:
        # simple synthetic frame
        img = np.zeros((480, 640, 3), np.uint8)
        img[:] = (0, 255, 0)  # green
        q.put(img)

class vslam():
    def __init__(self):
        sdl2.ext.init()
        print("no segfault through sdl init")

        self.vid_q = Queue()

        self.W = 1920
        self.H = 800

        self.map_win, self.vid_win = self.init_views()
        
        print("no segfault through init views")
        #self.map_surf = sdl2.SDL_GetWindowSurface(self.map_win.window)
        print("SDL_WasInit:", hex(sdl2.SDL_WasInit(0)))
        # self.map_surf = self.map_win.get_surface()
        print("no segfault through get window suface (map)")
        #self.vid_surf = sdl2.SDL_GetWindowSurface(self.vid_win.window)
        # self.vid_surf = self.vid_win.get_surface()
        print("no segfault through get window surface (vid)")

        print("no segfault till thread about to start")
        t_vid = Thread(target=fake_producer, args=(self.vid_q,), daemon=True)
        t_vid.start()
        
        self.run()

        self.cap.release()
        sdl2.ext.quit()
    
    def run(self):
        cap = cv.VideoCapture(0)
        while True:
            ret, img = cap.read()
            #img = self.vid_q.get()
            print(img)
            # display stuff
            if img is not None:
              self.run_viewer(self.vid_win, img)

    def init_views(self):
        #win_flags = (sdl2.SDL_WINDOW_MINIMIZED)
        map_win = sdl2.ext.Window("SLAM Map", (self.W, self.H))
        print("no segfault through map win")
        map_win.show()
        print("no segfault through map win show")
        vid_win = sdl2.ext.Window("SLAM Vid", (1920, 800))
        vid_win.show()
        print("no segfault through vid win show")
        print(f"map_win: {map_win}, vid_win: {vid_win}, map_win.window: {map_win.window}, vid_win.window: {vid_win.window}")
        if not map_win or not map_win.window:
            print("map_win or its window is null!")
        if not vid_win or not vid_win.window:
            print("vid_win or its window is null!")

        return map_win, vid_win

    def run_viewer(self, win, img):
        # process events
        events = sdl2.ext.get_events()
        print("right after the get events")
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                return

        surf = win.get_surface()
        print("right after the get surface command")
        if not surf:
            err = sdl2.SDL_GetError()
            raise RuntimeError(f"get_surface failed: {err!r}")

        sdl2.ext.fill(surf, 0)
        print("right after the fill command")

        windowArray = sdl2.ext.pixels2d(surf)
        h, w = windowArray.shape

        # resize expects (width, height)
        img = cv.resize(img, (w, h))
        bgra = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        packed = (bgra[:,:,3].astype(np.uint32) << 24) | \
                (bgra[:,:,2].astype(np.uint32) << 16) | \
                (bgra[:,:,1].astype(np.uint32) << 8)  | \
                (bgra[:,:,0].astype(np.uint32))

        #windowArray[:] = packed.swapaxes(0, 1)
        windowArray[:] = packed
        win.refresh()

if __name__ == "__main__":
    obj = vslam()







