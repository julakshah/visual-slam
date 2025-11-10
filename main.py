from queue import Queue
import numpy as np
from threading import Thread
from time import sleep
import cv2 as cv

import sdl2
import sdl2.ext

# local script installs
from views import Map, Vid, Frame
import camera_calib

class vslam():
    def __init__(self):

        # start a queue for backend processing
        self.vid_q = Queue()

        # define the size vars, 1080p
        self.W = 1920
        self.H = 800


        # creates the map and vid views
        self.map_win, self.vid_win = self.init_views()
        self.map_surf = sdl2.SDL_GetWindowSurface(self.map_win.window)
        self.vid_surf = sdl2.SDL_GetWindowSurface(self.vid_win.window)

        # creates the helper objects for Map & Vid
        self.mapp = Map()
        self.vidd = Vid(self.vid_q)

        self.cap = cv.VideoCapture("v-slam-dataset/test_vid_rect.mp4")

        self.run()

    def run(self):
        while True:
            # Compute image matches
            ret, img = self.cap.read()
            print(type(img))
            img = cv.resize(img, (self.W, self.H))
            frame = Frame(self.mapp, img)

            if frame.id == 0:
                continue

            # do feature matching
            im3 = self.vidd.feature_matching(self.mapp.frames[-2].img, self.mapp.frames[-1].img)
            im3 = cv.resize(im3, (self.W, self.H))

            # display stuff
            self.run_viewer(self.vid_win, self.vid_surf, im3)

    def init_views(self):
        # start sdl2 so that you can do stuff
        sdl2.ext.init()

        win_flags = (sdl2.SDL_WINDOW_MINIMIZED)
        map_win = sdl2.ext.Window("SLAM Map", (self.W, self.H), flags=win_flags)
        map_win.show()
        vid_win = sdl2.ext.Window("SLAM Vid", (self.W, self.H), flags=win_flags)
        vid_win.show()

        return map_win, vid_win


    def run_viewer(self, win, surf, img):
        # this doesn't seem to work?
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                return

        # paint surface black
        sdl2.ext.fill(surf, 0)
        windowArray = sdl2.ext.pixels2d(surf.contents)
        h, w = windowArray.shape

        img = cv.resize(img, (h, w))
        bgra = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        # pack into uint32 (AARRGGBB)
        packed = (bgra[:,:,3].astype(np.uint32) << 24) | \
                 (bgra[:,:,2].astype(np.uint32) << 16) | \
                 (bgra[:,:,1].astype(np.uint32) << 8)  | \
                 (bgra[:,:,0].astype(np.uint32))

        windowArray[:] = packed.swapaxes(0,1)
        win.refresh()


if __name__ == "__main__":
    vslam()







