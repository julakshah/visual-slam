from queue import Queue
from threading import Thread
from time import sleep

import cv2 as cv
import numpy as np
import sdl2
import sdl2.ext

# local script installs
from views import Map, Vid, Frame


fpath = "v-slam-dataset/test_vid_rect.mp4"

class vslam():
    def __init__(self):

        # start some queues for video frames to be sent over
        self.vid_q = Queue()
        self.map_q = Queue()

        # define the size vars, 1080p
        self.W = 1920
        self.H = 800
        vid_size = (1920, 800)
        map_size = (1200, 1200)


        # creates the map and vid views
        self.map_win, self.vid_win = self.init_views(map_size, vid_size)
        self.map_surf = sdl2.SDL_GetWindowSurface(self.map_win.window)
        self.vid_surf = sdl2.SDL_GetWindowSurface(self.vid_win.window)

        # creates the helper objects for Map & Vid
        self.mapp = Map(map_size, self.map_q)

        cap = cv.VideoCapture(fpath)
        self.vidd = Vid(self.mapp, cap, self.vid_q)

        #thread for the video visualization
        t_vid = Thread(target=self.vidd.run, daemon=True)
        t_vid.start()

        #thread for the map visualization
        t_map = Thread(target=self.mapp.run, daemon=True)
        t_map.start()


        self.run()

    def run(self):
        while True:

            vid_img = self.vid_q.get()
            # display stuff
            if vid_img is not None:
              self.run_viewer(self.vid_win, self.vid_surf, vid_img)

            map_img = self.map_q.get()
            if map_img is not None:
              self.run_viewer(self.map_win, self.map_surf, map_img)


    def init_views(self, map_size, vid_size):
        # start sdl2 so that you can do stuff
        sdl2.ext.init()

        win_flags = (sdl2.SDL_WINDOW_MINIMIZED)
        map_win = sdl2.ext.Window("SLAM Map", map_size, flags=win_flags)
        map_win.show()
        vid_win = sdl2.ext.Window("SLAM Vid", vid_size, flags=win_flags)
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



