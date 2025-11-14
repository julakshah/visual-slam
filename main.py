from queue import Queue
import numpy as np
from threading import Thread
from time import sleep
import ctypes
import cv2 as cv
import faulthandler
faulthandler.enable()
import sdl2
import sdl2.ext
import sys

# local script installs
from views import Vid, Frame
from third_party.descriptor import Descriptor

class vslam():
    def __init__(self,path):
        """
        Main class to run visual odometry 
        Args:
            path: argument to pass as input to cv.VideoCapture
                0 for computer webcam, "path/to/file" for a video file
        """
        # start sdl2 so that you can do stuff
        sdl2.ext.init()
        print("no segfault through sdl init")

        # start a queue for backend processing
        self.vid_q = Queue()

        # define the size vars, 1080p
        self.W = 1920
        self.H = 800

        # creates the map and vid views
        self.vid_win = self.init_views()
        
        ### Old code for writing to window via SDL_Surface
        ### Segfaults every so often, couldn't easily diagnose
        ### Replaced by using SDL_Texture
        #print("no segfault through init views")
        #self.map_surf = sdl2.SDL_GetWindowSurface(self.map_win.window)
        #print("SDL_WasInit:", hex(sdl2.SDL_WasInit(0)))
        #self.map_surf = self.map_win.get_surface()
        #rint("no segfault through get window suface (map)")
        #self.vid_surf = sdl2.SDL_GetWindowSurface(self.vid_win.window)
        #self.vid_surf = self.vid_win.get_surface()
        #print("no segfault through get window surface (vid)")

        # Create renderers
        
        #self.map_renderer = sdl2.SDL_CreateRenderer(
        #    self.map_win.window, -1,
        #    sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
        #)
        #if not self.map_renderer:
        #    raise RuntimeError("failed to create map renderer: " + sdl2.SDL_GetError().decode())
        
        self.vid_renderer = sdl2.SDL_CreateRenderer(
            self.vid_win.window, -1,
            sdl2.SDL_RENDERER_ACCELERATED | sdl2.SDL_RENDERER_PRESENTVSYNC
        )
        if not self.vid_renderer:
            raise RuntimeError("failed to create vid renderer: " + sdl2.SDL_GetError().decode())
        print("Created renderers!")
        
        # We'll create and destroy textures as needed if sizes change
        self.map_texture = None
        self.vid_texture = None
        self.map_tex_w = 0
        self.vid_tex_w = 0
        self.map_tex_h = 0
        self.vid_tex_h = 0
        
        # creates the helper objects for Map & Vid
        # self.mapp = Map()
        self.mapp = Descriptor()

        cap = cv.VideoCapture(path)
        #cap = cv.VideoCapture(0)
        self.vidd = Vid(self.mapp, cap, self.vid_q)

        print("no segfault till thread about to start")
        t_vid = Thread(target=self.vidd.run, daemon=True)
        t_vid.start()

        self.run()

    def run(self):
        while True:
            img = self.vid_q.get()
            print(img)
            print("\n\nGot image!\n\n")
            # display stuff
            if img is not None:
                self.run_viewer(img)

    def init_views(self):
        #win_flags = (sdl2.SDL_WINDOW_MINIMIZED)
        #map_win = sdl2.ext.Window("SLAM Map", (self.W, self.H))
        #print("no segfault through map win")
        #map_win.show()
        #print("no segfault through map win show")
        vid_win = sdl2.ext.Window("SLAM Vid", (1920, 800))
        vid_win.show()
        #print("no segfault through vid win show")
        #print(f"map_win: {map_win}, vid_win: {vid_win}, map_win.window: {map_win.window}, vid_win.window: {vid_win.window}")
        if not vid_win or not vid_win.window:
            print("vid_win or its window is null!")

        return vid_win
    
    def _ensure_texture(self, w, h, tex):
        # Recreate texture if size changed
        if tex and (w != self.vid_tex_w or h != self.vid_tex_h):
            sdl2.SDL_DestroyTexture(tex)
            self.vid_texture = None

        if not tex:
            self.vid_texture = sdl2.SDL_CreateTexture(
                self.vid_renderer,
                sdl2.SDL_PIXELFORMAT_ARGB8888,
                sdl2.SDL_TEXTUREACCESS_STREAMING,
                w,
                h
            )
            if not self.vid_texture:
                raise RuntimeError("failed to create texture: " + sdl2.SDL_GetError().decode())
            self.vid_tex_w, self.vid_tex_h = w, h

    def run_viewer(self, img):
        # process events
        events = sdl2.ext.get_events()
        print("right after the get events")
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                return
            
        h_win = self.H
        w_win = self.W

        frame = cv.resize(img, (w_win, h_win))
        # Convert BGR to BGRA
        bgra = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)

        h, w, _ = bgra.shape
        self._ensure_texture(w, h, self.vid_texture)

        #surf = win.get_surface()
        #print("right after the get surface command")
        #if not surf:
        #    err = sdl2.SDL_GetError()
        #    raise RuntimeError(f"get_surface failed: {err!r}")

        #sdl2.ext.fill(surf, 0)
        #print("right after the fill command")

        #windowArray = sdl2.ext.pixels2d(surf)
        #w, h = windowArray.shape

        # resize expects (width, height)
        # img = cv.resize(img, (self.W, self.H))
        #img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        # bgra = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

        packed = (bgra[:,:,3].astype(np.uint32) << 24) | \
                (bgra[:,:,2].astype(np.uint32) << 16) | \
                (bgra[:,:,1].astype(np.uint32) << 8)  | \
                (bgra[:,:,0].astype(np.uint32))
        
        packed_ct = np.ascontiguousarray(packed)

        pitch = self.W * 4  # 4 bytes per pixel
        ptr = packed_ct.ctypes.data_as(ctypes.c_void_p)

        # SDL_UpdateTexture(texture, rect, pixels, pitch)
        ret = sdl2.SDL_UpdateTexture(
            self.vid_texture, None, ptr, pitch
        )
        if ret != 0:
            print("SDL_UpdateTexture error:", sdl2.SDL_GetError().decode())
            return

        # Render
        sdl2.SDL_RenderClear(self.vid_renderer)
        sdl2.SDL_RenderCopy(self.vid_renderer, self.vid_texture, None, None)
        sdl2.SDL_RenderPresent(self.vid_renderer)

        #windowArray[:] = packed.swapaxes(0, 1)
        #windowArray[:] = packed.swapaxes(0,1)
        #win.refresh()
    
    def __del__(self):
        """ Clean up SDL objects """
        """
        if self.map_texture:
            sdl2.SDL_DestroyTexture(self.map_texture)
        if self.vid_texture:
            sdl2.SDL_DestroyTexture(self.vid_texture)
        if self.map_renderer:
            sdl2.SDL_DestroyRenderer(self.map_renderer)
        if self.vid_renderer:
            sdl2.SDL_DestroyTexture(self.vid_renderer)
        if self.map_win:
            sdl2.SDL_DestroyWindow(self.map_win)
        if self.vid_win:
            sdl2.SDL_DestroyWindow(self.vid_win)
        """
        sdl2.ext.quit()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = 0
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit(): # cast to int in case we want a video device
            path = int(sys.argv[1])
        else:
            path = sys.argv[1]
    obj = vslam(path)







