# import Simple Directmedia Layer (SDL) or more specifically (PySDL2)
import sdl2
import sdl2.ext
import cv2


class Display(object):
    """A window display class to render numpy images through PySLD2"""

    def __init__(self, W, H):
        # Initialize SDL 2 library
        sdl2.ext.init()
        # Initialize window with params
        self.window = sdl2.ext.Window("V-SLAM", size=(W, H))
        self.window.show()
        # Save display parameters
        self.W, self.H = W, H

    def paint(self, img):
        # Make sure that the img to be painted fits window
        img = cv2.resize(img, (self.W, self.H))
        # Get the events queue (eg. mouse & keyboard inputs and buttom presses)
        events = sdl2.ext.get_events()
        # Ensure ability to quit
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        # Convert the image data layout to properly render through SDL2
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = img.swapaxes(0, 1)
        self.window.refresh()
