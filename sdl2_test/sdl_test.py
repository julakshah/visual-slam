import sys
import ctypes
import numpy as np
import sdl2
import sdl2.ext

def main():
    # Init SDL2
    sdl2.ext.init()
    print("SDL_WasInit:", hex(sdl2.SDL_WasInit(0)))

    win_w, win_h = 800, 600
    win = sdl2.ext.Window("SDL Surface Test", size=(win_w, win_h))
    win.show()
    win2 = sdl2.ext.Window("blah",size=(win_w, win_h))
    win2.show()

    print("win:", win, "win.window:", win.window)

    running = True
    frame = 0

    while running:
        # Process events
        events = sdl2.ext.get_events()
        for e in events:
            if e.type == sdl2.SDL_QUIT:
                running = False

        # Extra guard: if video somehow got shut down, bail gracefully
        if not (sdl2.SDL_WasInit(sdl2.SDL_INIT_VIDEO) & sdl2.SDL_INIT_VIDEO):
            print("Video subsystem not initialized anymore, exiting.")
            break

        # Get fresh surface every frame
        surf = win.get_surface()
        if not surf:
            err = sdl2.SDL_GetError()
            print("get_surface failed:", err)
            running = False
            break

        # Fill windowArray with a simple pattern (no OpenCV)
        sdl2.ext.fill(surf, 0)  # clear to black

        window_array = sdl2.ext.pixels2d(surf)  # shape (h, w), uint32
        h, w = window_array.shape

        # Simple moving gradient, stays in-bounds
        x = np.arange(w, dtype=np.uint32)
        y = np.arange(h, dtype=np.uint32)
        xx, yy = np.meshgrid(x, y)

        r = ((xx + frame) % 256).astype(np.uint32)
        g = ((yy + frame) % 256).astype(np.uint32)
        b = (((xx + yy) // 2 + frame) % 256).astype(np.uint32)
        a = np.full_like(r, 255, dtype=np.uint32)

        packed = (a << 24) | (r << 16) | (g << 8) | b  # ARGB / AARRGGBB

        window_array[:] = packed
        win.refresh()

        frame += 1

    sdl2.ext.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())
