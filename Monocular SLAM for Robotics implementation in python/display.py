import sdl2
import sdl2.ext
import cv2

class Display(object):
    def __init__(self, W, H):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("SLAM", size=(W, H))
        self.window.show()
        self.W, self.H = W, H

    def paint(self, img):
        img = cv2.resize(img, (self.W, self.H))
        # Retrieves a list of SDL2 events.
        events = sdl2.ext.get_events()
        for event in events:
            # Checks if the event type is SDL_QUIT (window close event).
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        # Retrieves a 3D numpy array that represents the pixel data of the window's surface.
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        # Updates the pixel data of the window's surface with the resized image. 
        # img.swapaxes(0, 1) swaps the axes of the image array to match the expected format of the SDL surface.
        surf[:, :, 0:3] = img.swapaxes(0, 1)
        # Refreshes the window to display the updated surface.
        self.window.refresh()

