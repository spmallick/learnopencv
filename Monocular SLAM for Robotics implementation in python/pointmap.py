from multiprocessing import Process, Queue
import numpy as np
import cv2
import pangolin
import OpenGL.GL as gl

# Global map // 3D map visualization using pangolin
class Map(object):
    def __init__(self):
        self.frames = [] # camera frames [means camera pose]
        self.points = [] # 3D points of map
        self.state = None # variable to hold current state of the map and cam pose
        self.q = None # A queue for inter-process communication. | q for visualization process
        self.q_image = None
    def create_viewer(self):
        # Parallel Execution: The main purpose of creating this process is to run 
        # the `viewer_thread` method in parallel with the main program. 
        # This allows the 3D viewer to update and render frames continuously 
        # without blocking the main execution flow.
        
        self.q = Queue() # q is initialized as a Queue
        self.q_image = Queue()

        # initializes the Parallel process with the `viewer_thread` function 
        # the arguments that the function takes is mentioned in the args var
        p = Process(target=self.viewer_thread, args=(self.q,)) 
        
        # daemon true means, exit when main program stops
        p.daemon = True
        
        # starts the process
        p.start()

    def viewer_thread(self, q):
        # `viewer_thread` takes the q as input
        # initializes the viz window
        self.viewer_init(1280, 720)
        # An infinite loop that continually refreshes the viewer
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        
        # This ensures that only the nearest objects are rendered, 
        # creating a realistic representation of the scene with 
        # correct occlusions.
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Sets up the camera with a projection matrix and a model-view matrix
        self.scam = pangolin.OpenGlRenderState(
            # `ProjectionMatrix` The parameters specify the width and height of the viewport (w, h), the focal lengths in the x and y directions (420, 420), the principal point coordinates (w//2, h//2), and the near and far clipping planes (0.2, 10000). The focal lengths determine the field of view, 
            # the principal point indicates the center of the projection, and the clipping planes define the range of distances from the camera within which objects are rendered, with objects closer than 0.2 units or farther than 10000 units being clipped out of the scene. 
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            # pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0) sets up the camera view matrix, which defines the position and orientation of the camera in the 3D scene. The first three parameters (0, -10, -8) specify the position of the camera in the world coordinates, indicating that the camera is located at coordinates (0, -10, -8). The next three parameters (0, 0, 0) 
            # define the point in space the camera is looking at, which is the origin in this case. The last three parameters (0, -1, 0) represent the up direction vector, indicating which direction is considered 'up' for the camera, here pointing along the negative y-axis. This setup effectively positions the camera 10 units down and 8 units back from the origin, looking towards the origin with the 'up' direction being downwards in the y-axis, which is unconventional and might be used to achieve a specific orientation or perspective in the rendered scene.
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))
        # Creates a handler for 3D interaction.
        self.handler = pangolin.Handler3D(self.scam)
        

 
        # Creates a display context.
        self.dcam = pangolin.CreateDisplay()
        # Sets the bounds of the display
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        # assigns handler for mouse clicking and stuff, interactive
        self.dcam.SetHandler(self.handler)
        # self.darr = None

        # image
        width, height = 480, 270
        self.dimg = pangolin.Display('image')
        self.dimg.SetBounds(0, height / 768., 0.0, width / 1024., 1024 / 768.)
        self.dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        self.texture = pangolin.GlTexture(width, height, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.image = np.ones((height, width, 3), 'uint8')



    def viewer_refresh(self, q):
        width, height = 480, 270

        # Checks if the current state is None or if the queue is not empty.
        if self.state is None or not q.empty():
            # Gets the latest state from the queue.
            self.state = q.get()
        
        # Clears the color and depth buffers.
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Sets the clear color to white.
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        # Activates the display context with the current camera settings.
        self.dcam.Activate(self.scam)

        # camera trajectory line and color setup
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        # 3d point cloud color setup
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1])
        
        # show image
        if not self.q_image.empty():
            self.image = self.q_image.get()
            if self.image.ndim == 3:
                self.image = self.image[::-1, :, ::-1]
            else:
                self.image = np.repeat(self.image[::-1, :, np.newaxis], 3, axis=2)
            self.image = cv2.resize(self.image, (width, height))
        if True:         
            self.texture.Upload(self.image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            self.dimg.Activate()
            gl.glColor3f(1.0, 1.0, 1.0)
            self.texture.RenderToViewport()

        # Finishes the current frame and swaps the buffers.
        pangolin.FinishFrame()

    
    def display(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            # updating pose
            poses.append(f.pose)

        for p in self.points:
            # updating map points
            pts.append(p.pt)
        
        # updating queue
        self.q.put((np.array(poses), np.array(pts)))

    def display_image(self, ip_image):
        # if self.q is None:
        #     return
        self.q_image.put(ip_image)


class Point(object):
    # A Point is a 3-D point in the world
    # Each point is observed in multiple frames

    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []

        # assigns a unique ID to the point based on the current number of points in the map.
        self.id = len(mapp.points)
        # adds the point instance to the map’s list of points.
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)

