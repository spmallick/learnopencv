from multiprocessing import Process, Queue
import numpy as np
import cv2
# import pangolin
# import OpenGL.GL as gl

# Global map // 3D map visualization using pangolin
class Map(object):
    def __init__(self):
        self.frames = [] # camera frames [means camera pose]
        self.points = [] # 3D points of map
        self.frame_pts = None
    
    def display(self):

        poses, pts = [], []
        for f in self.frames:
            # updating pose
            poses.append(f.pose)

        for p in self.points:
            # updating map points
            pts.append(p)
        



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
        # print((self.pt))
        mapp.points.append(self.pt[:3])
        # mapp.frame_pts = self

    def add_observation(self, frame, idx):
        # Frame is the frame class
        self.frames.append(frame)
        self.idxs.append(idx)

