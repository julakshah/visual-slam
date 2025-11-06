from multiprocessing import Process, Queue
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import threading

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

class Map(object):
    def __init__(self):
        self.frames = [] # camera frames [means camera pose]
        self.points = [] # 3D points of map
        self.state = None # variable to hold current state of the map and cam pose
        self.q = None # A queue for inter-process communication. | q for visualization process
 
    def create_viewer(self):
        print("\n\n\n\n\n\n\n\nCREATE VIEWER\n\n\n\n\n\n")
        self.q = Queue() # q is initialized as a Queue
        p = Process(target=self.viewer_thread, args=(self.q,)) 
        p.daemon = True
        p.start()
 
    def viewer_thread(self, q):
        print("\n\n\n\n\n\n\n\nINITIALIZE VIEWER THREAD\n\n\n\n\n\n")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SLAM Map", width=1280, height=720)

        self.pcd = o3d.geometry.PointCloud()
        self.traj = o3d.geometry.LineSet()

        vis.add_geometry(self.pcd)
        vis.add_geometry(self.traj)

        # Set a reasonable view
        ctr = vis.get_view_control()
        ctr.set_lookat([0,0,0])
        ctr.set_front([0,-1,-0.8])
        ctr.set_up([0,-1,0])
        ctr.set_zoom(0.2)

        while True:
            # non-blocking poll
            print(f"Q but maybe things are empty: {q.get()}")
            while not q.empty():
                poses, pts = q.get()
                #print(f"Poses: {poses}")
                if len(poses) > 0:
                    # Update point cloud
                    self.pcd.points = o3d.utility.Vector3dVector(pts) if len(pts) else o3d.utility.Vector3dVector(np.zeros((0,3)))
                    # Build a simple trajectory as connected line segments between camera centers
                    if len(poses) >= 2:
                        # camera centers are the translation components of inv(T_wc) or directly T_wc[:3,3] depending on convention
                        centers = np.array([T[:3,3] for T in poses])
                        lines = [[i, i+1] for i in range(len(centers)-1)]
                    else:
                        centers, lines = np.zeros((0,3)), []
                    self.traj.points = o3d.utility.Vector3dVector(centers)
                    self.traj.lines  = o3d.utility.Vector2iVector(lines)

                    vis.update_geometry(self.pcd)
                    vis.update_geometry(self.traj)

                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)
     
    def display(self):
        print("\nDISPLAY! yay\n")
        if self.q is None:
            return

        poses = [f.pose for f in self.frames]
        pts = [p.pt for p in self.points]
        print(f"Poses: {poses}, pts: {pts}")
         
        # Update queue
        self.q.put((np.array(poses), np.array(pts)))

class Display:
    def __init__(self, W, H, title="SLAM", is_rgb=False):
        """
        W, H: desired window size
        is_rgb: set True if your incoming frames are RGB (will convert to BGR for imshow)
        """
        self.W, self.H = W, H
        self.title = title
        self.is_rgb = is_rgb

        # Create a resizable window and set its initial size
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.W, self.H)

    def _window_closed(self) -> bool:
        # If the window was closed by the user, this returns < 1
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1

    def paint(self, img):
        print("\n\nPAINT\n\n")
        # Resize to the display size to match your SDL2 behavior
        if (img.shape[1], img.shape[0]) != (self.W, self.H):
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)

        # Convert if needed (cv2.imshow expects BGR for color images)
        if self.is_rgb and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow(self.title, img)

        # Pump window events (1ms). ESC to quit.
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or self._window_closed():  # ESC or window closed
            cv2.destroyWindow(self.title)
            raise SystemExit(0)
