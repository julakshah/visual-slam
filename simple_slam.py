import numpy as np
import cv2

from point_map import Display, Map, Point
from feature_extraction import Frame, triangulate, denormalize, match_frames, IRt, add_ones
import numpy as np


def process_frame(img,display,map_in,w,h,k):
    img = cv2.resize(img, (w, h))
    frame = Frame(map_in, img, k)
    if frame.id == 0:
        return
 
    # previous frame f2 to the current frame f1.
    f1 = map_in.frames[-1]
    f2 = map_in.frames[-2]
 
    idx1, idx2, Rt = match_frames(f1, f2)
     
    # X_f1 = E * X_f2, f2 is in world coordinate frame, multiplying that with
    # Rt transforms the f2 pose wrt the f1 coordinate frame
    f1.pose = np.dot(Rt, f2.pose)
 
 
    # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
    # returns an array of size (n, 4), n = feature points
    print(f"Pose 1: {f1.pose}, pose 2: {f2.pose}")
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
 
 
    # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates
    pts4d /= pts4d[:, 3:]
 
 
    # Reject points without enough "Parallax" and points behind the camera
    # returns, A boolean array indicating which points satisfy both criteria.
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
 
    for i, p in enumerate(pts4d):
        # If the point is not good (i.e., good_pts4d[i] is False), 
        # the loop skips the current iteration and moves to the next point.
        if not good_pts4d[i]:
            continue
        pt = Point(map_in, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)
 
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(k, pt1)
        u2, v2 = denormalize(k, pt2)
 
        cv2.circle(img, (u1,v1), 3, (0,255,0))
        cv2.line(img, (u1,v1), (u2, v2), (255,0,0))
 
    # 2-D display
    display.paint(img)
 
    # 3-D display
    map_in.display()
 
if __name__== "__main__":
    ### Camera intrinsics
    # define principal point offset or optical center coordinates
    W, H = 1920//2, 1080//2
    
    # define focus length
    F = 270
    
    # define intrinsic matrix and inverse of that
    K = np.array(([F, 0, W//2], [0,F,H//2], [0, 0, 1]))
    Kinv = np.linalg.inv(K)
    
    # image display initialization
    display = Display(W, H)
    
    # initialize a map
    mapp = Map()
    mapp.create_viewer()

    #cap = cv2.VideoCapture("test.mp4")
    num_frames = 1000
    for i in range(num_frames):
        print(f"Starting frame {i}")
        num = str(i).zfill(4)
        img_path = f"sequence_02/0{num}.jpg"
        frame = cv2.imread(img_path)
        process_frame(frame,display=display,map_in=mapp,w=W,h=H,k=K)
    #while cap.isOpened():
    #    ret, frame = cap.read()
    #    if ret == True:
    #        process_frame(frame)
    #    else:
    #        break