import numpy as np
import cv2

from point_map import Display, Map, Point
from feature_extraction import Frame, triangulate, denormalize, match_frames, IRt, add_ones
import numpy as np

def reprojection_error_norm(pose, X_w, x_norm_obs):
    """
    pose: 4x4 world->camera
    X_w: (N,3) world points
    x_norm_obs: (N,2) normalized image coords (what you used to estimate F)
    Returns: (N,) reprojection error in normalized coords
    """
    N = X_w.shape[0]
    X_h = np.hstack([X_w, np.ones((N,1))])   # (N,4)
    X_c = (pose @ X_h.T).T                   # (N,4)

    x_norm_pred = X_c[:, :2] / X_c[:, 2:3]   # (N,2)
    err = np.linalg.norm(x_norm_pred - x_norm_obs, axis=1)
    return err

def process_frame(img,display,map_in,w,h,k,reset_vis):
    img = cv2.resize(img, (w, h))
    frame = Frame(map_in, img, k)
    if frame.id == 0:
        return
 
    # previous frame f2 to the current frame f1.
    f1 = map_in.frames[-1]
    f2 = map_in.frames[-2]
 
    idx1, idx2, Rt = match_frames(f1, f2)

    # R = Rt[:3, :3]
    # t = Rt[:3, 3]

    # Rt_cam_new_to_world_if_world_is_old = np.eye(4)
    # Rt_cam_new_to_world_if_world_is_old[:3, :3] = R.T
    # Rt_cam_new_to_world_if_world_is_old[:3, 3] = -R.T @ t

    # f1.pose = f2.pose @ Rt_cam_new_to_world_if_world_is_old
     
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
 
    # 1) Camera center in world:
    R = f1.pose[:3, :3]
    t = f1.pose[:3, 3]
    C = -R.T @ t  # world coords

    # 2) All world points (after dehomogenizing):
    X_w = pts4d[good_pts4d][:, :3]

    # 3) Direction from camera center to each point in world:
    dirs_world = X_w - C[None, :, 0]
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)

    # 4) Observed normalized image directions in this frame:
    x_norm_obs = f1.pts[idx1[good_pts4d]]  # these are what you used to triangulate
    obs_dirs = np.hstack([x_norm_obs, np.ones((x_norm_obs.shape[0], 1))])
    obs_dirs /= np.linalg.norm(obs_dirs, axis=1, keepdims=True)

    # 5) Compare angle between "ray from camera to point" and "observed direction"
    dots = np.sum(dirs_world * (R.T @ obs_dirs.T), axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))

    if angles.size > 0:
        print("Angle between reprojection rays and point rays:")
        print("  mean:", angles.mean(), "deg")
        print("  max :", angles.max(), "deg")
        
    pts_good = pts4d[good_pts4d][:, :3]
    X_w = pts4d[good_pts4d][:, :3]

    idx_good1 = idx1[good_pts4d]
    idx_good2 = idx2[good_pts4d]

    err1 = reprojection_error_norm(f1.pose, X_w, f1.pts[idx_good1])
    err2 = reprojection_error_norm(f2.pose, X_w, f2.pts[idx_good2])

    if err1.size > 0 and err2.size > 0:
        print("Reproj f1: mean =", err1.mean(), "max =", err1.max())
        print("Reproj f2: mean =", err2.mean(), "max =", err2.max())

    # camera centers in world (pose is world->camera)
    def cam_center_from_pose(pose):
        R = pose[:3, :3]
        t = pose[:3, 3]
        return -R.T @ t

    C1 = cam_center_from_pose(f1.pose)
    C2 = cam_center_from_pose(f2.pose)
    baseline = np.linalg.norm(C1 - C2)

    dists = np.linalg.norm(X_w - C1[None, :, 0], axis=1)
    if dists.size > 0:
        print("baseline =", baseline)
        print("depth: min =", dists.min(), "max =", dists.max(), "mean =", dists.mean())
        print("baseline/depth (mean) =", baseline / dists.mean())

    if pts_good.shape[0] > 0:
        print("3D stats:")
        print("  count:", pts_good.shape[0])
        print("  min:", pts_good.min(axis=0))
        print("  max:", pts_good.max(axis=0))
        print("  mean:", pts_good.mean(axis=0))
    else:
        print("No good 3D points this frame")

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
    map_in.display(reset_vis=reset_vis)
 
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
    # mapp.create_viewer()

    #cap = cv2.VideoCapture("test.mp4")
    num_frames = 10000
    for i in range(num_frames):
        print(f"Starting frame {i}")
        num = str(i).zfill(4)
        img_path = f"sequence_02/0{num}.jpg"
        frame = cv2.imread(img_path)
        process_frame(frame,display=display,map_in=mapp,w=W,h=H,k=K,reset_vis=bool(i==10))
    #while cap.isOpened():
    #    ret, frame = cap.read()
    #    if ret == True:
    #        process_frame(frame)
    #    else:
    #        break