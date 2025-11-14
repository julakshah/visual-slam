import numpy as np
import cv2
import os,sys,time,g2o
from triangulation import triangulate
from Camera import denormalize, normalize, Camera
from display import Display
from match_frames import generate_match
from descriptor import Descriptor, Point


F= int(os.getenv("F","500"))
W, H = 1920//2, 1080//2
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
desc_dict = Descriptor()
# if os.getenv("D3D") is not None:
desc_dict.create_viewer()

# disp = None
# if os.getenv("D2D") is not None:
disp = Display(W, H)


def calibrate(image):
    # camera intrinsics...<================Check this
    image = cv2.resize(image, (W,H))
    return image

def generate_SLAM(image):
    image = calibrate(image)
    print("Thisis a test0")
    frame = Camera(desc_dict, image, K)
    if frame.id == 0:
        return
    frame1 = desc_dict.frames[-1]
    frame2 = desc_dict.frames[-2]

    x1,x2,Id = generate_match(frame1,frame2)
    frame1.pose =np.dot(Id,frame2.pose)
    for i,idx in enumerate(x2):
        if frame2.pts[idx] is not None:
            frame2.pts[idx].add_observation(frame1,x1[i])
    # homogeneous 3-D coords
    print("Thisis a test1")
    pts4d = triangulate(frame1.pose, frame2.pose, frame1.key_pts[x1], frame2.key_pts[x2])
    pts4d /= pts4d[:, 3:]

    pts3d = pts4d[:, :3]
    per_pt_err, mean_err, rms_err = compute_reprojection_error(pts3d,frame2.key_pts[x2],Id,K)
    print(f"Mean reprojection error: {mean_err}, RMS error: {rms_err}")

    unmatched_points = np.array([frame1.pts[i] is None for i in x1])
    print("Adding:  %d points" % np.sum(unmatched_points))
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
          continue
        pt = Point(desc_dict, p)
        pt.add_observation(frame1, x1[i])
        pt.add_observation(frame2, x2[i])

    for pt1, pt2 in zip(frame1.key_pts[x1], frame2.key_pts[x2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)
        cv2.circle(image, (u1, v1), color=(0,255,0), radius=1)
        cv2.line(image, (u1, v1), (u2, v2), color=(255, 255,0))

    # 2-D display
    if disp is not None:
        disp.display2D(image)
    # 3-D display
    desc_dict.display()


def compute_reprojection_error(points_3d, points_2d_obs, Rt, K):
    """
    points_3d: (N, 3) world coordinates
    points_2d_obs: (N, 2) observed pixel coords
    Rt: (4,4) transform ([R, t],[0,0,0,1])
    K: (3,3) camera intrinsic matrix
    """
    R = Rt[:3,:3] # get rotation
    t = Rt[:3,3] # get translation

    # map world coords to camera coords
    X_cam = (R @ points_3d.T + t.reshape(3, 1)).T

    # Normalize
    x = X_cam[:, 0] / X_cam[:, 2]
    y = X_cam[:, 1] / X_cam[:, 2]

    # Apply intrinsics
    #fx, fy = K[0, 0], K[1, 1]
    #cx, cy = K[0, 2], K[1, 2]

    #u_pred = fx * x + cx
    #v_pred = fy * y + cy
    #pts_2d_pred = np.stack([u_pred, v_pred], axis=1)
    pts_2d_pred = np.stack([x,y],axis=1)

    # error per each point
    diffs = points_2d_obs - pts_2d_pred
    per_point_err = np.linalg.norm(diffs, axis=1)

    mean_err = per_point_err.mean()
    rms_err = np.sqrt((per_point_err ** 2).mean())

    return per_point_err, mean_err, rms_err


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s takes in .mp4 as an arg" %sys.argv[0])
        exit(-1)
    print("Thisis a test-1")
    test= Display(W,H)
    # Use static frames 
    # num_frames = 10000
    # for i in range(num_frames):
    #     print(f"Starting frame {i}")
    #     num = str(i).zfill(4)
    #     img_path = f"../sequence_02/0{num}.jpg"
    #     frame = cv2.imread(img_path)
    #     generate_SLAM(frame)
    
    print("Thisis a test-2")
    cap = cv2.VideoCapture(sys.argv[1]) # Can try Realtime(highly unlikely though)
    while cap.isOpened():
        ret, frame = cap.read()
        print("Thisis a test-3")
        frame1 = cv2.resize(frame, (720,400)) #Resizing the original window
        if ret == True:
          print("Thisis a test")
          cv2.imshow("Frame",frame1)    
          if cv2.waitKey(1) & 0xFF == ord('q'):   #Quit Condition
              break
          generate_SLAM(frame)
        else:
          break
    # cap.release() 
    cv2.destroyAllWindows()