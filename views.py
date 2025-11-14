import numpy as np
import cv2 as cv
import time
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import build.orb_project as orb
from third_party.triangulation import triangulate # triangulate pts in space
from third_party.cameraFrame import normalize, denormalize # used for camera intrinsics calc
from third_party.descriptor import Descriptor, Point

def arr_to_keypts(arr):
    """
    Convert numpy array to list of keypoints 
    We pass keypoints via Pybind as a (N,7) numpy arr
    We want a list of KeyPoint objects

    Args:
        arr: (N,7) Numpy array of floats
    Returns:
        keypts: list[cv2.KeyPoint()] of converted keypts
    """
    keypts = []
    for row in arr:
        # each row is a single keypoint
        if len(row) < 7:
            print("Skipping keypoint with wrong dimensions")
            continue
        kpt = cv.KeyPoint()
        kpt.pt = (row[0],row[1])
        kpt.size = row[2]
        kpt.angle = row[3]
        kpt.response = row[4]
        kpt.octave = int(row[5])
        kpt.class_id = int(row[6])
        keypts.append(kpt)

    return keypts

class Vid:
    """
    Class to run visual odom every frame, hold persistent parameters
    """
    def __init__(self, mapp, cap, disp_queue):
        """
        Initialize Vid object
        Args:
            mapp (Descriptor): object to hold persistent map and pose data for plotting 
                (defined in third party file)
            cap (cv2.VideoCapture): video capture argument to get frames from
            disp_queue (Queue): queue to pass annotated images for viewing with SDL
        """
        # Camera intrinsic matrix K --- ballpark nums
        Fx = 500
        Fy = 500
        cx = 1920//2
        cy = 1280//2
        self.K = np.asarray([[Fx,0,cx//2],[0,Fy,cy//2],[0,0,1]])

        # init ORB algorithm
        self.orb = cv.ORB_create(nfeatures=5000)
        # init Brute Force matcher for
        # cross check disallows kNN matching, which we're currently using
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        self.mapp = mapp
        self.disp_q = disp_queue  # queue to send results to
        self.cap = cap  # the VideoCapture object for the sequency of images

        #self.desc_dict = Descriptor() # holds global state, num frames
        self.mapp.create_viewer()

    def run(self):
        while True:
            # Compute image matches
            ret, img = self.cap.read()
            frame = Frame(self.mapp, img, self.orb, self.K)

            if frame.id == 0:
                continue

            # do feature matching
            good_f1, good_f2, Rt, img_with_lines = self.feature_matching(
                self.mapp.frames[-1], self.mapp.frames[-2]
            )

            self.disp_q.put(img_with_lines)
            print("Put img with lines!")

            # Pose update --- apply transformation to previous pose
            self.mapp.frames[-1].pose = np.dot(Rt, self.mapp.frames[-2].pose)

            print(f"Transform: {Rt}")

            #for i,idx in enumerate(good_f2):
            #    if self.mapp.frames[-2].pts[idx] is not None:
            #        self.mapp.frames[-2].pts[idx].add_observation(self.mapp.frames[-1],good_f1[i])
            #print(f"About to do pts4d: {self.mapp.frames[-1].keypts}, {type(good_f2[0])}")
            kp1 = np.asarray([self.mapp.frames[-1].keypt_coords[i] for i in good_f1])
            kp2 = np.asarray([self.mapp.frames[-2].keypt_coords[i] for i in good_f2])
            pts4d = triangulate(self.mapp.frames[-1].pose, self.mapp.frames[-2].pose, kp1, kp2)
            pts4d /= pts4d[:, 3:]

            pts3d = pts4d[:,:3]
            per_pt_err, mean_err, rms_err = compute_reprojection_error(pts3d,kp2,Rt,self.K)
            print(f"Mean reprojection error: {mean_err}, RMS error: {rms_err}")

            unmatched_points = np.array([self.mapp.frames[-1].pts[i] is None for i in good_f1])
            print(f"Num unmatched pts located: {len(unmatched_points)}")
            good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points
            print(f"How many good points to add to the drawing? {sum([int(x) for x in good_pts4d])}")
            #print(f"Good pts: {good_pts4d}")

            # Add points to known pts to draw
            for i,p in enumerate(pts4d):
                if not good_pts4d[i]:
                    continue
                #print("good!")
                pt = Point(self.mapp, p)
                pt.add_observation(self.mapp.frames[-1], good_f1[i])
                pt.add_observation(self.mapp.frames[-2], good_f2[i])

            self.mapp.display()

    def feature_matching(self, frame1, frame2, filter_matches=True):
        """ Match features between two frames and return filtered keypts and transform """
        
        k1 = frame1.keypts
        k2 = frame2.keypts
        d1 = frame1.descs
        d2 = frame2.descs
        
        if not filter_matches:
            matches = self.matcher.match(d1, d2)
            matches = sorted(matches, key=lambda x: x.distance)
            im3 = cv.drawMatches(
                frame1.img,
                k1,
                frame2.img,
                k2,
                matches[:10],
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            # self.disp_q.put(matches)
            return im3
        else:
            # Only choose matches that are sufficiently better than the second best (Lowe's ratio)
            matches = self.matcher.knnMatch(d1, d2, 2)
            filter_scale = 0.55 # 1 accepts everything, 0 rejects everything
            good_f1 = []
            good_f2 = []
            ret = []
            for m1,m2 in matches:
                if m1.distance < m2.distance * filter_scale:
                    good_f1.append(m1.queryIdx)
                    good_f2.append(m2.trainIdx)
                    ret.append((k1[m1.queryIdx].pt,k2[m1.trainIdx].pt))
            
            print(f"Num filtered matches by Lowe's: {len(set(ret))}")
            ret = np.array(ret)
            good_f1 = np.array(good_f1)
            good_f2 = np.array(good_f2)

            model, pts_rs = ransac((ret[:,0], ret[:,1]),FundamentalMatrixTransform,min_samples=8,residual_threshold=1,max_trials=1000)
            print(f"Num pts after RANSAC removes outliers: {len(pts_rs)}")

            # Get transform from fundamental matrix

            # Convert to essential matrix using intrinsic matrix
            E = self.K.T @ model.params @ self.K
            print(f"Fundamental Matrix = {model.params}\n")
            print(f"Essential Matrix: {E}\n\n")
            W = np.asmatrix([[0,-1,0],[1,0,0],[0,0,1]]) # used for conversion after svd
            U, d, Vt = np.linalg.svd(E) # do SVD on essential matrix
            if np.linalg.det(Vt) < 0:
                Vt *= -1
            R = np.dot(np.dot(U,W),Vt) # one of two solutions to rotation matrix
            if np.sum(R.diagonal()) < 0: # ensure sum of diagonal is positive, else use other sol
                R = np.dot(np.dot(U, W.T),Vt)
            t = U[:,2] # get translation vector from U
            #print("det(R) =", np.linalg.det(R))

            # Pose update matrix (transform between frames)
            Rt = np.eye(4)
            Rt[:3,:3] = R
            Rt[:3,3] = t

            # Return simpler matches and pose update (consistent with earlier code for now)
            matches_simple = self.matcher.match(d1, d2)
            matches_simple = sorted(matches_simple, key=lambda x: x.distance)
            im3 = cv.drawMatches(
                frame1.img,
                k1,
                frame2.img,
                k2,
                matches_simple[:10],
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            #print(f"Transformation: {Rt}")
            return good_f1[pts_rs], good_f2[pts_rs], Rt, im3

def compute_reprojection_error(points_3d, points_2d_obs, Rt, K):
    """
    points_3d: (N, 3) world coordinates
    points_2d_obs: (N, 2) observed pixel coords
    Rt: (4,4) transform ([R, t],[0,0,0,1])
    K: (3,3) camera intrinsic matrix
    """
    Rt = np.linalg.inv(Rt)
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
    pts_2d_pred = np.stack([x, y], axis=1)

    print("Sample obs vs pred:")
    for i in range(min(5, len(points_2d_obs))):
        print("obs:", points_2d_obs[i], "pred:", pts_2d_pred[i])

    # error per each point
    diffs = points_2d_obs - pts_2d_pred
    per_point_err = np.linalg.norm(diffs, axis=1)

    mean_err = per_point_err.mean()
    rms_err = np.sqrt((per_point_err ** 2).mean())

    return per_point_err, mean_err, rms_err


def feature_extraction(frame, orb_alg):
    """ Extract keypoints and descriptors from an image and store them in the Frame object """
    print("About to extract keypoints and such")
    t0 = time.perf_counter()
    k1, d1 = orb_alg.detectAndCompute(frame.img, None)
    #k1, d1 = orb.extract(frame.img)
    t1 = time.perf_counter()
    print(f"Time to get keypts + descriptors: {t1-t0}")
    #k1 = arr_to_keypts(k1)
    t2 = time.perf_counter()
    print(f"Time to convert incoming arr: {t2-t1}")

    if d1 is None:
        print("No Descriptors fourd")

    frame.keypts = k1
    frame.descs = d1

class Map:
    def __init__(self):
        self.frames = []
        self.points = []


class Frame:
    def __init__(self, mapp, img, orb_alg, K):
        self.id = len(mapp.frames)  # give self the last id
        self.img = img
        self.K = K
        self.pose = np.eye(4)
        feature_extraction(self, orb_alg) # writes to object attrs directly, no return
        self.keypt_coords = np.array([(k.pt[0], k.pt[1]) for k in self.keypts])
        #print(self.keypt_coords)
        self.keypt_coords = normalize(np.linalg.inv(K),self.keypt_coords)
        #self.keypt_coords = np.array([(k.pt[0], k.pt[1]) for k in self.keypts])
        self.pts = [None]*len(self.keypts)
        mapp.frames.append(self)  # append itself to the map
