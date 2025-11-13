from queue import Queue
import numpy as np
import cv2 as cv
import yaml

import sdl2
import sdl2.ext

# define camera intrinsics
with open("v-slam-dataset/intrinsics.yaml") as f:
    camera = yaml.safe_load(f)

# Create camera matrix K
K = np.array(
    [[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]]
)
Kinv = np.linalg.inv(K)

print(f"K is \n{K}")
print(f"Kinv is \n{Kinv}")


class Vid:
    def __init__(self, mapp, cap, disp_queue):
        # init ORB algorithm
        self.orb = cv.ORB_create(nfeatures=5000)
        # init Brute Force matcher for
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        self.mapp = mapp
        self.disp_q = disp_queue  # queue to send results to
        self.cap = cap  # the VideoCapture object for the sequency of images

    def run(self):
        while True:
            # Compute image matches
            ret, img = self.cap.read()
            frame = Frame(self.mapp, img)

            if frame.id == 0:
                continue

            # do feature matching
            im3 = self.feature_matching(
                self.mapp.frames[-2].img, self.mapp.frames[-1].img
            )

            self.disp_q.put(im3)

    def feature_matching(self, im1, im2):
        # args are InputArray image, InputArray mask
        # output is keypoints, descriptors
        k1, d1 = self.orb.detectAndCompute(im1, None)
        k2, d2 = self.orb.detectAndCompute(im2, None)

        if d1 is None or d2 is None:
            print("No Descriptors fourd")
            return im1
        matches = self.matcher.match(d1, d2)
        matches = sorted(matches, key=lambda x: x.distance)

        im3 = cv.drawMatches(
            im1,
            k1,
            im2,
            k2,
            matches[:10],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        print(f"essential is \n{E}")
        return im3

    def get_essential_matrix(self, k1, k2):
        # Do image formation for 3Dness
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])
        F, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 3.0)
        E = K.T @ F @ K

        mask = [inliers.ravel() == 1]
        pts1_inliers = pts1[mask]
        pts2_inliers = pts1[mask]

        # Decompose essential matrix
        _, R, t, mask = cv.recoverPose(E, pts1_inliers, pts2_inliers, K)

        return R, t

    def triangulate(pose1, pose2, pts1, pts2):
        """
        Use linear algebrea to triangulate the position of the points in 3D
        space. These points should automatically be mapped into the world frame.
          pose1: 3x4 matrix translating world -> camera1
          pose2: 3x4 matrix translating world -> camera2
          pts1: the 2D coordinates of the points in pose1
          pts2: the 2D coordinates of the poitns in pose2
        """
        # variable to store triangulated poitns
        pts3d = np.zeros((pts1.shape[0], 4))

        # compute projection matrices
        P1 = K @ self.fast_pose_inverse(pose1)
        P2 = K @ self.fast_pose_inverse(pose2)

        for i, p in enumerate(zip(self.add_ones(pts1), self.add_ones(pts2)))
          u1 = p[0][0]; v1 = p[0][1];
          u2 = p[1][0]; v2 = p[1][1];

          A = np.zeros((4,4))
          A[0] = u1 * P1[3] - P1[1]
          A[1] = u1 * P1[3] - P1[2]
          A[2] = u1 * P2[3] - P2[1]
          A[3] = u1 * P2[3] - P2[2]

          _, _, vt = np.linalg.svd(A)

          pts3d[i] = vt[3]

        return pts3d


    def fast_pose_inverse(self, pose):
        """
        uses orthogonal property of rotation matrices to speed up inversion.
        Args:
          pose: a 3x4 matrix [R|t]
        Returns:
          pose_inv: a 3x4 matrix inversion of the input pose
        """

        R = pose[:3, :3]
        t = pose[:3, 3]

        pose_inv = np.zeros((3, 4))
        pose_inv[:, :3] = R.T
        pose_inv[:, 3] = -R.T @ t

        return pose_inv

    def add_ones(pts):
        """
        Helper function to add a column of ones to a 2D array (homogeneous
        coordinates).
        """
        return np.hstack([pts, np.ones((pts.shape[0], 1))])


class Map:
    def __init__(self):
        self.frames = []
        self.points = []


class Frame:
    def __init__(self, mapp, img):
        self.id = len(mapp.frames)  # give self the last id
        self.img = img
        mapp.frames.append(self)  # append itself to the map
