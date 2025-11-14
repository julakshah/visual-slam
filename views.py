from queue import Queue
from time import sleep
import numpy as np
import cv2 as cv
import yaml

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
        # Note: previous frame is 1, current frame is 2
        while True:
            # Compute image matches
            ret, img = self.cap.read()
            # don't compute if there is not frame
            if not ret:
                continue

            frame = Frame(self.mapp, img)
            if frame.id == 0:
                continue

            # do feature matching and create image to display
            im3, pts1, pts2 = self.feature_matching(
                self.mapp.frames[-2].img, self.mapp.frames[-1].img
            )

            self.disp_q.put(im3)

            # Make sure pts1 & pts2 are valid to continue algorithm
            if (
                pts1 is None
                or pts2 is None
                or np.shape(pts1)[0] < 9  # can't get FundMat with < 9 points
                or np.shape(pts2)[0] < 9
            ):
                continue

            # get R, t from get_essential_matrix
            R, t, pts1, pts2 = self.get_frame_transformation(pts1, pts2)

            # transform pose2 into world frame using R, t for triangulation
            T_world_1 = self.mapp.frames[-2].pose
            T_1_2 = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
            T_world_2 = T_1_2 @ T_world_1
            frame.pose = T_world_2

            # do triangulation between frames
            pose1 = self.mapp.frames[-2].pose
            pose2 = self.mapp.frames[-1].pose
            self.triangulate(pose1, pose2, pts1, pts2)

    def feature_matching(self, im1, im2):
        # ensure grayscale for ORB
        if im1.ndim == 3:
            im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        else:
            im1_gray = im1
        if im2.ndim == 3:
            im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        else:
            im2_gray = im2

        k1, d1 = self.orb.detectAndCompute(im1_gray, None)
        k2, d2 = self.orb.detectAndCompute(im2_gray, None)

        if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
            # no features
            # return a display image (draw nothing) and None pts
            return im1, None, None

        matches = self.matcher.match(d1, d2)
        matches = sorted(matches, key=lambda x: x.distance)

        im3 = cv.drawMatches(
            im1,
            k1,
            im2,
            k2,
            matches[:50],  # draw more to see more info
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # build arrays only from valid matches
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches])

        return im3, pts1, pts2

    def get_frame_transformation(self, pts1, pts2):
        # find fundamental matrix (in pixels)
        F, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 3.0)

        if F is None or inliers is None:
            return np.eye(3), np.zeros((3, 1)), None, None

        # compute essential
        E = K.T @ F @ K

        # enforce E singular values (two equal, one zero)
        U, S, Vt = np.linalg.svd(E)
        S_corrected = [(S[0] + S[1]) / 2.0, (S[0] + S[1]) / 2.0, 0.0]
        E = U @ np.diag(S_corrected) @ Vt

        # build boolean mask of inliers
        mask_bool = inliers.ravel() == 1
        pts1_inliers = pts1[mask_bool]
        pts2_inliers = pts2[mask_bool]

        if pts1_inliers.shape[0] < 5:
            # not enough inliers
            return np.eye(3), np.zeros((3, 1)), None, None

        # recoverPose returns inliers too; we pass K and pixel points
        _, R, t, pose_mask = cv.recoverPose(E, pts1_inliers, pts2_inliers, K)

        # normalize translation (direction only)
        t = t / (np.linalg.norm(t) + 1e-9)

        return R, t, pts1_inliers, pts2_inliers

    def triangulate(self, pose1, pose2, pts1, pts2):
        if pts1 is None or pts2 is None or pts1.shape[0] == 0:
            return

        # Prepare homogeneous output
        pts3d = np.zeros((pts1.shape[0], 4))

        # Projection matrices: P = K * [R | t]
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]

        for i, (p1, p2) in enumerate(zip(pts1, pts2)):
            u1, v1 = p1
            u2, v2 = p2

            A = np.zeros((4, 4))
            A[0] = u1 * P1[2] - P1[0]
            A[1] = v1 * P1[2] - P1[1]
            A[2] = u2 * P2[2] - P2[0]
            A[3] = v2 * P2[2] - P2[1]

            _, _, vt = np.linalg.svd(A)
            X_hom = vt[-1]  # last row is smallest singular vector
            pts3d[i] = X_hom / (X_hom[3] + 1e-12)  # normalize w to avoid huge numbers

        # Convert to Cartesian
        pts3d_cartesian = pts3d[:, :3]  # already normalized above

        # compute depth in camera frames: z = R * X + t
        z1 = (pose1[:3, :3] @ pts3d_cartesian.T + pose1[:3, 3:4])[2, :]
        z2 = (pose2[:3, :3] @ pts3d_cartesian.T + pose2[:3, 3:4])[2, :]

        mask = (z1 > 0) & (z2 > 0) & (z1 < 100) & (z2 < 100)

        good_points = pts3d[mask]

        if good_points.shape[0] > 0:
            # store homogeneous points
            self.mapp.points.append(good_points)

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

    def add_ones(self, pts):
        """
        Helper function to add a column of ones to a 2D array (homogeneous
        coordinates).
        """
        return np.hstack([pts, np.ones((pts.shape[0], 1))])


class Map:
    def __init__(self, size, q):
        self.img_size = size  # int tuple (W, H)
        self.disp_q = q
        self.frames = []
        self.points = []
        self.max_coord = 500

    def run(self):
        while True:
            img = self.compute_map_img()
            self.disp_q.put(img)

            # sleep for a bit to decrease compute overhead
            sleep(0.1)
            print()

    def compute_map_img(self):
        if self.points != []:
            all_points = np.vstack(self.points)

            # Convert from homogeneous coordinates (divide by w)
            points_3d = all_points[:, :3] / all_points[:, 3:4]

            # Extract x, z coordinates
            x = points_3d[:, 0]
            z = points_3d[:, 2]

            # Map to pixel coordinates using fixed scale (X-Z plane, top-down view)
            pixel_x = (
                (x + self.max_coord) / (2 * self.max_coord) * (self.img_size[0] - 1)
            ).astype(int)
            pixel_y = (
                (self.max_coord - z) / (2 * self.max_coord) * (self.img_size[1] - 1)
            ).astype(int)

            # Clip to image boundaries
            pixel_x = np.clip(pixel_x, 0, self.img_size[0] - 1)
            pixel_y = np.clip(pixel_y, 0, self.img_size[1] - 1)

            # Create black image
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

            # Draw white points where keypoints are
            img[pixel_y, pixel_x] = [255, 255, 255]

            self.disp_q.put(img)
            return img
        else:
            # Return empty image if no points
            empty_img = np.zeros(
                (self.img_size[1], self.img_size[0], 3), dtype=np.uint8
            )
            self.disp_q.put(empty_img)
            return empty_img


class Frame:
    def __init__(self, mapp, img):
        self.id = len(mapp.frames)  # give self the last id
        self.img = img
        self.pose = np.identity(4)
        mapp.frames.append(self)  # append itself to the map
