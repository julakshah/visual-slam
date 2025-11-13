from queue import Queue
import numpy as np
import cv2 as cv
import time
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
import build.orb_project as orb

def arr_to_keypts(arr):
    """ Convert numpy array to list of keypoints """
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

def arr_to_desc(arr):
    """ Convert numpy array to list of descriptors """
    descs = []

class Vid:
    def __init__(self, mapp, cap, disp_queue):
        # init ORB algorithm
        self.orb = cv.ORB_create(nfeatures=5000)
        # init Brute Force matcher for
        # cross check disallows kNN matching, which we're currently using
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

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
            good_f1, good_f2, Rt, img_with_lines = self.feature_matching(
                self.mapp.frames[-2].img, self.mapp.frames[-1].img
            )

            self.disp_q.put(img_with_lines)



    def feature_matching(self, im1, im2, filter_matches=True):
        # args are InputArray image, InputArray mask
        # output is keypoints, descriptors
        print("About to extract keypoints and such")
        t0 = time.perf_counter()
        k1, d1 = self.orb.detectAndCompute(im1, None)
        k2, d2 = self.orb.detectAndCompute(im2, None)
        #k1, d1 = orb.extract(im1)
        #k2, d2 = orb.extract(im2)
        t1 = time.perf_counter()
        print(f"Time to get keypts + descriptors: {t1-t0}")
        #k1 = arr_to_keypts(k1)
        #k2 = arr_to_keypts(k2)
        t2 = time.perf_counter()
        print(f"Time to convert incoming arr: {t2-t1}")

        if d1 is None or d2 is None:
            print("No Descriptors fourd")
            return im1
        
        if not filter_matches:
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
            # self.disp_q.put(matches)
            return im3
        else:
            # Only choose matches that are sufficiently better than the second best (Lowe's ratio)
            matches = self.matcher.knnMatch(d1, d2, 2)
            filter_scale = 0.95 # 1 accepts everything, 0 rejects everything
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
            W = np.asmatrix([[-1,0,0],[1,0,0],[0,0,1]]) # used for conversion after svd
            U, d, Vt = np.linalg.svd(model.params) # do SVD on fundamental matrix
            if np.linalg.det(Vt) < 0:
                Vt *= -1
            R = np.dot(np.dot(U,W),Vt) # one of two solutions to rotation matrix
            if np.sum(R.diagonal()) < 0: # ensure sum of diagonal is positive, else use other sol
                R = np.dot(np.dot(U, W.T),Vt)
            t = U[:,2] # get translation vector from U

            # Pose update matrix (transform between frames)
            Rt = np.eye(4)
            Rt[:3,:3] = R
            Rt[:3,3] = t

            # Return simpler matches and pose update (consistent with earlier code for now)
            matches_simple = self.matcher.match(d1, d2)
            matches_simple = sorted(matches_simple, key=lambda x: x.distance)
            im3 = cv.drawMatches(
                im1,
                k1,
                im2,
                k2,
                matches_simple[:10],
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            print(f"Transformation: {Rt}")
            return good_f1[pts_rs], good_f2[pts_rs], Rt, im3
            





class Map:
    def __init__(self):
        self.frames = []
        self.points = []


class Frame:
    def __init__(self, mapp, img):
        self.id = len(mapp.frames)  # give self the last id
        self.img = img
        mapp.frames.append(self)  # append itself to the map
