from queue import Queue
import numpy as np
import cv2 as cv
import time
import build.orb_project as orb

def arr_to_keypts(arr):
    """ Convert numpy array to tuple of keypoints """
    keypts = ()
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
        keypts += (kpt,)

    return keypts

def arr_to_desc(arr):
    """ Convert numpy array to list of descriptors """
    descs = []

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
        k11, d11 = self.orb.detectAndCompute(im1, None)
        k22, d22 = self.orb.detectAndCompute(im2, None)
        print("About to extract keypoints and such")
        t0 = time.perf_counter()
        #k1, d1 = orb.extract(im1)
        #k2, d2 = orb.extract(im2)
        t1 = time.perf_counter()
        print(f"Time to get keypts + descriptors: {t1-t0}")
        #k1 = arr_to_keypts(k1)
        #k2 = arr_to_keypts(k2)
        t2 = time.perf_counter()
        print(f"Time to convert incoming arr: {t2-t1}")

        if d11 is None or d22 is None:
            print("No Descriptors fourd")
            return im1
        matches = self.matcher.match(d11, d22)
        matches = sorted(matches, key=lambda x: x.distance)

        im3 = cv.drawMatches(
            im1,
            k11,
            im2,
            k22,
            matches[:10],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        # self.disp_q.put(matches)
        return im3


class Map:
    def __init__(self):
        self.frames = []
        self.points = []


class Frame:
    def __init__(self, mapp, img):
        self.id = len(mapp.frames)  # give self the last id
        self.img = img
        mapp.frames.append(self)  # append itself to the map
