from queue import Queue
import numpy as np
import cv2 as cv

import sdl2
import sdl2.ext


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
