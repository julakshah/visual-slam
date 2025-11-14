import numpy as np
import cv2
import build.orb_project as orb

def testing():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    for _ in range(50):
        k1, d1 = orb.extract(img)
        print(k1.shape, d1.shape)

if __name__ == "__main__":
    testing()
