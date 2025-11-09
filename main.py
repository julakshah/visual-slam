import cv2
from display import Display
from extractor import Frame, denormalize, match_frames, add_ones
import numpy as np
from pointmap import Map, Point

## Define camera instrinsics
# W/H centers
W, H = 1920 // 2, 1080 // 2

# Focal length?
F = 270

# Place intrinsic into a matrix (intrinsic matrix)
K = np.array(([F, 0, W // 2], [0, F, H // 2], [0, 0, 1]))
Kinv = np.linalg.inv(K)

# initialize display object
display = Display(W, H)

# Init map object
mapp = Map()
# Init map viewer
mapp.create_viewer()


def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)

    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])

    pts4d /= pts4d[:, 3:]

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1, v1), 3, (0, 255, 0))
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))

    display.paint(img)

    mapp.display()


if __name__ == "__main__":
    # Init the image capture
    cap = cv2.VideoCapture("./v-slam-dataset/test_vid.mp4")
    print(f"cap is {cap}")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
