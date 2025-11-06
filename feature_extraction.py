import cv2
import numpy as np
import skimage

IRt = np.asmatrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

def add_ones(x):
    # map x to homogeneous coords by appending ones for w
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def triangulate(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))
 
    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
 
    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A to hold the linear equations
        A = np.zeros((4, 4))
 
        # Populate the matrix A with the equations derived from the projection matrices and the points
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # Perform SVD on A
        _, _, vt = np.linalg.svd(A)
 
        # The solution is the last row of V transposed (V^T), corresponding to the smallest singular value
        ret[i] = vt[3]
 
    # Return the 3D points in homogeneous coordinates
    return ret

def extractPose(F):
    W = np.asmatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # SVD on the fundamental matrix
    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0

    # Correct Vt if determinant is negative
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Compute the initial rotation matrix R using U, W, and Vt
    R = np.dot(np.dot(U, W), Vt)

    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    # translation is third column of matrix
    t = U[:,2]

    ret = np.eye(4)

    # map rotation and translation to identity matrix
    ret[:3, :3] = R
    ret[:3, 3] = t

    print(d)
    return ret

def extract(img):
    orb = cv2.ORB_create()
 
    # Detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=-1).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=10)
 
    # Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(img, kps)
 
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def normalize(Kinv, pts):
    # The inverse camera intrinsic matrix 𝐾^(−1) transforms 2D homogeneous points 
    # from pixel coordinates to normalized image coordinates. 
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    # Converts a normalized point to pixel coordinates by applying the 
    # intrinsic camera matrix and normalizing the result.
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def match_frames(f1, f2):
    # The code performs k-nearest neighbors matching on feature descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
 
    # applies Lowe's ratio test to filter out good 
    # matches based on a distance threshold.
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
             
            # Distance test
            # dditional distance test, ensuring that the 
            # Euclidean distance between p1 and p2 is less than 0.1
            if np.linalg.norm((p1-p2)) < 0.1:
                # Keep idxs
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))
                pass
 
 
    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
 
    # Fit matrix
    model, inliers = skimage.measure.ransac((ret[:, 0], 
                            ret[:, 1]), skimage.transform.FundamentalMatrixTransform, 
                            min_samples=8, residual_threshold=0.005, 
                            max_trials=200)
     
    # Ignore outliers
    ret = ret[inliers]
    Rt = extractPose(model.params)
 
    return idx1[inliers], idx2[inliers], Rt

class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K  # Intrinsic camera matrix
        self.Kinv = np.linalg.inv(self.K)  # Inverse of the intrinsic camera matrix
        self.pose = IRt  # Initial pose of the frame (assuming IRt is predefined)
        #self.pose = 0

        self.id = len(mapp.frames)  # Unique ID for the frame based on the current number of frames in the map
        mapp.frames.append(self)  # Add this frame to the map's list of frames

        pts, self.des = extract(img)  # Extract feature points and descriptors from the image
        self.pts = normalize(self.Kinv, pts)  # Normalize the feature points using the inverse intrinsic matrix