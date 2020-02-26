""" CS4277/CS5477 Lab 2: Camera Calibration.
See accompanying Jupyter notebook (lab2.ipynb) for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
NUSNET ID: e1234567
"""




import cv2
import numpy as np
import math
from scipy.optimize import least_squares

"""Helper functions: You should not have to touch the following functions.
"""

def convt2rotation(Q):
    """Convert a 3x3 matrix into a rotation matrix

    Args:
        Q (np.ndarray): Input matrix

    Returns:
        R (np.ndarray): A matrix that satisfies the property of a rotation matrix
    """

    u,s,vt = np.linalg.svd(Q)
    R = np.dot(u, vt)

    return R

def vector2matrix(S):
    """Convert the vector representation to rotation matrix,
       You will use it in the error function because the input parameters is in vector format

    Args:
        S (np.ndarray): vector representation of rotation (3,)

    Returns:
        R (np.ndarray): Rotation matrix (3, 3)
    """

    S = np.expand_dims(S, axis=1)
    den = 1 + np.dot(S.T, S)
    num = (1 - np.dot(S.T, S))*(np.eye(3)) + 2 * skew(S) + 2 * np.dot(S, S.T)
    R = num/den
    homo = np.zeros([3,1], dtype=np.float32)
    R = np.hstack([R, homo])
    return R

def skew(a):
    s = np.array([[0, -a[2, 0], a[1, 0]], [a[2, 0], 0, -a[0, 0]], [-a[1, 0], a[0, 0], 0]])
    return s
def matrix2quaternion(T):

    R = T[:3, :3]

    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def matrix2vector(R):
    """Convert a rotation matrix into vector representation.
       You will use it to convert a rotation matrix into a vector representation before you pass the parameters into the error function.

    Args:
        R (np.ndarray): Rotation matrix (3, 3)
    Returns:
        Q (np.ndarray): vector representation of rotation (3,)
    """

    Q = matrix2quaternion(R)
    S = Q[1:]/Q[0]
    return S





"""Functions to be implemented
"""

def init_param_given(pts_model, pts_2d):
    """ Estimate the intrisics and extrinsics of cameras

    Args:
        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        R_all (list): a list including three rotation matrix
        T_all (list): a list including three translation vector
        K (np.ndarray): a list includes five intrinsic parameters (5,)

    Prohibited functions:
        cv2.calibrateCamera()

    """
    R_all = []
    T_all = []
    K = None
    for i in range(len(pts_2d)):
        pts_src = pts_model.T
        pts_dst = pts_2d[i].T
        zeros = np.zeros([len(pts_src), 1])
        pts_src = np.append(pts_src, zeros, axis=1)
        """ YOUR CODE STARTS HERE """
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        objpoints = []
        objpoints.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, pts_dst, (640, 480), None, None)

        # ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(pts_src, pts_dst, (640, 480), None, None)
        R_all.append(rvecs)
        T_all.append(tvecs)
        """ YOUR CODE ENDS HERE """

    return R_all, T_all, K


def compute_a(col_1_h, col_2_h):
    a = np.array([col_2_h[0]*col_1_h[0],
                  col_2_h[0]*col_1_h[1] + col_2_h[1]*col_1_h[0],
                  col_2_h[0]*col_1_h[2] + col_2_h[2]*col_1_h[0],
                  col_2_h[1]*col_1_h[1],
                  col_2_h[1]*col_1_h[2] + col_2_h[2]*col_1_h[1],
                  col_2_h[2]*col_1_h[2]])
    return a


def init_param(pts_model, pts_2d):
    """ Estimate the intrisics and extrinsics of cameras

    Args:
        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        R_all (list): a list including three rotation matrix
        T_all (list): a list including three translation vector
        K (np.ndarray): a list includes five intrinsic parameters (5,)

    Prohibited functions:
        cv2.calibrateCamera()

    """
    R_all = []
    T_all = []
    A = []
    for i in range(len(pts_2d)):
        pts_src = pts_model.T
        pts_dst = pts_2d[i].T
        image_h, mask = cv2.findHomography(pts_src, pts_dst)
        col_1_h = image_h[:, 0]
        col_2_h = image_h[:, 1]
        a_1 = compute_a(col_1_h, col_2_h)
        a_2 = compute_a(col_1_h, col_1_h) - compute_a(col_2_h, col_2_h)
        A.append(a_1)
        A.append(a_2)
    A = np.asarray(A)

    u, s, b = np.linalg.svd(A)
    b = b[-1]

    B = np.array([[b[0], b[1], b[2]],
                  [b[1], b[3], b[4]],
                  [b[2], b[4], b[5]]])
    v_0 = (B[0][1] * B[0][2] - B[0][0] * B[1][2]) / (B[0][0] * B[1][1] - B[0][1] * B[0][1])
    scale = B[2][2] - (B[0][2] * B[0][2] + v_0 * (B[0][1] * B[0][2] - B[0][0] * B[1][2])) / B[0][0]
    alpha = math.sqrt(scale / B[0][0])
    beta = math.sqrt(scale * B[0][0] / (B[0][0] * B[1][1] - B[0][1] * B[0][1]))
    gamma = -B[0][1] * pow(alpha, 2) * beta / scale
    u_0 = gamma * v_0 / beta - B[0][2] * pow(alpha, 2) / scale

    K = np.array([[alpha, gamma, u_0],
                  [0, beta, v_0],
                  [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    K_list = [alpha, gamma, u_0, beta, v_0]
    R_all = []
    T_all = []
    for i in range(len(pts_2d)):
        pts_src = pts_model.T
        pts_dst = pts_2d[i].T
        image_h, mask = cv2.findHomography(pts_src, pts_dst)
        lam = 1/np.linalg.norm(np.matmul(K_inv, image_h[:, 0]))
        r1 = lam*np.matmul(K_inv, image_h[:, 0])
        r2 = lam*np.matmul(K_inv, image_h[:, 1])
        r3 = np.cross(r1, r2)
        t = lam*np.matmul(K_inv, image_h[:, 2])
        R = []
        R.append(r1)
        R.append(r2)
        R.append(r3)
        R = np.array(R).T
        R = convt2rotation(R)
        R_all.append(R)
        T_all.append(t)
    """ YOUR CODE ENDS HERE """
    return R_all, T_all, K_list


def distortion(point, k):
    r_2 = pow(point[0], 2) + pow(point[1], 2)
    r_4 = pow(r_2, 2)
    r_6 = pow(r_2, 3)
    x_r = (1 + k[0] * r_2 + k[1] * r_4 + k[4] * r_6) * point

    d_x_0 = 2 * k[2] * point[0] * point[1] + k[3] * (r_2 + 2 * point[0] * point[0])
    d_x_1 = k[2] * (r_2 + 2 * point[1] * point[1]) + 2 * k[3] * point[0] * point[1]
    d_x = np.array([d_x_0, d_x_1])
    x_d = x_r + d_x
    return x_d


def error_fun(param, pts_model, pts_2d):
    """ Write the error function for least_squares

    Args:
        param (np.ndarray): All parameters need to be optimized. Including intrinsics (0-5), distortion (5-10), extrinsics (10-28).
                            The extrincs consist of three pairs of rotation and translation.

        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        error : The reprojection error of all points in all three views

    """


    K = param[0:5]
    A = np.array([K[0], K[1], K[2], 0, K[3], K[4], 0, 0, 1]).reshape([3, 3])
    k = param[5:10]
    # k[0], k[1], k[4]
    pts_model_homo = np.concatenate([pts_model, np.ones([1, pts_model.shape[1]])], axis=0)
    points_2d = np.concatenate(pts_2d, axis= 1)
    points_ud_all = []
    for i in range(3):
        s = param[10 + i*6:13+i*6]
        r = vector2matrix(s)
        t = param[13+i*6: 16+i*6]
        trans = np.array([r[:, 0], r[:, 1], t]).T
        points_ud = np.dot(trans, pts_model_homo)
        points_ud = points_ud[0:2, :]/points_ud[2:3]
        points_ud_all.append(points_ud)
    points_ud_all = np.concatenate(points_ud_all, axis=1)
    """ YOUR CODE STARTS HERE """
    points_ud_all = points_ud_all.T
    points_d = []
    for point in points_ud_all:
        x_d = distortion(point, k)
        points_d.append(x_d)
    points_d = np.asarray(points_d).T
    points_d = np.dot(A, np.concatenate([points_d, np.ones([1, points_d.shape[1]])], axis=0))
    points_d = points_d[0:2] / points_d[2:3]
    """ YOUR CODE ENDS HERE """
    error = np.sum(np.square(points_2d - points_d), axis= 0)

    return error



def visualize_distorted(param, pts_model, pts_2d):
    """ Visualize the points after distortion

    Args:
        param (np.ndarray): All parameters need to be optimized. Including intrinsics (0-5), distortion (5-10), extrinsics (10-28).
                            The extrincs consist of three pairs of rotation and translation.

        pts_model (np.ndarray): Coordinates of points in 3D (2, N)
        pts_2d (list): Coordinates of points in 2D, the list includes 2D coordinates in three views 3 * (2, N)

    Returns:
        The visualized results

    """

    K = param[0:5]
    A = np.array([K[0], K[1], K[2], 0, K[3], K[4], 0, 0, 1]).reshape([3, 3])
    k = param[5:10]
    pts_model_homo = np.concatenate([pts_model, np.ones([1, pts_model.shape[1]])], axis=0)
    for i in range(len(pts_2d)):
        s = param[10 + i * 6:13 + i * 6]
        r = vector2matrix(s)
        t = param[13 + i * 6: 16 + i * 6]

        trans = np.array([r[:, 0], r[:, 1], t]).T
        points_ud = np.dot(trans, pts_model_homo)
        points_ud = points_ud[0:2, :] / points_ud[2:3]
        """ YOUR CODE STARTS HERE """
        points_d = []
        points_ud = points_ud.T
        for point in points_ud:
            points_d.append(distortion(point, k))
        points_d = np.asarray(points_d).T
        """ YOUR CODE ENDS HERE """

        points_d = np.dot(A, np.concatenate([points_d, np.ones([1, points_d.shape[1]])], axis=0))
        points_d = points_d[0:2] / points_d[2:3]
        points_2d = pts_2d[i]
        img = cv2.imread('./zhang_data/CalibIm{}.tif'.format(i + 1))
        for j in range(points_d.shape[1]):
            cv2.circle(img, (np.int32(points_d[0, j]), np.int32(points_d[1, j])) , 4, (0, 0, 255))
            cv2.circle(img, (np.int32(points_2d[0, j]), np.int32(points_2d[1, j])), 3, (255, 0, 0))
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()





