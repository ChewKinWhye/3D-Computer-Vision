"""
Name: Chew Kin Whye
Email: e0200920@u.nus.edu
Student ID: A0171350R

Name2: Kok Jia Xuan
Email2: e0203403@u.nus.edu
Student ID: A0173833B
"""

import numpy as np
import cv2
from sympy.polys import subresultants_qq_zz
import sympy as sym


"""Helper functions: You should not have to touch the following functions.
"""
def extract_coeff(x1, x2, x3, cos_theta12, cos_theta23, cos_theta13, d12, d23, d13):
    """
    Extract coefficients of a polynomial

    Args:
        x1, x2, x3: symbols representing the unknown camera-object distance
        cos_theta12, cos_theta23, cos_theta13: cos values of the inter-point angles
        d12, d23, d13: square of inter-point distances

    Returns:
        a: the coefficients of the polynomial of x1
    """
    f12 = x1 ** 2 + x2 ** 2 - 2 * x1 * x2 * cos_theta12 - d12
    f23 = x2 ** 2 + x3 ** 2 - 2 * x2 * x3 * cos_theta23 - d23
    f13 = x1 ** 2 + x3 ** 2 - 2 * x1 * x3 * cos_theta13 - d13
    matrix = subresultants_qq_zz.sylvester(f23, f13, x3)
    f12_ = matrix.det()
    f1 = subresultants_qq_zz.sylvester(f12, f12_, x2).det()
    a1 = f1.func(*[term for term in f1.args if not term.free_symbols])
    a2 = f1.coeff(x1 ** 2)
    a3 = f1.coeff(x1 ** 4)
    a4 = f1.coeff(x1 ** 6)
    a5 = f1.coeff(x1 ** 8)
    a = np.array([a1, a2, a3, a4, a5])
    return a



def icp(points_s, points_t):
    """
    Estimate the rotation and translation using icp algorithm

    Args:
        points_s : 10 x 3 array containing 3d points in the world coordinate
        points_t : 10 x 3 array containing 3d points in the camera coordinate

    Returns:
        r: rotation matrix of the camera
        t: translation of the camera
    """
    us = np.mean(points_s, axis=0, keepdims=True)
    ut = np.mean(points_t, axis=0, keepdims=True)
    points_s_center = points_s - us
    points_t_center = points_t - ut
    w = np.dot(points_s_center.T, points_t_center)
    u, s, vt = np.linalg.svd(w)
    r = vt.T.dot(u.T)
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T.dot(u.T)
    t = ut.T - np.dot(r, us.T)
    return r, t

def reconstruct_3d(X, K, points2d):
    """
    Reconstruct the 3d points from camera-point distance

    Args:
        X: a list containing camera-object distances for all points
        K: intrinsics of camera
        points2d: 10x1x3 array containing 2d coordinates of points in the homogeneous coordinate

    Returns:
        points3d_c: 3d coordinates of all points in the camera coordinate
    """
    points3d_c = []
    for i in range(len(X)):
        points3d_c.append(X[i] * np.dot(np.linalg.inv(K), points2d[i].T))
    points3d_c = np.hstack(points3d_c)
    return points3d_c


def visualize(r, t, points3d, points2d, K):
    """
    Visualize reprojections of all 3d points in the image and compare with ground truth

    Args:
        r: rotation matrix of the camera
        t: tranlation of the camera
        points3d:  10x3 array containing 3d coordinates of points in the world coordinate
        points3d:  10x2 array containing ground truth 2d coordinates of points in the image space
    """
    scale = 0.2
    img = cv2.imread('data/img_id4_ud.JPG')
    dim = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    img = cv2.resize(img, dim)
    trans = np.hstack([r, t])
    points3d_homo = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
    points2d_re = np.dot(K, np.dot(trans, points3d_homo.T))
    points2d_re = np.transpose(points2d_re[:2, :]/points2d_re[2:3, :])

    for j in range(points2d.shape[0]):
        cv2.circle(img, (int(points2d[j, 0]*scale), int(points2d[j, 1]*scale)), 3,  (0, 0, 255))
        cv2.circle(img, (int(points2d_re[j, 0]*scale), int(points2d_re[j, 1]*scale)), 4,  (255, 0, 0))
    cv2.imshow('img', img)

    cv2.waitKey(0)


def calc_squared_distance(p1, p2):
    dist = pow(p1[0][0] - p2[0][0], 2) + pow(p1[0][1] - p2[0][1], 2) + pow(p1[0][2] - p2[0][2], 2)
    return dist


def calc_cosine_angle(u1, u2, f):
    u1 = u1.flatten()
    u2 = u2.flatten()

    j1 = np.append(u1, np.array([f]), axis=0)
    j2 = np.append(u2, np.array([f]), axis=0)
    j1 = j1 / pow(pow(j1[0], 2) + pow(j1[1], 2) + pow(j1[2], 2), 0.5)
    j2 = j2 / pow(pow(j2[0], 2) + pow(j2[1], 2) + pow(j2[2], 2), 0.5)
    cosine_angle = np.dot(j1, j2)
    return cosine_angle


def to_homo_2d(points2d):
    points2d_ones = np.ones((10, 1, 1))
    points2d_homo = np.concatenate((points2d, points2d_ones), axis=2)
    return points2d_homo


def calc_s(choice, points2d, points3d, f):
    p1 = points3d[choice]
    q1 = points2d[choice]
    A = []
    counter = 0
    # Pick 3 points, with choice included
    for i in range(0, len(points3d)):
        if i == choice:
            continue
        for ii in range(i, len(points3d)):
            if ii == i or ii == choice:
                continue
            p2 = points3d[i]
            p3 = points3d[ii]
            q2 = points2d[i]
            q3 = points2d[ii]

            d12 = calc_squared_distance(p1, p2)
            d23 = calc_squared_distance(p2, p3)
            d13 = calc_squared_distance(p1, p3)

            cos_theta_12 = calc_cosine_angle(q1, q2, f)
            cos_theta_23 = calc_cosine_angle(q2, q3, f)
            cos_theta_13 = calc_cosine_angle(q1, q3, f)
            x1, x2, x3 = sym.symbols('x1, x2, x3')
            a = extract_coeff(x1, x2, x3, cos_theta_12, cos_theta_23, cos_theta_13, d12, d23, d13)
            A.append(list(a))
            counter += 1
    A = np.asarray(A).astype(np.float32)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    t = vh[-1]
    s = pow(((t[1] / t[0] + t[2] / t[1] + t[3] / t[2] + t[4] / t[3]) / 4), 0.5)
    return s


def pnp_algo(K, points2d_og, points3d_og):
    points2d = np.ndarray.copy(points2d_og)
    points3d = np.ndarray.copy(points3d_og)
    """
    Estimate the rotation and translation of camera by using pnp algorithm

    Args:
        K: intrinsics of camera
        points2d: 10x1x2 array containing 2d coordinates of points in the image space
        points2d: 10x1x3 array containing 3d coordinates of points in the world coordinate
    Returns:
        r: 3x3 array representing rotation matrix of the camera
        t: 3x1 array representing translation of the camera
    """
    """YOUR CODE STARTS HERE"""
    f = K[0][0]

    # Covert 2D points to camera centre as reference point
    for i in range(len(points2d)):
        points2d[i][0][0] -= K[0][2]
        points2d[i][0][1] -= K[1][2]

    s_values = []
    for choice in range(0, 10):
        s_values.append(calc_s(choice, points2d, points3d, f))

    # s_values = find_s_values(s_1, points2d, points3d, f)
    # print(s_values)

    for i in range(len(points2d)):
        points2d[i][0][0] += K[0][2]
        points2d[i][0][1] += K[1][2]

    points2d_homo = to_homo_2d(points2d)

    points3d_reconstruct = reconstruct_3d(s_values, K, points2d_homo).astype(float)

    r, t = icp(np.squeeze(points3d), np.squeeze(points3d_reconstruct).T)
    # """YOUR CODE ENDS HERE"""
    return r, t













