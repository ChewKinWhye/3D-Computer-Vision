import numpy as np
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt


"""Helper functions: You should not have to touch the following functions.
"""
def compute_right_epipole(F):

    U, S, V = np.linalg.svd(F.T)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(img1, img2, F, x1, x2, epipole=None, show_epipole=False):
    """
    Visualize epipolar lines in the imame

    Args:
        img1, img2: two images from different views
        F: fundamental matrix
        x1, x2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
    Returns:

    """
    plt.figure()
    plt.imshow(img1)
    for i in range(x1.shape[1]):
      plt.plot(x1[0, i], x1[1, i], 'bo')
      m, n = img1.shape[:2]
      line1 = np.dot(F.T, x2[:, i])
      t = np.linspace(0, n, 100)
      lt1 = np.array([(line1[2] + line1[0] * tt) / (-line1[1]) for tt in t])
      ndx = (lt1 >= 0) & (lt1 < m)
      plt.plot(t[ndx], lt1[ndx], linewidth=2)
    plt.figure()
    plt.imshow(img2)

    for i in range(x2.shape[1]):
      plt.plot(x2[0, i], x2[1, i], 'ro')
      if show_epipole:
        if epipole is None:
          epipole = compute_right_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


      m, n = img2.shape[:2]
      line2 = np.dot(F, x1[:, i])

      t = np.linspace(0, n, 100)
      lt2 = np.array([(line2[2] + line2[0] * tt) / (-line2[1]) for tt in t])

      ndx = (lt2 >= 0) & (lt2 < m)
      plt.plot(t[ndx], lt2[ndx], linewidth=2)
    plt.show()


def compute_essential(data1, data2, K):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
        K: intrinsic matrix of the camera
    Returns:
        E: Essential matrix
    """

    """YOUR CODE STARTS HERE"""
    E, _ = cv2.cv2.findEssentialMat(data1[:2, :].T, data2[:2, :].T, cameraMatrix=K)
    """YOUR CODE ENDS HERE"""

    return E


def decompose_e(E, K, data1, data2):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        E: Essential matrix
        K: intrinsic matrix of the camera
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        trans: 3x4 array representing the transformation matrix
    """
    """YOUR CODE STARTS HERE"""
    _, r, t, _ = cv2.recoverPose(E, data1[:2, :].T, data2[:2, :].T, K)
    trans = np.concatenate([r, t], axis =1)
    """YOUR CODE ENDS HERE"""
    return trans


def compute_fundamental(data1, data2):
    """
    Compute the fundamental matrix from point correspondences

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        F: fundamental matrix
    """

    """YOUR CODE STARTS HERE"""
    F, _ = cv2.findFundamentalMat(data1[:2, :].T, data2[:2, :].T, method = cv2.FM_8POINT)
    """YOUR CODE ENDS HERE"""

    return F











