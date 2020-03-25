"""
Name: Chew Kin Whye
Email: e0200920@u.nus.edu
Student ID: A0171350R

Name2: Kok Jia Xuan
Email2: e0203403@u.nus.edu
Student ID: A0173833B
"""

import numpy as np
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt
import math


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


def getCentroid(data):
    data = data.T
    size = len(data)
    sum_x, sum_y = 0,0
    for data in data:
        sum_x += data[0]
        sum_y += data[1]
    
    return sum_x / size, sum_y /size


def getDistance(point, centroid):
    return math.sqrt(math.pow(point[0]-centroid[0],2) + math.pow(point[1]-centroid[1],2))


def findS(data, centroid):
    sum_distance = 0
    for i in range(data.shape[1]):
        sum_distance += getDistance((data[0][i],data[1][i]), centroid)
    mean_distance = sum_distance / data.shape[1]
    return math.sqrt(2) / mean_distance


def normalise(data):
    centroid = getCentroid(data)
    s = findS(data, centroid)
    T = np.array([[s, 0, -s * centroid[0]],
                  [0, s, -s * centroid[1]],
                  [0, 0, 1]])
    return T.dot(data), T


def eight_point_algo(data1, data2):
    # data1, data2 normalised
    
    A = np.zeros((data1.shape[1], 9))
    for i in range(data1.shape[1]):
        xprime, yprime = data2[0][i], data2[1][i]
        x, y = data1[0][i], data1[1][i]
        A[i] = np.array([xprime * x, xprime * y, xprime, yprime * x, yprime * y, yprime, x, y, 1])
        
    # To find matrix, take SVD of A
    u, s, v_t = np.linalg.svd(A)
    m = v_t[-1] # last column of v is last row of v_t
    M = m.reshape(3,3)
    return M


def computeLinearTriangulation(data1, data2, P, Pprime):
    X = np.zeros((data1.shape[1], 4)) # transpose it back later
    P1 = P[0]
    P2 = P[1]
    P3 = P[2]    
    Pprime1 = Pprime[0]
    Pprime2 = Pprime[1]
    Pprime3 = Pprime[2]
    
    for i in range(data1.shape[1]):
        x = data1[0][i]
        y = data1[1][i]
        xprime = data2[0][i]
        yprime = data2[1][i]
        
        A = np.zeros((4, 4))
        A[0] = x * P3 - P1
        A[1] = y * P3 - P2
        A[2] = xprime * Pprime3 - Pprime1
        A[3] = yprime * Pprime3 - Pprime2
        
        u, s, v_t = np.linalg.svd(A)
        X[i] = v_t[-1] / v_t[-1][3]
        
    return X.T


def isInFrontOfBothCameras(X, P, Pprime):
    imageX = P.dot(X)
    imageXprime = Pprime.dot(X)
    
    if (imageX[2][0] > 0 and imageXprime[2][0] > 0):
        return True
    
    return False




 
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
    # E, _ = cv2.cv2.findEssentialMat(data1[:2, :].T, data2[:2, :].T, cameraMatrix=K)
    
    # normalise data
    data1 = np.linalg.inv(K).dot(data1)
    data2 = np.linalg.inv(K).dot(data2)
    
    E = eight_point_algo(data1, data2)

    # Take into account singularity constraint of E matrix
    u, s, v_t = np.linalg.svd(E)
    s[0] = (s[0] + s[1])/2
    s[1] = s[0]
    s[2] = 0

    E = u.dot(np.diagflat(s)).dot(v_t)
    
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
    # _, r, t, _ = cv2.recoverPose(E, data1[:2, :].T, data2[:2, :].T, K)
    # trans = np.concatenate([r, t], axis =1)

    u, s, v_t = np.linalg.svd(E) 
    t1 = u[:,-1] # +-
    t2 = -t1
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = u.dot(W).dot(v_t)
    R2 = u.dot(W.T).dot(v_t)
    
    if (np.linalg.det(R1) < 0):
        R1 = -R1
        
    if (np.linalg.det(R2) < 0):
        R2 = -R2
           
    t1 = np.reshape(t1, (-1,1))
    t2 = np.reshape(t2, (-1,1))
    
    ls = []
    trans1 = np.concatenate([R1, t1], axis = 1)
    trans2 = np.concatenate([R1, t2], axis = 1)
    trans3 = np.concatenate([R2, t1], axis = 1)
    trans4 = np.concatenate([R2, t2], axis = 1)
    ls.append(trans1)
    ls.append(trans2)
    ls.append(trans3)
    ls.append(trans4)
    
    numCheck = 0; # should be 1 at the end of the loop
    P1 = K.dot(np.concatenate([np.eye(3), np.zeros((3,1))], axis = 1))
    for i in range(len(ls)):
        P2 = K.dot(ls[i])
        X = computeLinearTriangulation(data1, data2, P1, P2)
        if isInFrontOfBothCameras(X, P1, P2):
            numCheck += 1
            chosenTrans = i
            
    if (numCheck != 1):
        print("something wrong")

    return ls[chosenTrans]

    """YOUR CODE ENDS HERE"""


def compute_fundamental(data1, data2):
    """
    Compute the fundamental matrix from point correspondences

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        F: fundamental matrix
    """

    """YOUR CODE STARTS HERE"""
    # F, _ = cv2.findFundamentalMat(data1[:2, :].T, data2[:2, :].T, method = cv2.FM_8POINT)
   
    # normalise
    data1, T1 = normalise(data1)
    data2, T2 = normalise(data2)
    
    F = eight_point_algo(data1, data2)
    
    # To find F', take SVD of F (Singularity Constraint of F matrix)
    u_of_F, s_of_F, v_of_F_t = np.linalg.svd(F)
    diagonal_of_Fprime = s_of_F
    diagonal_of_Fprime[2] = 0
    Fprime = u_of_F.dot(np.diagflat(diagonal_of_Fprime)).dot(v_of_F_t)
    
    # denormalise
    F = T2.T.dot(Fprime).dot(T1)

    """YOUR CODE ENDS HERE"""

    return F











