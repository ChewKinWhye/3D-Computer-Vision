""" CS4277/CS5477 Lab 1: Fun with Homographies.
See accompanying Jupyter notebook (lab1.ipynb) for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X

Name2: <Name of second member, if any>
Email2: <username>@u.nus.edu
Student ID: A0123456X
"""

import cv2
import numpy as np
import math
from math import floor, ceil
import random


np.set_printoptions(precision=6)
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)


"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def draw_matches(im1, im2, im1_pts, im2_pts, inlier_mask=None):
    """Generates a image line correspondences

    Args:
        im1 (np.ndarray): Image 1
        im2 (np.ndarray): Image 2
        im1_pts (np.ndarray): Nx2 array containing points in image 1
        im2_pts (np.ndarray): Nx2 array containing corresponding points in
          image 2
        inlier_mask (np.ndarray): If provided, inlier correspondences marked
          with True will be drawn in green, others will be in red.

    Returns:

    """
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2

    canvas = np.zeros((canvas_height, canvas_width, 3), im1.dtype)
    canvas[:height1, :width1, :] = im1
    canvas[:height2, width1:width1+width2, :] = im2

    im2_pts_adj = im2_pts.copy()
    im2_pts_adj[:, 0] += width1

    if inlier_mask is None:
        inlier_mask = np.ones(im1_pts.shape[0], dtype=np.bool)

    # Converts all to integer for plotting
    im1_pts = im1_pts.astype(np.int32)
    im2_pts_adj = im2_pts_adj.astype(np.int32)

    # Draw points
    all_pts = np.concatenate([im1_pts, im2_pts_adj], axis=0)
    for pt in all_pts:
        cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

    # Draw lines
    for i in range(im1_pts.shape[0]):
        pt1 = tuple(im1_pts[i, :])
        pt2 = tuple(im2_pts_adj[i, :])
        color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas


def matches2pairs(matches, kp1, kp2):
    """Converts OpenCV's DMatch to point pairs

    Args:
        matches (list): List of DMatches from OpenCV's matcher
        kp1 (list): List of cv2.KeyPoint from OpenCV's detector for image 1 (query)
        kp2 (list): List of cv2.KeyPoint from OpenCV's detector for image 2 (train)

    Returns:
        pts1, pts2: Nx2 array containing corresponding coordinates for both images
    """

    pts1, pts2 = [], []
    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.stack(pts1, axis=0)
    pts2 = np.stack(pts2, axis=0)

    return pts1, pts2


"""Functions to be implemented
"""


# Part 1(a)
def to_coordinates(homo_coordinates):
    coordinates = np.zeros((homo_coordinates.shape[0], 2))
    for i in range(homo_coordinates.shape[0]):
        coordinates[i, 0] = homo_coordinates[i, 0] / homo_coordinates[i, 2]
        coordinates[i, 1] = homo_coordinates[i, 1] / homo_coordinates[i, 2]
    return coordinates


def to_homogenous(input_pts):
    to_add = np.ones(len(input_pts)).reshape((1, len(input_pts)))
    homogenous_pts = np.concatenate((input_pts, to_add.T), axis=1)
    return homogenous_pts


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    src = to_homogenous(src)
    transformed = h_matrix.dot(src.T).T
    transformed = to_coordinates(transformed)
    return transformed


def calc_centroid(points):
    x=0
    y=0
    for point in points:
        x += point[0]
        y += point[1]
    return (x/len(points), y/len(points))


def calculateDistance(x, y):
    dist = math.sqrt((x[1] - y[1])**2 + (x[0] - y[0])**2)
    return dist


def calc_S(points):
    centroid = calc_centroid(points)
    total_distance = 0
    for point in points:
        total_distance += calculateDistance(centroid, point)
    mean_distance = total_distance/len(points)
    return math.sqrt(2)/mean_distance


def calc_T(points):
    s = calc_S(points)
    c_x, c_y = calc_centroid(points)
    T = np.array([[s, 0, -s*c_x], [0, s, -s*c_y], [0, 0, 1]])
    return T


def calc_sub_a(src_points, dst_points):
    sub_a = np.zeros((2, 9))
    sub_a[0, 3:6] = -dst_points[2]*src_points
    sub_a[0, 6:9] = dst_points[1]*src_points
    sub_a[1, 0:3] = dst_points[2]*src_points
    sub_a[1, 6:9] = -dst_points[0]*src_points
    return sub_a


def compute_homography(src1, dst1):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    Trans = calc_T(src1)
    Trans_prime = calc_T(dst1)

    scr1_transformed = transform_homography(src1, Trans)
    dst1_transformed = transform_homography(dst1, Trans_prime)

    scr1_transformed = to_homogenous(scr1_transformed)
    dst1_transformed = to_homogenous(dst1_transformed)

    a = np.zeros((len(src1) * 2, 9))
    for i in range(len(src1)):
        a[i * 2:i * 2 + 2] = calc_sub_a(scr1_transformed[i], dst1_transformed[i])

    u, s, vh = np.linalg.svd(a)
    h_dlt = vh[-1]
    h_dlt = h_dlt.reshape((3, 3))
    h_dlt = np.linalg.inv(Trans_prime).dot(h_dlt).dot(Trans)
    h_dlt = h_dlt / h_dlt[-1, -1]

    return h_dlt


# Part 2

def warp_image(template, original, homo):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    """
    original = original.copy()  # deep copy to avoid overwriting the original image

    output_coordinates = np.zeros((2, original.shape[0] * original.shape[1]))
    for i in range(original.shape[0] * original.shape[1]):
        output_coordinates[0, i] = i // original.shape[0]
        output_coordinates[1, i] = i % original.shape[0]
    output_homo = to_homogenous(output_coordinates.T).T
    homo_inv = np.linalg.inv(homo)
    corresponding_coordinates = to_coordinates(homo_inv.dot(output_homo).T).T
    modified = np.array(original)
    for i in range(original.shape[0] * original.shape[1]):
        template_row = int(corresponding_coordinates[1, i])
        template_col = int(corresponding_coordinates[0, i])
        if template_row >= template.shape[0] or template_row < 0 or template_col >= template.shape[1] or template_col < 0:
            continue
        modified_row = int(output_coordinates[1, i])
        modified_col = int(output_coordinates[0, i])
        modified[modified_row, modified_col] = template[template_row, template_col]
    return modified


def warp_images_all(images, h_matrices):
    """Warps all images onto a black canvas.

    Note: We implemented this function for you, but it'll be useful to note
     the necessary steps
     1. Compute the bounds of each of the images (which can be negative)
     2. Computes the necessary size of the canvas
     3. Adjust all the homography matrices to the canvas bounds
     4. Warp images

    Requires:
        transform_homography(), warp_image()

    Args:
        images (List[np.ndarray]): List of images to warp
        h_matrices (List[np.ndarray]): List of homography matrices

    Returns:
        stitched (np.ndarray): Stitched images
    """
    assert len(images) == len(h_matrices) and len(images) > 0
    num_images = len(images)

    corners_transformed = []
    for i in range(num_images):
        h, w = images[i].shape[:2]
        bounds = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        transformed_bounds = transform_homography(bounds, h_matrices[i])
        corners_transformed.append(transformed_bounds)
    corners_transformed = np.concatenate(corners_transformed, axis=0)

    # Compute required canvas size
    min_x, min_y = np.min(corners_transformed, axis=0)
    max_x, max_y = np.max(corners_transformed, axis=0)

    min_x, min_y = floor(min_x), floor(min_y)

    max_x, max_y = ceil(max_x), ceil(max_y)

    canvas = np.zeros((max_y-min_y, max_x-min_x, 3), images[0].dtype)

    for i in range(num_images):
        # adjust homography matrices
        trans_mat = np.array([[1.0, 0.0, -min_x],
                              [0.0, 1.0, -min_y],
                              [0.0, 0.0, 1.0]], h_matrices[i].dtype)
        h_adjusted = trans_mat @ h_matrices[i]

        # Warp
        canvas = warp_image(images[i], canvas, h_adjusted)

    return canvas


# Part 3
def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)

    for i in range(src.shape[0]):
        homography_inv = np.linalg.inv(homography)
        error_1 = pow(calculateDistance(src[i], transform_homography(dst[i].reshape(1, 2), homography_inv).reshape(2)), 2)
        error_2 = pow(calculateDistance(dst[i], transform_homography(src[i].reshape(1, 2), homography).reshape(2)), 2)
        total_error = error_1 + error_2
        d[i] = total_error


    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """

    """ YOUR CODE STARTS HERE """
    best_indexes = []
    best_counter = 0
    for i in range(num_tries):
        counter = 0
        set_of_src = []
        set_of_dst = []
        indexes = random.sample(range(0, len(src)), 4)
        for index in indexes:
            set_of_src.append(src[index])
            set_of_dst.append(dst[index])

        h_hypothesis = compute_homography(set_of_src, set_of_dst)
        errors = compute_homography_error(src, dst, h_hypothesis)
        for i, error in enumerate(errors):
            if error < thresh:
                counter += 1
        if counter > best_counter:
            best_indexes = indexes
            best_counter = counter
    set_of_src = []
    set_of_dst = []
    for index in best_indexes:
        set_of_src.append(src[index])
        set_of_dst.append(dst[index])
    h_hypothesis = compute_homography(set_of_src, set_of_dst)
    errors = compute_homography_error(src, dst, h_hypothesis)
    set_of_src = []
    set_of_dst = []
    mask = np.zeros(src.shape[0], dtype=np.bool)
    for i, error in enumerate(errors):
        if error < thresh:
            set_of_src.append(src[i])
            set_of_dst.append(dst[i])
            mask[i] = 1
    h_hypothesis = compute_homography(set_of_src, set_of_dst)

    """ YOUR CODE ENDS HERE """

    return h_hypothesis, mask


# Part 4
def concatenate_homographies(pairwise_h_matrices, ref):
    """Transforms pairwise relative transformations to absolute transformations.

    Args:
        pairwise_h_matrices (list): List of N-1 pairwise homographies, the i'th
          matrix maps points in the i'th image to the (i+1)'th image, e.g..
          x_1 = H[0] * x_0
        ref (int): Reference image to warp all images towards.

    Returns:
        abs_h_matrices (list): List of N homographies. abs_H[i] warps points
           in the i'th image to the reference image. abs_H[ref] should be the
           identity transformation.
    """

    abs_h_matrices = []
    num_images = len(pairwise_h_matrices) + 1
    assert ref < num_images

    # abs_h_matrices.append(pairwise_h_matrices[0])
    # abs_h_matrices.append(np.identity(3))
    # abs_h_matrices.append(np.linalg.inv(pairwise_h_matrices[1]))
    # abs_h_matrices.append(np.linalg.inv(pairwise_h_matrices[1]).dot(np.linalg.inv(pairwise_h_matrices[2])))
    for i in range(len(pairwise_h_matrices) + 1):
        to_add = np.identity(3)
        if i == ref - 1:
            abs_h_matrices.append(np.identity(3))
        if i < ref - 1:
            for ii in range(i, ref-1):
                to_add = pairwise_h_matrices[ii].dot(to_add)
            abs_h_matrices.append(to_add)
        if i > ref - 1:
            for ii in range(i, ref - 1, -1):
                to_add = np.linalg.inv(pairwise_h_matrices[ii-1]).dot(to_add)
            abs_h_matrices.append(to_add)

    return abs_h_matrices
