""" CS4277/CS5477 Lab 4: Plane Sweep Stereo
See accompanying Jupyter notebook (lab4.ipynb) for instructions.

Name: Chew Kin Whye
Email: e0200920@u.nus.edu
Student ID: A0171350R
NUSNET ID: ____________________

Name2: Kok Jia Xuan
Email2: e0203403@u.nus.edu
Student ID: A0173833B
NUSNET ID2: e0203403
"""

import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage

"""Helper functions: You should not have to touch the following functions.
"""
class Image(object):
    """
    Image class. You might find the following member variables useful:
    - image: RGB image (HxWx3) of dtype np.float64
    - pose_mat: 3x4 Camera extrinsics that transforms points from world to
        camera frame
    """
    def __init__(self, qvec, tvec, name, root_folder=''):
        self.qvec = qvec
        self.tvec = tvec
        self.name = name  # image filename
        self._image = self.load_image(os.path.join(root_folder, name))

        # Extrinsic matrix: Transforms from world to camera frame
        self.pose_mat = self.make_extrinsic(qvec, tvec)

    def __repr__(self):
        return '{}: qvec={}\n tvec={}'.format(
            self.name, self.qvec, self.tvec
        )

    @property
    def image(self):
        return self._image.copy()

    @staticmethod
    def load_image(path):
        """Loads image and converts it to float64"""
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im.astype(np.float64) / 255.0

    @staticmethod
    def make_extrinsic(qvec, tvec):
        """ Make 3x4 camera extrinsic matrix from colmap pose

        Args:
            qvec: Quaternion as per colmap format (q_cv) in the order
                  q_w, q_x, q_y, q_z
            tvec: translation as per colmap format (t_cv)

        Returns:

        """
        rotation = Rotation.from_quat(np.roll(qvec, -1))
        return np.concatenate([rotation.as_dcm(), tvec[:, None]], axis=1)

def write_json(outfile, images, intrinsic_matrix, img_hw):
    """Write metadata to json file.

    Args:
        outfile (str): File to write to
        images (list): List of Images
        intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix
        img_hw (tuple): (image height, image width)
    """

    img_height, img_width = img_hw

    images_meta = []
    for im in images:
        images_meta.append({
            'name': im.name,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
        })

    data = {
        'img_height': img_height,
        'img_width': img_width,
        'K': intrinsic_matrix.tolist(),
        'images': images_meta
    }
    with open(outfile, 'w') as fid:
        json.dump(data, fid, indent=2)


def load_data(root_folder):
    """Loads dataset.

    Args:
        root_folder (str): Path to data folder. Should contain metadata.json

    Returns:
        images, K, img_hw
    """
    print('Loading data from {}...'.format(root_folder))
    with open(os.path.join(root_folder, 'metadata.json')) as fid:
        metadata = json.load(fid)

    images = []
    for im in metadata['images']:
        images.append(Image(np.array(im['qvec']), np.array(im['tvec']),
                            im['name'], root_folder=root_folder))
    img_hw = (metadata['img_height'], metadata['img_width'])
    K = np.array(metadata['K'])

    print('Loaded data containing {} images.'.format(len(images)))
    return images, K, img_hw


def rgb2hex(rgb):
    """Converts color representation into hexadecimal representation for K3D

    Args:
        rgb (np.ndarray): (N, 3) array holding colors

    Returns:
        hex (np.ndarray): array (N, ) of size N, each element indicates the
          color, e.g. 0x0000FF = blue
    """
    rgb_uint = (rgb * 255).astype(np.uint8)
    hex = np.sum(rgb_uint * np.array([[256 ** 2, 256, 1]]),
                 axis=1).astype(np.uint32)
    return hex

"""Functions to be implemented
"""
# Part 1
def compute_relative_pose(cam_pose, ref_pose):
    """Compute relative pose between two cameras

     Args:
        cam_pose (np.ndarray): Extrinsic matrix of camera of interest C_i (3,4).
          Transforms points in world frame to camera frame, i.e.
            x_i = C_i @ x_w  (taking into account homogeneous dimensions)
        ref_pose (np.ndarray): Extrinsic matrix of reference camera C_r (3,4)

    Returns:
        relative_pose (np.ndarray): Relative pose of size (3,4). Should transform 
          points in C_r to C_i, i.e. x_i = M @ x_r

    Prohibited functions:
        Do NOT use np.linalg.inv() or similar functions
    """
    relative_pose = np.zeros((3, 4), dtype=np.float64)

    """ YOUR CODE STARTS HERE """

    Ri = cam_pose[:, 0:3]
    ti = cam_pose[:, -1]
    
    Rref = ref_pose[:, 0:3]
    tref = ref_pose[:, -1]
    
    rel_rotation = np.zeros((3,3))
    rel_translation = np.zeros((3,1))
    
    rel_rotation = Ri.dot(Rref.T)
    rel_translation = ti - Ri.dot(Rref.T).dot(tref)
    
    relative_pose[:, 0:3] = rel_rotation
    relative_pose[:, -1] = rel_translation

    """ YOUR CODE ENDS HERE """
    return relative_pose


def get_plane_sweep_homographies(K, relative_pose, inv_depths):
    """Compute plane sweep homographies, assuming fronto parallel planes w.r.t.
    reference camera

    Args:
        K (np.ndarray): Camera intrinsic matrix (3,3)
        relative_pose (np.ndarray): Relative pose between the two cameras
          of shape (3, 4)
        inv_depths (np.ndarray): Inverse depths to warp of size (D, )

    Returns:
        homographies (D, 3, 3)
    """

    homographies = None

    """ YOUR CODE STARTS HERE """

    D = len(inv_depths)
    homographies = np.zeros((D, 3, 3))
    for i in range(D):
        inv_depth = inv_depths[i]
        K_inv = np.linalg.inv(K)
        line_to_point_array = [[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1],
                              [0, 0, inv_depth]]
        homography = np.matmul(line_to_point_array, K_inv)
        homography = np.matmul(relative_pose, homography)
        homography = np.matmul(K, homography)
        homographies[i] = homography
 
    """ YOUR CODE ENDS HERE """

    return homographies

# Part 2
def computeVariance(dests, mask, pixel, count):
    x = pixel[0]
    y = pixel[1]

    sumRGB = 0
    # For each colour channel
    for i in range(3):
        sum = 0
        for j in range(len(mask)):
            if mask[j] == 1:
                sum += dests[j][x][y][i]
        mean = sum / count
        var = 0
        for j in range(len(mask)):
            if mask[j] == 1:
                var += pow(dests[j][x][y][i] - mean, 2)
        var /= (count - 1)
        sumRGB += var
        
    return sumRGB / 3


def compute_plane_sweep_volume(images, ref_pose, K, inv_depths, img_hw):
    """Compute plane sweep volume, by warping all images to the reference camera
    fronto-parallel planes, before computing the variance for each pixel and
    depth.

    Args:
        images (list[Image]): List of images which contains information about
          the camera extrinsics for each image
        ref_pose (np.ndarray): Reference camera pose
        K (np.ndarray): 3x3 intrinsic matrix (assumed same for all cameras)
        inv_depths (list): List of inverse depths to consider for plane sweep
        img_hw (tuple): tuple containing (H, W), which are the output height
          and width for the plane sweep volume.

    Returns:
        ps_volume (np.ndarray):
          Plane sweep volume of size (D, H, W), with dtype=np.float64, where
          D is len(inv_depths), and (H, W) are the image heights and width
          respectively. Each element should contain the variance of all pixel
          intensities that warp onto it.
        accum_count (np.ndarray):
          Accumulator count of same size as ps_volume, and dtype=np.int32.
          Keeps track of how many images are warped into a certain pixel,
          i.e. the number of pixels used to compute the variance.
        extras (any type):
          Any additional information you might want to keep for part 4.
    """

    D = len(inv_depths)
    H, W = img_hw
    ps_volume = np.zeros((D, H, W), dtype=np.float64)
    accum_count = np.zeros((D, H, W), dtype=np.int32)
    extras = []

    """ YOUR CODE STARTS HERE """

    white = np.full((H,W,3), 255, dtype=np.float64)
    # Here, we calculate the homographies that wrap the reference image to the other images
    # We get the 10 homographies corresponding to 10 images for each depth value
    homographies = [] # for all images
    identity_matrix_d = np.zeros((D, 3, 3))
    identity_matrix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
    for d in range(len(inv_depths)):
        identity_matrix_d[d] = identity_matrix
    
    for i in range(len(images)):
        if i == 4:
            # need identity matrices for all depths d identity matrices
            homographies.append(identity_matrix_d)            
            
        else:
            # Compute relative pose between images[i] and reference
            rel_pose = compute_relative_pose(ref_pose, images[i].pose_mat)
            
            # get homographies to warp reference onto image (for each d)
            homographies.append(get_plane_sweep_homographies(K, rel_pose, inv_depths)) # numImages x d x 3 x 3
    
    for j in range(len(inv_depths)):
        # compute all warped images at this depth
        # warp each image (src) onto reference view (dest)
        dests = []
        check = []
        for k in range(len(images)): 
            dests.append(cv2.warpPerspective(images[k].image, homographies[k][j], (W,H))) # numImages of images
            check.append(cv2.warpPerspective(white, homographies[k][j], (W,H))) # numImages of images

        for x in range(H):
            for y in range(W):
                pixel = (x,y)
                count = 0
                mask = np.zeros(len(images))
                for index in range(len(images)):
                    # if check == white, pixel is valid
                    if (check[index][x][y][0] == 255):
                        count += 1
                        mask[index] = 1
                if (count == 0 or count == 1):
                    ps_volume[j][x][y] = 0
                else:
                    ps_volume[j][x][y] = computeVariance(dests, mask, pixel, count)
                accum_count[j][x][y] = count

    """ YOUR CODE ENDS HERE """

    return ps_volume, accum_count, extras

def compute_depths(ps_volume, inv_depths):
    """Computes inverse depth map from plane sweep volume as the
    argmin over plane sweep volume variances.

    Args:
        ps_volume (np.ndarray): Plane sweep volume of size (D, H, W) from
          compute_plane_sweep_volume()
        inv_depths (np.ndarray): List of depths considered in the plane
          sweeping (D,)

    Returns:
        inv_depth_image (np.ndarray): inverse-depth estimate (H, W)
    """

    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)

    """ YOUR CODE STARTS HERE """

    D = len(ps_volume)
    H = len(ps_volume[0])
    W = len(ps_volume[0][0])
    
    # ps_volume: D x H x W
    # inv_depths: D
    # return H x W
    
    # for each pixel, go through all possible d, 
    # take minimum
    for x in range(H):
        for y in range(W):
            minimum = -1
            for d in range(D):
                if minimum == -1 or ps_volume[d][x][y] < minimum:
                    minimum = ps_volume[d][x][y]
            inv_depth_image[x][y] = minimum
    
    """ YOUR CODE ENDS HERE """

    return inv_depth_image


# Part 3
def unproject_depth_map(image, inv_depth_image, K, mask=None):
    """Converts the depth map into points by unprojecting depth map into 3D

    Note: You will also need to implement the case where no mask is provided

    Args:
        image (np.ndarray): Image bitmap (H, W, 3)
        inv_depth_image (np.ndarray): Inverse depth image (H, W)
        K (np.ndarray): 3x3 Camera intrinsics
        mask (np.ndarray): Optional mask of size (H, W) and dtype=np.bool.

    Returns:
        xyz (np.ndarray): Nx3 coordinates of points, dtype=np.float64.
        rgb (np.ndarray): Nx3 RGB colors, where rgb[i, :] is the (Red,Green,Blue)
          colors for the points at position xyz[i, :]. Should be in the range
          [0, 1] and have dtype=np.float64.
    """

    xyz = np.zeros([0, 3], dtype=np.float64)
    rgb = np.zeros([0, 3], dtype=np.float64)  # values should be within (0, 1)

    """ YOUR CODE STARTS HERE """

    """ YOUR CODE ENDS HERE """

    return xyz, rgb


# Part 4
def post_process(ps_volume, inv_depth, accum_count, extras):
    """Post processes the plane sweep volume and compute a mask to indicate
    which pixels have confident estimates of the depth

    Args:
        ps_volume: Plane sweep volume from compute_plane_sweep_volume()
          of size (D, H, W)
        inv_depths (List[float]): List of depths considered in the plane
          sweeping
        accum_count: Accumulator count from compute_plane_sweep_volume(), which
          can be used to indicate which pixels are not observed by many other
          images.
        extras: Extra variables from compute_plane_sweep_volume() in Part 2

    Returns:
        inv_depth_image: Denoised Inverse depth image (similar to compute_depths)
        mask: np.ndarray of size (H, W) and dtype np.bool.
          Pixels with values TRUE indicate valid pixels.
    """

    mask = np.ones(ps_volume.shape[1:], dtype=np.bool)
    inv_depth_filtered = inv_depth.copy()

    """ YOUR CODE STARTS HERE """
    
    """ YOUR CODE ENDS HERE """

    return inv_depth_filtered, mask
