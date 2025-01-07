import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def compute_homography(image1points, image2points):
    num_points = image1points.shape[0]
    A = []
    for i in range(num_points):
        x, y = image1points[i][0], image1points[i][1]
        x_prime, y_prime = image2points[i][0], image2points[i][1]

        A.append([-x, -y, -1,  0,  0,  0, x_prime * x, x_prime * y, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A) # SVD to solve the total_least_squares problem
    h = Vt[-1] # smallest eigenvalue
    H = h.reshape(3, 3) # reshape
    H = H / H[2, 2] # Normalize

    return H

def perspective_transform(points, H):
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    transformed_points = H @ homogeneous_points.T  # [3, N]
    transformed_points = transformed_points[:2, :] / transformed_points[2, :] # Normalize
    return transformed_points.T

def warp_image(image, image_b, H):
    h, w = image.shape[:2] # Shape
    h2, w2, _ = image_b.shape
    corners = np.array([ # four corners
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ], dtype=np.float32)

    transformed_corners = perspective_transform(corners, H) # for output

    x_min, x_max = transformed_corners[:, 0].min(), max(transformed_corners[:, 0].max(), w2)
    y_min, y_max = transformed_corners[:, 1].min(), max(transformed_corners[:, 1].max(), h2)
    output_width = int(np.ceil(x_max - x_min))
    output_height = int(np.ceil(y_max - y_min))

    translation_matrix = np.array([ # make sure all the locations are positive
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    H_translated = translation_matrix @ H  
    
    warped_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    x_coords, y_coords = np.meshgrid(np.arange(output_width), np.arange(output_height))
    homogeneous_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()])
    # Reverse
    H_inv = np.linalg.inv(H_translated)
    source_coords = H_inv @ homogeneous_coords
    source_coords /= source_coords[2]  # Norm

    src_x = source_coords[0].reshape(output_height, output_width)
    src_y = source_coords[1].reshape(output_height, output_width)

    map_x = src_x.ravel()
    map_y = src_y.ravel()
    
    # Interpolation for all three RGB channels
    for i in range(3):  
        warped_image[..., i] = map_coordinates(image[..., i], [map_y, map_x], order=1, mode='constant', cval=0).reshape(output_height, output_width)
    wrap_l = warped_image.copy()
    non_wrap_r = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    non_wrap_r[int(-y_min) : int(h2 - y_min), int(-x_min) : int(w2 - x_min)] = img2 # place the non-wraped img2

    # add img2 on it, combine them into full image
    warped_image[int(-y_min) : int(h2 - y_min), int(-x_min) : int(w2 - x_min)] = img2
    # return warped_image, save_img2
    return warped_image, non_wrap_r, wrap_l



img1 = cv2.imread('./sittingroom1.jpeg')
img2 = cv2.imread('./sittingroom2.jpeg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Select points on the image by Mouse
# plt.imshow(img1)
# plt.title('Click at least 4 points on Image 1')
# image1points = np.array(plt.ginput(n=-1, timeout=0))
# plt.close()

# plt.imshow(img2)
# plt.title('Click at least 4 points on Image 2')
# image2points = np.array(plt.ginput(n=-1, timeout=0))
# plt.close()

# For sittingroom
image1points = np.array([[1832.40909091, 1891.13636364],
 [1968.40909091 ,2484.59090909],
 [2784.40909091 ,2571.13636364],
 [3204.77272727 ,2534.04545455],
 [2833.86363636 ,3164.59090909],
 [1548.04545455 ,3943.5       ]])

image2points = [[ 249.86363636 ,1854.04545455],
 [ 398.22727273 ,2472.22727273],
 [1325.5        ,2546.40909091],
 [1708.77272727 ,2546.40909091],
 [1399.68181818 ,3152.22727273],
 [  27.31818182 ,4166.04545455]]

# For bedroom
# image1points = np.array([[1696.40909091 ,1718.04545455],
#  [2920.40909091 ,1619.13636364],
#  [1745.86363636 ,4487.5       ],
#  [2945.13636364 ,4623.5       ],
#  [2067.31818182 ,4042.40909091],
#  [2784.40909091 ,4128.95454545]])

# image2points = np.array([[ 113.86363636 ,1730.40909091],
#  [1473.86363636 ,1767.5       ],
#  [ 175.68181818 ,4808.95454545],
#  [1498.59090909 ,4747.13636364],
#  [ 571.31818182 ,4314.40909091],
#  [1350.22727273 ,4277.31818182]])


# Compute the homography matrix
H = compute_homography(image1points, image2points)
print('Homography matrix H:')
print(H)

# Warp image and show the results
# warped_image, save_img2 = warp_image(img1, img2, H)
warped_image, non_wrap_r, wrap_l = warp_image(img1, img2, H)

plt.figure(figsize=(6, 6))
plt.imshow(warped_image)
plt.axis('off')
plt.show()
plt.imsave('./save_full_sittingroom.png', warped_image)

plt.figure(figsize=(6, 6))
plt.imshow(non_wrap_r)
plt.axis('off')
plt.show()
plt.imsave('./save_sittingroom_r.png', non_wrap_r)


plt.figure(figsize=(6, 6))
plt.imshow(wrap_l)
plt.axis('off')
plt.show()
plt.imsave('./save_sittingroom_wrap_l.png', wrap_l)