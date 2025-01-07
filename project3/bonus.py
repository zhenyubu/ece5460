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

    return warped_image

# Load images
factor = 1.5
ori_h, ori_w = 73.7 * factor, 92.1 * factor 
img1 = cv2.imread('./bonus.jpeg')
img2 = np.zeros((int(ori_h), int(ori_w), 3), dtype=np.uint8)

# Convert img1 from BGR to RGB for consistency in display and processing
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# # Select points on the image
# plt.imshow(img1)
# plt.title('Click at least 4 points on Image 1')
# image1points = np.array(plt.ginput(n=-1, timeout=0))
# plt.close()


image1points = np.array([[ 52.83333333,  28.86796537],
 [165.31818182,  64.33116883],
 [164.76406926, 192.33116883],
 [ 52.27922078, 230.01082251]])

# Define destination points in the second image
image2points = np.array([
    [0, 0],
    [ori_w, 0],
    [ori_w, ori_h],
    [0, ori_h]
])

# Compute the homography matrix
H = compute_homography(image1points, image2points)
print('Homography matrix H:')
print(H)

# Warp image using computed homography
warped_image = warp_image(img1, img2, H)

# Display and save the result
plt.figure(figsize=(6, 6))
plt.imshow(warped_image)
plt.axis('off')
plt.show()

# Save the warped image in RGB format
plt.imsave('./warped_bonus.png', warped_image)
