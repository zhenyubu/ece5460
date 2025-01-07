import numpy as np
import matplotlib.pyplot as plt

def compute_homography(image1points, image2points):
    num_points = image1points.shape[0]
    A = []
    for i in range(num_points):
        x, y = image1points[i][0], image1points[i][1]
        x_prime, y_prime = image2points[i][0], image2points[i][1]

        A.append([-x, -y, -1,  0,  0,  0, x_prime * x, x_prime * y, x_prime])
        A.append([ 0,  0,  0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A) # SVD to sovle the total_least_squares problem
    h = Vt[-1] # smallest eigen value
    H = h.reshape(3, 3) # reshape
    H = H / H[2, 2] # Normalize

    return H

# Load 
img1 = plt.imread('./room-1.jpeg')
img2 = plt.imread('./room-2.jpeg')

plt.imshow(np.rot90(img1, k=3))
plt.title('Click at least 4 points on Image 1')
image1points = np.array(plt.ginput(n=-1, timeout=0))
plt.close()

plt.imshow(np.rot90(img2, k=3))
plt.title('Click corresponding points on Image 2 in the same order')
image2points = np.array(plt.ginput(n=len(image1points), timeout=0))
plt.close()

# Compute the homography matrix
H = compute_homography(image1points, image2points)
print('Homography matrix H:')
print(H)


