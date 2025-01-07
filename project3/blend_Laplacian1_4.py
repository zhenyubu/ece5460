import cv2
import numpy as np

def build_gaussian_pyramid(img, levels):
    gaussian_pyramid = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        gaussian_pyramid.append(img)
    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        next_level = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], next_level)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # The smallest level remains the same
    return laplacian_pyramid

def blend_pyramids(laplacianA, laplacianB, mask_pyramid):
    blended_pyramid = []
    for la, lb, mask in zip(laplacianA, laplacianB, mask_pyramid):
        # Expand the mask to have three channels
        mask_3ch = cv2.merge([mask, mask, mask])
        blended = la * mask_3ch + lb * (1.0 - mask_3ch)
        blended_pyramid.append(blended)
    return blended_pyramid

def reconstruct_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = cv2.add(img, pyramid[i])
    return img

def laplacian_pyramid_blending(img1, img2, mask, levels=6):
    # Step 1: Build Gaussian pyramids for the images and the mask
    gpA = build_gaussian_pyramid(img1, levels)
    gpB = build_gaussian_pyramid(img2, levels)
    gpM = build_gaussian_pyramid(mask, levels)

    # Step 2: Build Laplacian pyramids for the images
    lpA = build_laplacian_pyramid(gpA)
    lpB = build_laplacian_pyramid(gpB)

    # Step 3: Blend each level of the Laplacian pyramids using the mask pyramid
    blended_pyramid = blend_pyramids(lpA, lpB, gpM)

    # Step 4: Reconstruct the image from the blended pyramid
    blended_img = reconstruct_from_pyramid(blended_pyramid)
    return blended_img


img1 = cv2.imread('./save_sittingroom_wrapl.png').astype(np.float32) / 255.0
img2 = cv2.imread('./save_sittingroom_r.png').astype(np.float32) / 255.0

rows, cols, _ = img1.shape
mask = np.zeros((rows, cols), dtype=np.float32)
mask[:, :cols // 2] = 1 
mask = cv2.GaussianBlur(mask, (21, 21), 0) 

blended_img = laplacian_pyramid_blending(img1, img2, mask)

blended_img = (blended_img * 255).astype(np.uint8)
cv2.imwrite('./Laplacian_sittingroom_blending.png', blended_img)  
cv2.imshow('Laplacian Pyramid Blending', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
