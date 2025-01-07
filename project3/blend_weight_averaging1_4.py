import cv2
import numpy as np

img1 = cv2.imread('./save_sittingroom_wrapl.png')
img2 = cv2.imread('./save_sittingroom_r.png')  

def blend_images(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    rows, cols, _ = img1.shape
    alpha_mask1 = np.tile(np.linspace(1, 0, cols), (rows, 1)).astype(np.float32) # alpha mask
    alpha_mask2 = np.tile(np.linspace(0, 1, cols), (rows, 1)).astype(np.float32)

    img1_blended = cv2.merge([alpha_mask1, alpha_mask1, alpha_mask1]) * img1
    img2_blended = cv2.merge([alpha_mask2, alpha_mask2, alpha_mask2]) * img2

    blended_image = img1_blended + img2_blended

    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

result = blend_images(img1, img2)
cv2.imwrite('./result_bedroom.png', result)
cv2.imshow('Blended Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
