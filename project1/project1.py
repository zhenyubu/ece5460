from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def zero_pad(img, pad_height, pad_width):
    if img.ndim == 3:
        h, w, c = img.shape
        pad_result = np.zeros((h + 2 * pad_height, w + 2 * pad_width, c))
        pad_result[pad_height:pad_height + h, pad_width:pad_width + w, :] = img
    else:
        h, w = img.shape
        pad_result = np.zeros((h + 2 * pad_height, w + 2 * pad_width))
        pad_result[pad_height:pad_height + h, pad_width:pad_width + w] = img
    return pad_result

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero 
    (you have to implement zero_pad) . Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image. you have to implement zero_pad

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO
    # you can test your code to match the output of 
    #cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    m, n = kernel.shape
    pad_height, pad_width = m // 2, n // 2
    img_padded = zero_pad(img, pad_height, pad_width)
    
    if img.ndim == 3:
        h, w, ch = img.shape
        correlation_result = np.zeros((h, w, ch))
        for c in range(ch):
            for i in range(h):
                for j in range(w):
                    roi = img_padded[i:i + m, j:j + n, c]
                    correlation_result[i, j, c] = np.sum(roi * kernel)
    else:
        h, w = img.shape
        correlation_result = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                roi = img_padded[i:i + m, j:j + n]
                correlation_result[i, j] = np.sum(roi * kernel)
                
    return np.clip(correlation_result, 0, 255)


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO
    # you can test your code to match the output of
    #cv2.filter2D(img, -1, cv2.flip(kernel, -1), borderType=cv2.BORDER_CONSTANT)
    flipped_kernel = np.flip(kernel, axis=(0, 1))
    convolve_img = np.clip(cross_correlation_2d(img, flipped_kernel), 0, 255)
    return convolve_img

   

def gaussian_blur_kernel_2d(size,sigma):
    '''Returns a square Gaussian blur kernel of the given size  and with the given
    sigma. 

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        size:  The width and height of the  square kernel (must be odd)
        

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO
    # test with cv2.getGaussianKernel(size, sigma)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    sum_total = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            exp_value = - (x ** 2 + y ** 2) / (2 * sigma ** 2)
            kernel[i, j] = np.exp(exp_value)
            sum_total += kernel[i, j]

    kernel /= sum_total
    
    return kernel

    

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO
    kernel = gaussian_blur_kernel_2d(size, sigma)
    filtered_img = convolve_2d(img, kernel)
    
    return filtered_img.astype(img.dtype)
    
    

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO
    
    # Call low pass filter
    low_pass_img = low_pass(img, sigma, size)
     
    high_pass_img = img - low_pass_img

    high_pass_img = high_pass_img - np.min(high_pass_img)

    high_pass_img = np.clip(high_pass_img, 0, 1)
    return high_pass_img.astype(img.dtype)


