import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2
import scipy.signal as ssg
import numpy as np

def lowpass(img, sigma):
    gauss = cv2.getGaussianKernel(ksize=max(int(2*sigma//3), 1), sigma=sigma)
    gauss = gauss @ gauss.T

    r = ssg.convolve2d(img[:, :, 0], gauss, mode='same', boundary="symm")
    g = ssg.convolve2d(img[:, :, 1], gauss, mode='same', boundary="symm")
    b = ssg.convolve2d(img[:, :, 2], gauss, mode='same', boundary="symm")

    blurred = np.stack([r, g, b], axis=2)
    return blurred

def highpass(img, sigma):
    return img - lowpass(img, sigma)

def hybrid_image(im1, im2, sigma1, sigma2):
    im1 = lowpass(im1, sigma1)
    im2 = highpass(im2, sigma2)

    hybrid = im1 + im2
    return hybrid, im1, im2

def make_gray(im):
    return np.stack([im[:, :, 0], im[:, :, 0], im[:, :, 0]], axis=2)

# First load images

# high sf
name = "happy_mad"
im1 = plt.imread('imgs/happy.jpg')/255

# low sf
im2 = plt.imread('imgs/mad.jpg')/255
make_gray(im2)

# Next align images (this code is provided, but may be improved)
im2_aligned, im1_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

sigma1 = 4
sigma2 = 128

hybrid, filtered_im1, filtered_im2 = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.imshow(hybrid)
plt.savefig("outs/" + name + "/hybrid.jpg")
plt.show()

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(filtered_im1[:,:,0])))))
plt.savefig("outs/" + name + "/filt_im1_fft.jpg")
plt.show()

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(filtered_im2[:,:,0])))))
plt.savefig("outs/" + name + "/filt_im2_fft.jpg")
plt.show()

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1[:,:,0])))))
plt.savefig("outs/" + name + "/im1_fft.jpg")
plt.show()

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2[:,:,0])))))
plt.savefig("outs/" + name + "/im2_fft.jpg")
plt.show()

plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid[:,:,0])))))
plt.savefig("outs/" + name + "/hybrid_fft.jpg")
plt.show()

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
pyramids(hybrid, N)