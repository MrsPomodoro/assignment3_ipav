# the task - use frequency domain image processing to eliminate the artifact
# from the image
#
# The image file provided (MRI_scan_with_artifact.png) contains a
# frequency-dependent artefact. Write a Python script that removes the artifact
# by doing frequency domain image filtering.
# Apply what you learned in the lectures to pick the correct method (filter
# type) and solve the task.
# Some things to consider:
#   - You will have to manually inspect the frequency spectrum to localize the
#     artifact frequencies, therefore apply what you learned about how to
#     visualize the frequency spectrum best
#   - It might make sense to shift the frequency spectrum before designing
#     the filter
#   - Apply what you learned about preventing ringing artifacts in the
#     reconstructed image
#
# You are asked to visualize these things (preferable in a subplot
# configuration):
#   - The original image
#   - The frequency spectrum
#   - The altered frequency spectrum after filter application (you might just
#     update the above spectrum visualization)
#   - The reconstructed image (result image) after filtering and after
#     back-transformation with the IFT
#
# =============================================================================


import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('data/MRI_scan_with_artifact.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# FFT – convert to frequency domain
imagefft = np.fft.fft2(image)

imagefft_shifted = np.fft.fftshift(imagefft)      # shift zero frequency to center

# show the magnitude spectrum to manually identify artifact frequencies based on  example16
img_magnitude = np.log(1 + np.abs(imagefft_shifted))

plt.imshow(np.log(img_magnitude+1), cmap='gray') 
plt.title('Frequency spectrum')
plt.show()


