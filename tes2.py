import cv2

def divide_and_conquer_image(img):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build the Gaussian pyramid
    pyramid = [gray_img]
    while True:
        downsampled = cv2.pyrDown(pyramid[-1])
        if downsampled.shape[0] < 30 or downsampled.shape[1] < 30:
            break
        pyramid.append(downsampled)

    # Apply denoising using bilateral filter on each level of the pyramid
    for level, img in enumerate(pyramid[::-1]):
        denoised_img = cv2.bilateralFilter(img, 9, 75, 75)
        pyramid[len(pyramid) - level - 1] = denoised_img

    # Apply sharpening using unsharp masking
    for level, img in enumerate(pyramid):
        blurred_img = cv2.GaussianBlur(img, (0, 0), 3)
        sharp_img = cv2.addWeighted(img, 1.5, blurred_img, -0.5, 0)
        pyramid[level] = sharp_img

    # Reconstruct the enhanced image from the pyramid
    reconstructed_img = pyramid[0]
    for img in pyramid[1:]:
        reconstructed_img = cv2.pyrUp(reconstructed_img)
        reconstructed_img = cv2.resize(reconstructed_img, (img.shape[1], img.shape[0]))
        reconstructed_img += img

    # Apply contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(reconstructed_img)

    return enhanced_img

# Load the image
input_img = cv2.imread('nois.jpg')

# Apply the divide-and-conquer framework for image enhancement
result_img = divide_and_conquer_image(input_img)

# Display the original and enhanced images
cv2.imshow('Original Image', input_img)
cv2.imshow('Enhanced Image', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
