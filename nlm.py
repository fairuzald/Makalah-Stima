import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def denoise_block(block):
    return cv2.fastNlMeansDenoising(block, None, 30, 7, 21)

def divide_and_conquer_denoise(image, block_size):
    height, width = image.shape[:2]
    denoised_image = np.zeros_like(image)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            denoised_block = denoise_block(block)
            denoised_image[i:i + block_size, j:j + block_size] = denoised_block
    return denoised_image

def decrease_and_conquer_denoise(image, scale_factor):
    small_img = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    denoised_small_img = cv2.fastNlMeansDenoising(small_img, None, 30, 7, 21)
    # Sharpen the denoised image
    denoised_img = cv2.resize(denoised_small_img, (image.shape[1], image.shape[0]))
    return denoised_img

# Membaca gambar asli tanpa noise
original_img = cv2.imread('real/8310512013_4db1ab3a79_c.jpg')
gray_original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Membaca gambar noisy
noisy_img = cv2.imread('noisy/noisy_8310512013_4db1ab3a79_c.jpg')
gray_noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)

# Menghilangkan noise dengan Gaussian Blur sebagai baseline
denoised_img_gaussian = cv2.fastNlMeansDenoising(gray_noisy_img, None, 30, 7, 21)

# Menghilangkan noise dengan metode divide and conquer
block_size = 50
denoised_img_divide_and_conquer = divide_and_conquer_denoise(gray_noisy_img, block_size)

# Menghilangkan noise dengan metode decrease and conquer
scale_factor = 0.5
denoised_img_decrease_and_conquer = decrease_and_conquer_denoise(gray_noisy_img, scale_factor)

# Menghitung SSIM untuk setiap metode denoising
ssim_gaussian = ssim(gray_original_img, denoised_img_gaussian)
ssim_divide_and_conquer = ssim(gray_original_img, denoised_img_divide_and_conquer)
ssim_decrease_and_conquer = ssim(gray_original_img, denoised_img_decrease_and_conquer)

# Menampilkan hasil SSIM
print(f"SSIM before denoising: {ssim(gray_original_img, gray_noisy_img) * 100:.2f}%")
print(f"SSIM (Gaussian Blur): {ssim_gaussian * 100:.2f}%")
print(f"SSIM (Divide and Conquer): {ssim_divide_and_conquer * 100:.2f}%")
print(f"SSIM (Decrease and Conquer): {ssim_decrease_and_conquer * 100:.2f}%")

# Menyiapkan citra gabungan dengan label
combined_img = np.concatenate((
    np.concatenate((gray_original_img, gray_noisy_img, denoised_img_divide_and_conquer), axis=1),
    np.concatenate((denoised_img_decrease_and_conquer, denoised_img_gaussian, np.zeros_like(gray_original_img)), axis=1)
), axis=0)

# Menambahkan label pada gambar dengan ukuran teks yang lebih kecil
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined_img, 'Original Image', (10, 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Noisy Image', (combined_img.shape[1] // 3 + 10, 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Denoised (Divide and Conquer)', (combined_img.shape[1] // 3 * 2 + 10, 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Denoised (Decrease and Conquer)', (10, combined_img.shape[0] // 2 + 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Denoised (Gaussian Blur)', (combined_img.shape[1] // 3 + 10, combined_img.shape[0] // 2 + 30), font, 0.7, (255, 255, 255), 1)

# Menampilkan citra gabungan
cv2.imshow('Combined Images', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()