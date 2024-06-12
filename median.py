import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def denoise_block(block):
    return cv2.fastNlMeansDenoising(block, None, 30, 7, 21)

def median_filter_block(block, ksize=3):
    return cv2.medianBlur(block, ksize)

def divide_and_conquer_denoise(image, block_size, denoise_method='median'):
    height, width = image.shape[:2]
    denoised_image = np.zeros_like(image)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if denoise_method == 'median':
                denoised_block = median_filter_block(block, ksize=3)
            else:
                denoised_block = denoise_block(block)
            denoised_image[i:i + block_size, j:j + block_size] = denoised_block
    return denoised_image

def decrease_and_conquer_denoise(image, scale_factor, denoise_method='median'):
    small_img = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    if denoise_method == 'median':
        denoised_small_img = median_filter_block(small_img, ksize=3)
    else:
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

# Menghilangkan noise dengan Median Filtering pada divide and conquer
block_size = 50
denoised_img_divide_and_conquer_median = divide_and_conquer_denoise(gray_noisy_img, block_size, 'median')

# Menghilangkan noise dengan Median Filtering pada decrease and conquer
scale_factor = 0.5
denoised_img_decrease_and_conquer_median = decrease_and_conquer_denoise(gray_noisy_img, scale_factor, 'median')

# Menghitung SSIM untuk setiap metode denoising dengan Median Filtering
ssim_divide_and_conquer_median = ssim(gray_original_img, denoised_img_divide_and_conquer_median)
ssim_decrease_and_conquer_median = ssim(gray_original_img, denoised_img_decrease_and_conquer_median)

# Menampilkan hasil SSIM
print(f"SSIM before denoising: {ssim(gray_original_img, gray_noisy_img) * 100:.2f}%")
print(f"SSIM (Divide and Conquer - Median): {ssim_divide_and_conquer_median * 100:.2f}%")
print(f"SSIM (Decrease and Conquer - Median): {ssim_decrease_and_conquer_median * 100:.2f}%")

# Menyiapkan citra gabungan dengan label
combined_img = np.concatenate((
    np.concatenate((gray_original_img, gray_noisy_img, denoised_img_divide_and_conquer_median), axis=1),
    np.concatenate((denoised_img_decrease_and_conquer_median, denoised_img_divide_and_conquer_median, denoised_img_divide_and_conquer_median), axis=1)
), axis=0)

# Menambahkan label pada gambar dengan ukuran teks yang lebih kecil
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined_img, 'Original Image', (10, 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Noisy Image', (combined_img.shape[1] // 3 + 10, 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Denoised (Divide and Conquer - Median)', (combined_img.shape[1] // 3 * 2 + 10, 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Denoised (Decrease and Conquer - Median)', (10, combined_img.shape[0] // 2 + 30), font, 0.7, (255, 255, 255), 1)
cv2.putText(combined_img, 'Denoised (Median Filter)', (combined_img.shape[1] // 3 + 10, combined_img.shape[0] // 2 + 30), font, 0.7, (255, 255, 255), 1)

# Menampilkan citra gabungan
cv2.imshow('Combined Images', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
