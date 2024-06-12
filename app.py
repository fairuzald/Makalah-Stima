import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
import numpy as np


def divide(image, y, x, kernel_half, kernel_size, height, width):
    if kernel_size == 1:
        return [image[y, x]]

    neighborhood = []
    for i in range(-kernel_half, kernel_half + 1):
        for j in range(-kernel_half, kernel_half + 1):
            ny = min(max(y + i, 0), height - 1)
            nx = min(max(x + j, 0), width - 1)
            neighborhood.append(image[ny, nx])

    return neighborhood


def conquer(neighborhood):
    return np.median(neighborhood)


def combine(result, y, x, median_value):
    result[y, x] = median_value


def parallel_median_filter(image, kernel_size):
    height, width = image.shape[:2]
    kernel_half = kernel_size // 2
    result = np.zeros_like(image)

    def process_pixel(y, x):
        neighborhood = divide(image, y, x, kernel_half, kernel_size, height, width)
        median_value = conquer(neighborhood)
        combine(result, y, x, median_value)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for y in range(height):
            for x in range(width):
                futures.append(executor.submit(process_pixel, y, x))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    return result


def manual_median_filter(image, kernel_size):
    height, width = image.shape[:2]
    kernel_half = kernel_size // 2
    result = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            neighborhood = divide(image, y, x, kernel_half, kernel_size, height, width)
            median_value = conquer(neighborhood)
            combine(result, y, x, median_value)
    return result


def naive_median_filter(image, kernel_size):
    height, width = image.shape[:2]
    kernel_half = kernel_size // 2
    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            neighborhood = []
            for i in range(-kernel_half, kernel_half + 1):
                for j in range(-kernel_half, kernel_half + 1):
                    ny = min(max(y + i, 0), height - 1)
                    nx = min(max(x + j, 0), width - 1)
                    neighborhood.append(image[ny, nx])

            result[y, x] = np.median(neighborhood)

    return result


noisy_images = [
    "tc/resized/noisy_tc1.jpg",
    "tc/resized/noisy_tc2.jpg",
    "tc/resized/noisy_tc3.jpg",
]

real_images = [
    "tc/resized/real_tc1.jpg",
    "tc/resized/real_tc2.jpg",
    "tc/resized/real_tc3.jpg",
]
import time, os


# Baca gambar
def processing(real_images, noisy_images, tc):
    print(f"====================Test Case {tc}=======================")
    real = cv2.imread(real_images, cv2.IMREAD_GRAYSCALE)

    # Ukuran kernel (harus ganjil)
    kernel_size = 3

    # Baca gambar asli
    noise = cv2.imread(noisy_images, cv2.IMREAD_GRAYSCALE)

    # Terapkan filter median tanpa metode divide and conquer
    start_time = time.time()
    filtered_image_naive = naive_median_filter(noise, kernel_size)
    end = time.time()
    print("Time for naive:", end - start_time)

    # Terapkan filter median dengan metode divide and conquer
    start_time = time.time()
    filtered_image_dc = parallel_median_filter(noise, kernel_size)
    end = time.time()
    print("Time for divide and conquer:", end - start_time)

    filtered_image_bawaan_cv = cv2.medianBlur(noise, kernel_size)

    # Hitung nilai SSIM antara gambar asli dan gambar hasil filter median dari kedua pendekatan
    ssim_dc = ssim(real, filtered_image_dc)
    ssim_before = ssim(real, noise)
    print("SSIM before denoising:", ssim_before*100,"%")
    print("SSIM with Divide and Conquer:", ssim_dc * 100, "%")
    print("SSIM with OpenCV Median Filter:", ssim(real, filtered_image_bawaan_cv) * 100, "%")
    base_name = os.path.basename(real_images)
    cv2.imwrite(f"tc/result/dc_{base_name}", filtered_image_dc)
    cv2.imwrite(f"tc/result/naive_{base_name}", filtered_image_naive)
    cv2.imwrite(f"tc/result/cv_{base_name}", filtered_image_bawaan_cv)
    print("=======================================================")

count = 1
for real_images, noisy_images in zip(real_images, noisy_images):
    processing(real_images, noisy_images, count)
    count += 1

# processing("real/8310512013_4db1ab3a79_c.jpg", "noisy/noisy_8310512013_4db1ab3a79_c.jpg", 1)