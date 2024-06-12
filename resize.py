from PIL import Image


def resize_and_save(image_path, target_size, output_path):
    # Open the image file
    image = Image.open(image_path)

    # Resize the image
    width, height = image.size
    new_size = (int(width * target_size), int(height * target_size))
    resized_image = image.resize(new_size)

    # Save the resized image
    resized_image.save(output_path)


# Usage:
image_paths = [
    "tc/real/tc1.jpg",
    "tc/real/tc2.jpg",
    "tc/real/tc3.jpg",
    "tc/nois/tc1.jpg",
    "tc/nois/tc2.jpg",
    "tc/nois/tc3.jpg",
]

target_sizes = [0.5, 1, 2, 0.5, 1, 2]
output_paths = [
    "tc/resized/real_tc1.jpg",
    "tc/resized/real_tc2.jpg",
    "tc/resized/real_tc3.jpg",
    "tc/resized/noisy_tc1.jpg",
    "tc/resized/noisy_tc2.jpg",
    "tc/resized/noisy_tc3.jpg",
]
count = 1
for image_path, target_size, output_path in zip(
    image_paths, target_sizes, output_paths
):
    resize_and_save(image_path, target_size, output_path)
