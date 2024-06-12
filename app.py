import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6 import QtCore, QtGui

import sys

class ImageDenoiser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Denoiser')
        self.layout = QVBoxLayout()

        self.original_img_label = QLabel()
        self.original_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.original_img_label)

        self.noisy_img_label = QLabel()
        self.noisy_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.noisy_img_label)

        self.load_original_button = QPushButton('Load Original Image')
        self.load_original_button.clicked.connect(self.load_original_image)
        self.layout.addWidget(self.load_original_button)

        self.load_noisy_button = QPushButton('Load Noisy Image')
        self.load_noisy_button.clicked.connect(self.load_noisy_image)
        self.layout.addWidget(self.load_noisy_button)

        self.process_button = QPushButton('Process and Save')
        self.process_button.clicked.connect(self.process_and_save)
        self.layout.addWidget(self.process_button)

        self.setLayout(self.layout)
        self.original_img_path = None
        self.noisy_img_path = None

    def load_original_image(self):
        file_dialog = QFileDialog()
        self.original_img_path, _ = file_dialog.getOpenFileName(self, 'Open Original Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        pixmap = QPixmap(self.original_img_path)
        self.original_img_label.setPixmap(pixmap)

    def load_noisy_image(self):
        file_dialog = QFileDialog()
        self.noisy_img_path, _ = file_dialog.getOpenFileName(self, 'Open Noisy Image', '', 'Image Files (*.png *.jpg *.jpeg)')
        pixmap = QPixmap(self.noisy_img_path)
        self.noisy_img_label.setPixmap(pixmap)

    def process_and_save(self):
        if self.original_img_path and self.noisy_img_path:
            original_img = cv2.imread(self.original_img_path)
            gray_original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

            noisy_img = cv2.imread(self.noisy_img_path)
            gray_noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)

            denoised_img = cv2.GaussianBlur(gray_noisy_img, (5, 5), 0)

            # Save the denoised image
            save_path, _ = QFileDialog().getSaveFileName(self, 'Save Image', '', 'Image Files (*.png *.jpg *.jpeg)')
            if save_path:
                cv2.imwrite(save_path, denoised_img)
                print('Image saved successfully.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageDenoiser()
    window.show()
    sys.exit(app.exec())
