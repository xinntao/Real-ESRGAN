import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, brightness=0, contrast=1.0):
        self.brightness = brightness if brightness is not None else 0
        self.contrast = contrast if contrast is not None else 1.0

    def process(self, image):
        image = cv2.addWeighted(image, self.contrast, image, 0, self.brightness)
        return image







