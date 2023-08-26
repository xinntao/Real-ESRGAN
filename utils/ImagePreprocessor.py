import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, brightness=0, contrast=1.0, scale=1.0):
        self.brightness = brightness if brightness is not None else 0
        self.contrast = contrast if contrast is not None else 1.0
        self.scale = scale if scale is not None else 1.0

    def process(self, image):
        image = cv2.addWeighted(image, self.contrast, image, 0, self.brightness)
        if self.scale != 1.0:
            image = self.scale_image(image)
        return image

    def scale_image(self, image):
        # Calculate the new dimensions after scaling
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)

        # Resize the image to the new dimensions
        return cv2.resize(image, (new_width, new_height))







