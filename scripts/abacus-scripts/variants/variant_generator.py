import cv2
import os
import random

META_INFO_PATH = r'C:\Users\ianko\Desktop\Projects\AbacusBooks\Real-ESRGAN\datasets\engravings_dataset\meta_info'


class ImageVariantGenerator:
    def __init__(self, image_directory, meta_info_path):
        self.image_directory = image_directory
        self.meta_info_path = meta_info_path

    def create_variants(self, gt_path, lr_path, num_variants):
        full_lr_path = os.path.join(self.image_directory, lr_path)
        lr_image = cv2.imread(full_lr_path)

        image_dir = os.path.dirname(lr_path)

        for i in range(num_variants):
            lr_basename = os.path.splitext(os.path.basename(lr_path))[0]

            variant_image = self.generate_variant(lr_image)

            variant_filename = f"{lr_basename}_{self.get_variant_name()}_{i+1}.png"
            local_variant_path = os.path.join(image_dir, variant_filename)
            variant_path = os.path.join(self.image_directory, local_variant_path)

            cv2.imwrite(variant_path, variant_image)

            with open(self.meta_info_path, 'a') as f:
                f.write(f"{gt_path}, {local_variant_path}\n")

    def generate_variant(self, original_image):
        raise NotImplementedError("Subclasses must implement generate_variant method.")

    def get_variant_name(self):
        raise NotImplementedError("Subclasses must implement get_variant_name method.")


class BrightnessContrastVariantGenerator(ImageVariantGenerator):
    def generate_variant(self, original_image):
        brightness_factor = random.randint(-150, 150)
        contrast_factor = random.randint(-46, 46)
        variant_image = self.set_brightness(original_image, brightness_factor)
        variant_image = self.set_contrast(variant_image, contrast_factor)
        return variant_image

    def set_brightness(self, original_image, brightness):
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        return cv2.addWeighted(original_image, alpha_b, original_image, 0, gamma_b)


    def set_contrast(self, original_image, contrast):
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        return cv2.addWeighted(original_image, alpha_c, original_image, 0, gamma_c)


    def get_variant_name(self):
        return "b_c"
