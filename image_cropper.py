from PIL import Image
import os

class ImageCropper:
    def __init__(self, input_folder='main_screenshots', output_folder='processed_screenshots'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.index = 0
        self.crop_percentages = {
            'top': 0.185,
            'bottom': 0.035,
            'left': 0.27,
            'right': 0.015
        }
    
    def setup_output_folder(self):
        """Создает выходную папку, если она не существует"""
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            return True
        except OSError as e:
            raise OSError(f"Ошибка при создании папки {self.output_folder}: {e}")
    
    def is_valid_image(self, filename):
        """Проверяет, является ли файл изображением с допустимым расширением"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))
    
    def calculate_crop_coordinates(self, width, height):
        """Вычисляет координаты для обрезки на основе размеров изображения"""
        left = int(width * self.crop_percentages['left'])
        top = int(height * self.crop_percentages['top'])
        right = int(width * (1 - self.crop_percentages['right']))
        bottom = int(height * (1 - self.crop_percentages['bottom']))
        return (left, top, right, bottom)
    
    def process_image(self, image_path):
        """Обрабатывает одно изображение"""
        try:
            image = Image.open(image_path)
            width, height = image.size
            crop_coords = self.calculate_crop_coordinates(width, height)
            cropped_image = image.crop(crop_coords)
            self._save_image(cropped_image)
            self.index += 1
            return True
        except Exception as e:
            raise Exception(f"Ошибка при обработке изображения {image_path}: {e}")
    
    def _save_image(self, image):
        """Сохраняет обработанное изображение"""
        try:
            output_filename = f"img{self.index}.png"
            output_path = os.path.join(self.output_folder, output_filename)
            image.save(output_path, format='png', quality=100)
        except Exception as e:
            raise Exception(f"Ошибка при сохранении изображения {output_path}: {e}")
    
    def crop_all_images(self):
        """Обрабатывает все изображения в входной папке"""
        processed_count = 0
        
        try:
            self.setup_output_folder()
            
            for filename in os.listdir(self.input_folder):
                if self.is_valid_image(filename):
                    image_path = os.path.join(self.input_folder, filename)
                    if self.process_image(image_path):
                        processed_count += 1
            
            return processed_count
            
        except Exception as e:
            return f"Ошибка: {str(e)}"

def main():
    cropper = ImageCropper()
    result = cropper.crop_all_images()
    
    if isinstance(result, int):
        print(f"Успешно обработано {result} изображений")
    else:
        print(result)

if __name__ == "__main__":
    main()