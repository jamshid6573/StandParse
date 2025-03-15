import os
from PIL import Image

class ImageSplitter:
    def __init__(self, input_folder='cropped_screenshots', output_folder='ready_screenshots'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.index = 0
        
    def setup_output_folder(self):
        """Создает выходную папку, если она не существует"""
        os.makedirs(self.output_folder, exist_ok=True)
    
    def is_valid_image(self, filename):
        """Проверяет, является ли файл изображением с допустимым расширением"""
        valid_extensions = ('.png', '.jpg', '.jpeg')
        return filename.lower().endswith(valid_extensions)
    
    def process_image(self, image_path):
        """Обрабатывает одно изображение"""
        screenshot = Image.open(image_path)
        width, height = screenshot.size
        
        # Вычисляем размеры карточки
        card_width = width // 4
        card_height = height // 2
        
        # Обрабатываем все карточки в изображении
        for row in range(2):
            for col in range(4):
                card = self._crop_card(screenshot, card_width, card_height, row, col)
                resized_card = self._resize_card(card, card_width, card_height)
                self._save_card(resized_card)
                self.index += 1
    
    def _crop_card(self, image, card_width, card_height, row, col):
        """Вырезает одну карточку из изображения"""
        left = col * card_width
        upper = row * card_height
        right = left + card_width
        lower = upper + card_height
        return image.crop((left, upper, right, lower))
    
    def _resize_card(self, card, card_width, card_height):
        """Увеличивает размер карточки в 2 раза"""
        new_width = int(card_width * 2)
        new_height = int(card_height * 2)
        return card.resize((new_width, new_height), Image.LANCZOS)
    
    def _save_card(self, card):
        """Сохраняет карточку в выходную папку"""
        output_filename = f'card_{self.index}.png'
        output_path = os.path.join(self.output_folder, output_filename)
        card.save(output_path, format="png", quality=100)
    
    def split_all_images(self):
        """Обрабатывает все изображения в входной папке"""
        self.setup_output_folder()
        
        for filename in os.listdir(self.input_folder):
            if self.is_valid_image(filename):
                image_path = os.path.join(self.input_folder, filename)
                self.process_image(image_path)

def main():
    splitter = ImageSplitter()
    splitter.split_all_images()

if __name__ == "__main__":
    main()

