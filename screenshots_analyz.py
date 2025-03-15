import cv2
import numpy as np
from easyocr import Reader
from Levenshtein import distance
import json
from sys import stderr
import ssl
import os
import re

ssl._create_default_context = ssl._create_unverified_context

class ScreenshotAnalyzer:
    def __init__(self, screenshots_dir='./ready_screenshots/', output_json='./json/results.json'):
        self.screenshots_dir = screenshots_dir
        self.output_json = output_json
        self.reader = Reader(lang_list=["en"], gpu=False, verbose=True, model_storage_directory='./easyocr_models')
        self.image_extensions = ('.png', '.jpg', '.jpeg')
        
    def setup_directories(self):
        """Создает необходимые директории для выходного JSON файла"""
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)

    def extract_text_regions(self, image):
        """Извлекает регионы с текстом из изображения"""
        hi, wi = image.shape[:2]
        text_region_height = int(hi * 0.37)
        text_region = image[hi - text_region_height:hi, 0:wi]
        count_and_price_r = int(hi * 0.18)
        count_and_price = image[hi - count_and_price_r:hi, 0:wi]

        hi1, wi1 = count_and_price.shape[:2]
        height_to_keep = int(hi1 * 1)
        
        name_region = text_region[hi1 - height_to_keep:hi1, 0:wi1]
        count_region = count_and_price[:, 0:int(wi * 0.5)]
        price_region = count_and_price[:, int(wi * 0.4):wi]

        return name_region, count_region, price_region

    def process_price(self, price_image):
        """Обрабатывает регион с ценой"""
        price_res = self.reader.readtext(price_image, allowlist='G0123456789.,')
        price_res.sort(key=lambda x: x[-1], reverse=True)
        best_text = price_res[0][1] if price_res else "0.0"
        match = re.search(r'(\d+\.\d+|\d+)', best_text)
        return float(match.group(1)) if match else 0.0

    def process_count(self, count_image):
        """Обрабатывает регион с количеством"""
        count_res = self.reader.readtext(count_image)
        count_text = count_res[0][-2] if count_res and count_res[0][-1] >= 0.4 else count_res[0][-2] if count_res else "0"
        match = re.search(r'(\d+)', count_text)
        return int(match.group(1)) if match else 0

    def process_name(self, name_image):
        """Обрабатывает регион с названием"""
        name_res = self.reader.readtext(name_image)
        name_text = ' '.join([res[-2] for res in name_res if res[-1] >= 0.4]) if name_res else "Неизвестно"
        return name_text

    def process_image(self, filepath):
        """Обрабатывает одно изображение"""
        image = cv2.imread(filepath)
        if image is None:
            return None

        name_image, count_image, price_image = self.extract_text_regions(image)
        
        try:
            price = self.process_price(price_image)
            count = self.process_count(count_image)
            name = self.process_name(name_image)

            return {
                'filename': os.path.basename(filepath),
                'name': name,
                'price': price,
                'count': count
            }
        except Exception as e:
            return None

    def analyze_screenshots(self):
        """Анализирует все скриншоты в директории"""
        if not os.path.exists(self.screenshots_dir):
            return []

        self.setup_directories()
        skins_list = []

        for filename in os.listdir(self.screenshots_dir):
            if filename.lower().endswith(self.image_extensions):
                filepath = os.path.join(self.screenshots_dir, filename)
                result = self.process_image(filepath)
                if result:
                    skins_list.append(result)

        return self.save_results(skins_list)

    def save_results(self, skins_list):
        """Сохраняет результаты в JSON файл и возвращает их"""
        try:
            with open(self.output_json, 'w', encoding='utf-8') as json_file:
                json.dump(skins_list, json_file, ensure_ascii=False, indent=4)
            return skins_list
        except Exception as e:
            return []

def main():
    analyzer = ScreenshotAnalyzer()
    skins_list = analyzer.analyze_screenshots()
    return len(skins_list)

if __name__ == "__main__":
    main()