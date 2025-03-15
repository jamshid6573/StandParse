import cv2
import numpy as np
from easyocr import Reader
from Levenshtein import distance
import json
import os
import re
from typing import List, Dict

class ScreenshotAnalyzer:
    def __init__(self, screenshot_path: str, output_json: str = 'json/price_data.json', skins_file: str = 'skins.txt'):
        self.screenshot_path = screenshot_path
        self.output_json = output_json
        self.output_dir = 'testscreen'
        self.skins_file = skins_file
        self.reader = Reader(lang_list=["en"], gpu=True, verbose=False, model_storage_directory='./easyocr_models')
        self._load_names()  # Загружаем имена один раз при инициализации

    def _load_names(self) -> None:
        """Загружает список имен из файла skins.txt один раз"""
        self.names = set()  # Используем set для быстрого поиска
        if not os.path.exists(self.skins_file):
            return
        try:
            with open(self.skins_file, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip().split(',')[0].strip()
                    if name:
                        self.names.add(name)
        except Exception:
            self.names = set()

    def setup_directories(self) -> None:
        """Создает директорию testscreen, если её нет"""
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_text_regions(self, image: np.ndarray) -> tuple:
        """Извлекает регионы с текстом из изображения"""
        hi, wi = image.shape[:2]
        text_region_height = int(hi * 0.37)
        count_and_price_r = int(hi * 0.18)

        text_region = image[hi - text_region_height:hi, :]
        count_and_price = image[hi - count_and_price_r:hi, :]

        hi1 = count_and_price.shape[0]
        name_region = text_region[hi1 - hi1:hi1, :]
        count_region = count_and_price[:, :int(wi * 0.45)]
        price_region = count_and_price[:, int(wi * 0.4):]

        # Сохранение регионов только для отладки (можно убрать в продакшене)
        cv2.imwrite(os.path.join(self.output_dir, 'text_region.png'), text_region)
        cv2.imwrite(os.path.join(self.output_dir, 'name_region.png'), name_region)
        cv2.imwrite(os.path.join(self.output_dir, 'count_region.png'), count_region)
        cv2.imwrite(os.path.join(self.output_dir, 'price_region.png'), price_region)

        return name_region, count_region, price_region

    def process_price(self, price_image):
        """Обрабатывает регион с ценой"""
        price_res = self.reader.readtext(price_image, allowlist='G0123456789.,')
        price_res.sort(key=lambda x: x[-1], reverse=True)
        best_text = price_res[0][1] if price_res else "0.0"
        match = re.search(r'(\d+\.\d+|\d+)', best_text)
        return float(match.group(1)) if match else 0.0


    def process_count(self, count_image: np.ndarray) -> int:
        """Обрабатывает регион с количеством"""
        count_res = self.reader.readtext(count_image, detail=0)
        if not count_res:
            return 0
        match = re.search(r'(\d+)', count_res[0])
        return int(match.group(1)) if match else 0

    def process_name(self, name_image: np.ndarray) -> str:
        """Обрабатывает регион с названием"""
        name_res = self.reader.readtext(name_image, detail=0)
        name_text = ' '.join(name_res) if name_res else "Неизвестно"
        
        if not name_text.strip() or not self.names:
            return name_text.strip() or "Неизвестно"
        
        # Поиск ближайшего имени через расстояние Левенштейна
        return min(self.names, key=lambda x: distance(name_text.strip(), x))

    def analyze_screenshot(self) -> List[Dict]:
        """Анализирует скриншот"""
        if not os.path.exists(self.screenshot_path):
            return []

        image = cv2.imread(self.screenshot_path)
        if image is None:
            return []

        self.setup_directories()
        name_image, count_image, price_image = self.extract_text_regions(image)
        skins = []

        try:
            skins.append({
                'name': self.process_name(name_image),
                'price': self.process_price(price_image),
                'count': self.process_count(count_image)
            })
        except Exception as e:
            return str(e)

        return skins

    def save_results(self, skins_list: List[Dict]) -> None:
        """Сохраняет результаты в JSON файл"""
        with open(self.output_json, 'w', encoding='utf-8') as json_file:
            json.dump(skins_list, json_file, ensure_ascii=False, indent=4)

def main():
    analyzer = ScreenshotAnalyzer('./card_0.png')
    try:
        skins_list = analyzer.analyze_screenshot()
        print(skins_list)
        analyzer.save_results(skins_list)
    except Exception as ex:
        print(f"Ошибка: {ex}")
        exit(4)

if __name__ == "__main__":
    main()