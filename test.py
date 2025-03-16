import cv2
import numpy as np
import os
import json

try:
    import easyocr
    OCR_ENABLED = True
    reader = easyocr.Reader(['en'], gpu=False)
except ModuleNotFoundError:
    OCR_ENABLED = False
    print("EasyOCR не установлен. Функция OCR отключена.")

class CardDetector:
    def __init__(self, input_folder='ready_screenshots', complete_json_file='all_card_data.json', incomplete_json_file='incomplete_card_data.json'):
        self.input_folder = input_folder
        self.complete_json_file = complete_json_file
        self.incomplete_json_file = incomplete_json_file
        self.complete_card_data = []
        self.incomplete_card_data = []
        self.setup_output_files()

    def setup_output_files(self):
        """Инициализирует JSON-файлы, обрабатывая возможные ошибки."""
        for file_path, data_list in [(self.complete_json_file, self.complete_card_data), 
                                     (self.incomplete_json_file, self.incomplete_card_data)]:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data_list.extend(json.load(f))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Ошибка при чтении {file_path}: {e}. Инициализируем пустой список.")

    def is_valid_image(self, filename):
        """Проверяет, является ли файл изображением."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

    def recognize_card_content(self, card):
        """Распознаёт текст на карточке с помощью EasyOCR."""
        if OCR_ENABLED:
            return "\n".join(res[1] for res in reader.readtext(card))
        return "OCR не доступен"

    def parse_card_text(self, text):
        """Извлекает данные (Name, Count(WT), Price) из текста."""
        lines = text.split("\n")
        card_data = {"Name": "", "Count(WT)": 0, "Price": 0.0}
        is_complete = True

        if not lines or "OCR не доступен" in text:
            card_data["error"] = "OCR failed"
            return card_data, False

        try:
            # Обработка Name и ST
            first_line = lines[0].strip()
            st_prefix, name = ("ST ", lines[1].strip()) if first_line.upper() in ["ST", "R"] else ("", first_line)
            if not st_prefix:
                first_line_words = first_line.split()
                st_prefix = "ST " if first_line_words and first_line_words[0].upper() in ["ST", "R"] else ""
                name = " ".join(first_line_words[1:]).strip() if st_prefix else first_line

            if not name:
                card_data["error"] = "Name is empty"
                is_complete = False
            card_data["Name"] = st_prefix + name

            # Поиск Count(WT)
            count, wt_idx = 0, -1
            start_idx = 1 if st_prefix else 0
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if any(wt_marker in line.lower() for wt_marker in ["wt.", "wt:", "wt", "wt;"]):
                    count = int(line.lower().split("wt")[0].strip() or 0)
                    wt_idx = i
                    break
                if line.isdigit():
                    count = int(line)
                    if i + 1 < len(lines) and any(wt_marker in lines[i + 1].lower() for wt_marker in ["wt.", "wt:", "wt", "wt;"]):
                        wt_idx = i + 1
                        break

            if wt_idx == -1:
                card_data["error"] = "Count(WT) or wt marker not found"
                is_complete = False
            if count <= 0:
                card_data["error"] = "Invalid Count(WT)"
                is_complete = False
            card_data["Count(WT)"] = count

            # Поиск Price
            price_idx = wt_idx + 1 if wt_idx != -1 else start_idx + 2
            price_line = lines[price_idx].strip() if len(lines) > price_idx else ""
            price_str = price_line.replace("G", "").strip()
            price = float(price_str) if price_str.replace(".", "").replace("-", "").isdigit() else 0.0
            if price <= 0:
                card_data["error"] = "Invalid Price"
                is_complete = False
            card_data["Price"] = price

        except (IndexError, ValueError) as e:
            print(f"Ошибка при разборе текста карточки: {e}")
            print(f"Текст карточки: {text}")
            card_data["error"] = f"Parsing error: {str(e)}"
            is_complete = False

        return card_data, is_complete

    def process_image(self, image_path):
        """Обрабатывает изображение и добавляет данные в JSON."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return

        image_name = os.path.basename(image_path)
        text = self.recognize_card_content(image)
        print(f"Card text: {text}")
        
        card_data, is_complete = self.parse_card_text(text)
        card_data["image_name"] = image_name
        print(f"Parsed data: {card_data}")

        target_list = self.complete_card_data if is_complete else self.incomplete_card_data
        target_list.append(card_data)
        print(f"{'Полные' if is_complete else 'Неполные'} данные карточки добавлены в: {self.complete_json_file if is_complete else self.incomplete_json_file}")

        with open(self.complete_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.complete_card_data, f, ensure_ascii=False, indent=4)
        with open(self.incomplete_json_file, 'w', encoding='utf-8') as f:
            json.dump(self.incomplete_card_data, f, ensure_ascii=False, indent=4)

    def process_all_images(self):
        """Обрабатывает все изображения в папке."""
        for filename in os.listdir(self.input_folder):
            if self.is_valid_image(filename):
                image_path = os.path.join(self.input_folder, filename)
                print(f"Обработка изображения: {image_path}")
                self.process_image(image_path)

def main():
    detector = CardDetector(input_folder='ready_screenshots')
    detector.process_all_images()

if __name__ == "__main__":
    main()