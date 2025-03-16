import cv2
import numpy as np
import os
import json
import time

try:
    import easyocr
    OCR_ENABLED = True
    reader = easyocr.Reader(['en'], gpu=False)  # Отключаем GPU, так как нет CUDA
except ModuleNotFoundError:
    OCR_ENABLED = False
    print("EasyOCR не установлен. Функция OCR отключена.")

# Функция для вычисления расстояния Левенштейна
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Функция для поиска ближайшего имени
def find_closest_name(name, correct_names):
    if not correct_names:
        return name
    closest_name = min(correct_names, key=lambda correct_name: levenshtein_distance(name.lower(), correct_name.lower()))
    if levenshtein_distance(name.lower(), closest_name.lower()) > len(name) // 2:
        return name
    return closest_name

class CardDetector:
    def __init__(self, input_folder='simple', complete_json_file='all_card_data.json', incomplete_json_file='incomplete_card_data.json', correct_names_file='correct_names.txt'):
        self.input_folder = input_folder
        self.complete_json_file = complete_json_file
        self.incomplete_json_file = incomplete_json_file
        self.correct_names_file = correct_names_file
        self.complete_card_data = []
        self.incomplete_card_data = []
        self.correct_names = self.load_correct_names()
        self.setup_output_files()
        self.is_stattrack = False  # Флаг для StatTrack

    def load_correct_names(self):
        correct_names = []
        try:
            with open(self.correct_names_file, 'r', encoding='utf-8') as f:
                correct_names = [line.strip() for line in f if line.strip()]
            print(f"Загружено {len(correct_names)} корректных имён из {self.correct_names_file}")
        except FileNotFoundError:
            print(f"Файл {self.correct_names_file} не найден. Коррекция имён не будет выполняться.")
        except Exception as e:
            print(f"Ошибка при чтении файла {self.correct_names_file}: {e}. Коррекция имён не будет выполняться.")
        return correct_names

    def setup_output_files(self):
        for file_path, data_list in [(self.complete_json_file, self.complete_card_data), 
                                     (self.incomplete_json_file, self.incomplete_card_data)]:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data_list.extend(json.load(f))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Ошибка при чтении {file_path}: {e}. Инициализируем пустой список.")

    def is_valid_image(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

    def detect_stattrack(self, image):
        """Обнаруживает оранжево-жёлтый прямоугольник как индикатор StatTrack."""
        # Преобразуем в HSV для сегментации по цвету
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Уточнённый диапазон оранжево-жёлтого цвета
        lower_orange = np.array([5, 100, 100])  # H: 5-25, S: 100-255, V: 100-255
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # Проверяем, находится ли прямоугольник в левой нижней части и имеет подходящий размер
            if (x < image.shape[1] // 3 and y > image.shape[0] // 2 and
                area > 100 and area < 5000):  # Площадь должна быть разумной
                return True
        return False

    def recognize_card_content(self, image):
        if OCR_ENABLED:
            # Проверяем наличие StatTrack через оранжево-жёлтый прямоугольник
            self.is_stattrack = self.detect_stattrack(image)
            if self.is_stattrack:
                print("Обнаружен StatTrack на изображении (оранжево-жёлтый прямоугольник).")
            return "\n".join(res[1] for res in reader.readtext(image))
        return "OCR не доступен"

    def parse_card_text(self, text):
        lines = text.split("\n")
        card_data = {"Name": "", "Count(WT)": 0, "Price": 0.0}
        is_complete = True

        if not lines or "OCR не доступен" in text:
            card_data["error"] = "OCR failed"
            return card_data, False

        try:
            # Обработка Name
            name = ""
            start_idx = 0
            first_line = lines[0].strip() if lines else ""
            # Если обнаружен StatTrack через изображение или текст "ST", "R", "U"
            if self.is_stattrack or first_line.upper() in ["ST", "R", "U"]:
                start_idx = 1  # Пропускаем первую строку
                name = lines[1].strip() if len(lines) > 1 else ""
                if name:
                    name = f"StatTrack {name}"  # Добавляем префикс StatTrack
            # Если первая строка из 1-2 букв и не "ST", "R", "U", пропускаем её
            elif len(first_line) <= 2 and first_line.upper() not in ["ST", "R", "U"]:
                start_idx = 1
                name = lines[1].strip() if len(lines) > 1 else ""
            else:
                # В остальных случаях берём первую строку как имя
                name = first_line

            # Если имя всё ещё пустое, ищем следующую непустую строку
            if not name:
                for line in lines[start_idx:]:
                    line = line.strip()
                    if line and not any(wt_marker in line.lower() for wt_marker in ["wt", "wt:", "wt.", "wt;"]):
                        name = line
                        break

            # Корректируем имя через расстояние Левенштейна
            if name:
                name = find_closest_name(name, self.correct_names)

            if not name:
                is_complete = False
                card_data["error"] = "Name is empty"
            card_data["Name"] = name

            # Поиск Count(WT)
            count, wt_idx = 0, -1
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if any(wt_marker in line.lower() for wt_marker in ["wt", "wt:", "wt.", "wt;"]):
                    count_str = line.lower().split("wt")[0].strip()
                    count = int(count_str) if count_str.isdigit() else 0
                    wt_idx = i
                    break
                if line.isdigit():
                    count = int(line)
                    if i + 1 < len(lines) and any(wt_marker in lines[i + 1].lower() for wt_marker in ["wt", "wt:", "wt.", "wt;"]):
                        wt_idx = i + 1
                        break

            if wt_idx == -1:
                is_complete = False
                card_data["error"] = "Count(WT) or wt marker not found"
            if count <= 0:
                is_complete = False
                card_data["error"] = "Invalid Count(WT)"
            card_data["Count(WT)"] = count

            # Поиск Price
            price_idx = wt_idx + 1 if wt_idx != -1 else 0
            price_line = lines[price_idx].strip() if len(lines) > price_idx else ""
            if price_line.startswith("G"):
                # Заменяем "O" на "0" для коррекции OCR-ошибок
                price_str = price_line.replace("G", "").replace("O", "0").replace("s", "5").strip()
                if price_str.replace(".", "").replace("-", "").isdigit():
                    price = float(price_str)
                else:
                    price = 0.0
            else:
                price = float(price_line) if price_line.replace(".", "").replace("-", "").isdigit() else 0.0

            if price <= 0:
                is_complete = False
                card_data["error"] = "Invalid Price"
            card_data["Price"] = price

        except (IndexError, ValueError) as e:
            print(f"Ошибка при разборе текста карточки: {e}")
            print(f"Текст карточки: {text}")
            card_data["error"] = f"Parsing error: {str(e)}"
            is_complete = False

        return card_data, is_complete

    def process_single_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            return None, None

        image_name = os.path.basename(image_path)
        text = self.recognize_card_content(image)
        print(text)
        card_data, is_complete = self.parse_card_text(text)
        card_data["image_name"] = image_name
        return card_data, is_complete

    def process_all_images(self):
        image_paths = [
            os.path.join(self.input_folder, filename)
            for filename in os.listdir(self.input_folder)
            if self.is_valid_image(filename)
        ]

        if not image_paths:
            print("Не найдено изображений для обработки.")
            return

        total_images = len(image_paths)
        print(f"Запуск обработки {total_images} изображений...")

        start_time = time.time()
        for idx, image_path in enumerate(image_paths, 1):
            print(f"Обработка изображения {idx}/{total_images}: {image_path}")
            result = self.process_single_image(image_path)
            if result[0]:  # Если данные не None
                card_data, is_complete = result
                target_list = self.complete_card_data if is_complete else self.incomplete_card_data
                target_list.append(card_data)

            # Периодическое сохранение каждые 100 изображений
            if idx % 100 == 0 or idx == total_images:
                with open(self.complete_json_file, 'w', encoding='utf-8') as f:
                    json.dump(self.complete_card_data, f, ensure_ascii=False, indent=4)
                with open(self.incomplete_json_file, 'w', encoding='utf-8') as f:
                    json.dump(self.incomplete_card_data, f, ensure_ascii=False, indent=4)
                elapsed_time = time.time() - start_time
                print(f"Обработано {idx}/{total_images} изображений за {elapsed_time:.2f} секунд.")

        elapsed_time = time.time() - start_time
        print(f"Обработка завершена за {elapsed_time:.2f} секунд.")
        print(f"Обработано {total_images} изображений.")
        print(f"Полные данные сохранены в: {self.complete_json_file}")
        print(f"Неполные данные сохранены в: {self.incomplete_json_file}")

def main():
    detector = CardDetector(input_folder='simple')
    detector.process_all_images()

if __name__ == "__main__":
    main()