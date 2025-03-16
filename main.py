import itertools
import os
import sys
import threading

import cv2
import gdown
import numpy as np
import requests
import tensorflow as tf
import wx
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, rotate
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class RedirectText:
    def __init__(self, text_ctrl):
        self.text_ctrl = text_ctrl

    def write(self, text):
        wx.CallAfter(self.text_ctrl.AppendText, text)

    def flush(self):
        pass


class FileWatcher(FileSystemEventHandler):
    def __init__(self, frame):
        self.frame = frame

    def on_created(self, event):
        """Срабатывает при создании нового файла в папке Object"""
        if not event.is_directory:
            wx.CallAfter(self.frame.process_new_file, event.src_path)


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Распознавание номеров", size=(1000, 800))
        self.panel = wx.Panel(self)
        self.SetMinSize((800, 600))

        # Элементы интерфейса
        self.text_ctrl = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.image_ctrl = wx.StaticBitmap(self.panel)

        # Настройка лэйаута
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.text_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        vbox.Add(self.image_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        self.panel.SetSizer(vbox)

        # Перенаправление вывода
        sys.stdout = RedirectText(self.text_ctrl)

        # Загрузка моделей
        self.download_models()

        # Запуск наблюдателя за папкой
        self.start_file_watcher()

    def download_models(self):
        def download_task():
            try:
                self.log_message("Начало загрузки моделей...")
                Download()
                if not all(
                    os.path.isfile(f)
                    for f in ["model_resnet.tflite", "model1_nomer.tflite"]
                ):
                    raise Exception("Не удалось загрузить все модели")
                self.log_message("Модели успешно загружены!\n")
            except Exception as e:
                self.log_message(f"Ошибка загрузки: {str(e)}\n")

        threading.Thread(target=download_task).start()

    def start_file_watcher(self):
        """Запуск наблюдателя за папкой Object"""
        path = "./Object"
        if not os.path.exists(path):
            os.makedirs(path)

        self.observer = Observer()
        self.observer.schedule(FileWatcher(self), path, recursive=False)
        self.observer.start()
        self.log_message(f"Наблюдение за папкой {path} запущено...\n")

    def process_new_file(self, file_path):
        """Обработка нового файла"""

        def recognition_task():
            try:
                self.log_message(
                    f"\nОбнаружен новый файл: {os.path.basename(file_path)}\n"
                )

                # Проверка наличия моделей
                if not all(
                    os.path.isfile(f)
                    for f in ["model_resnet.tflite", "model1_nomer.tflite"]
                ):
                    raise FileNotFoundError(
                        "Модели не найдены! Проверьте подключение к интернету"
                    )

                # Запуск распознавания
                recognized_number = Recognition(file_path, self)
                if recognized_number:
                    self.log_message(f"Распознанный номер: {recognized_number}\n")

                    # Проверка номера в файле registered.txt
                    if self.is_number_registered(recognized_number):
                        self.log_message("Вход разрешен\n")
                    else:
                        self.log_message("Вход запрещен\n")
                else:
                    self.log_message("Номер не распознан\n")

            except Exception as e:
                self.log_message(f"\nОШИБКА: {str(e)}\n")

        threading.Thread(target=recognition_task).start()

    def is_number_registered(self, number):
        """Проверка, зарегистрирован ли номер в файле registered.txt"""
        if not os.path.isfile("registered.txt"):
            self.log_message("Файл registered.txt не найден\n")
            return False

        with open("registered.txt", "r") as f:
            registered_numbers = [line.strip() for line in f.readlines()]

        return number in registered_numbers

    def log_message(self, message):
        wx.CallAfter(self.text_ctrl.AppendText, message)

    def show_image(self, img):
        """Отображает изображение в интерфейсе"""
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            img_pil = Image.fromarray(img)
            img_wx = wx.Bitmap.FromBuffer(w, h, img_pil.tobytes())
            wx.CallAfter(self.image_ctrl.SetBitmap, img_wx)


# Ваши оригинальные функции с модификациями
def Download():
    try:
        if not os.path.isfile("model_resnet.tflite"):
            print("Загрузка model_resnet.tflite...")
            API_ENDPOINT = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}"
            link_r = requests.get(
                API_ENDPOINT.format("https://disk.yandex.ru/d/QavLH1pvpRhLOA"),
                timeout=10,
            )
            response = requests.get(link_r.json()["href"], timeout=30)
            if response.status_code == 200:
                with open("./model_resnet.tflite", "wb") as f:
                    f.write(response.content)

        if not os.path.isfile("model1_nomer.tflite"):
            print("Загрузка model1_nomer.tflite...")
            gdown.download(
                "https://drive.google.com/uc?id=1aBqB4QDKfYpoPBWLIrjwokVXf2xJ1hfs",
                "model1_nomer.tflite",
                quiet=True,
            )
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка подключения: {str(e)}")


def DecodeBatch(out):
    letters = "0 1 2 3 4 5 6 7 8 9 A B C E H K M O P T X Y".split()
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = "".join([letters[c] for c in out_best if c < len(letters)])
        ret.append(outstr)
    return ret


def Recognition(file_path, frame):
    try:
        modelRecPath = "model_resnet.tflite"
        modelPath = "model1_nomer.tflite"

        # Чтение изображения
        image0 = cv2.imread(file_path, 1)
        if image0 is None:
            raise ValueError("Не удалось прочитать изображение")

        # Обработка изображения
        image_height, image_width, _ = image0.shape
        image = cv2.resize(image0, (1024, 1024))
        image = image.astype(np.float32)

        # Распознавание номера
        interpreter = tf.lite.Interpreter(model_path=modelRecPath)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
        interpreter.set_tensor(input_details[0]["index"], X_data1)
        interpreter.invoke()
        detection = interpreter.get_tensor(output_details[0]["index"])

        # Отрисовка прямоугольника вокруг номера
        img2 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        box_x = int(detection[0, 0, 0] * image_height)
        box_y = int(detection[0, 0, 1] * image_width)
        box_width = int(detection[0, 0, 2] * image_height)
        box_height = int(detection[0, 0, 3] * image_width)

        if np.min(detection[0, 0, :]) >= 0:
            cv2.rectangle(
                img2,
                (box_y, box_x),
                (box_height, box_width),
                (230, 230, 21),
                thickness=5,
            )

            # Отображение изображения в интерфейсе
            wx.CallAfter(frame.show_image, img2)

            # Распознавание текста
            image_crop = image0[box_x:box_width, box_y:box_height, :]
            grayscale = rgb2gray(image_crop)
            edges = canny(grayscale, sigma=3.0)
            out, angles, distances = hough_line(edges)
            _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
            angle = np.mean(np.rad2deg(angles_peaks))

            # Коррекция угла
            if 0 <= angle <= 90:
                rot_angle = angle - 90
            elif -45 <= angle < 0:
                rot_angle = angle - 90
            elif -90 <= angle < -45:
                rot_angle = 90 + angle
            if abs(rot_angle) > 20:
                rot_angle = 0

            # Поворот изображения
            rotated = rotate(image_crop, rot_angle, resize=True) * 255
            rotated = rotated.astype(np.uint8)
            rotated1 = rotated[:, :, :]
            minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
            if rotated.shape[1] / rotated.shape[0] < 2 and minus > 10:
                rotated1 = rotated[minus:-minus, :, :]

            # Улучшение контраста
            lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # Распознавание текста
            interpreter = tf.lite.Interpreter(model_path=modelPath)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            img = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 64))
            img = img.astype(np.float32)
            img /= 255
            img1 = img.T
            X_data1 = np.float32(img1.reshape(1, 128, 64, 1))
            interpreter.set_tensor(input_details[0]["index"], X_data1)
            interpreter.invoke()
            net_out_value = interpreter.get_tensor(output_details[0]["index"])
            pred_texts = DecodeBatch(net_out_value)

            # Возврат распознанного номера
            return pred_texts[0] if pred_texts else None
        else:
            wx.CallAfter(frame.show_image, image0)
            return None

    except Exception as e:
        print(f"Ошибка при распознавании: {str(e)}")
        return None


if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame()
    frame.Show()
    app.MainLoop()
