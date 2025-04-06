import itertools
import os
import sys
import threading

import cv2
import gdown
import numpy as np
import pandas as pd
import psycopg2
import requests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import wx
import wx.adv
import wx.grid
from dotenv import load_dotenv
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, rotate
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


TF_ENABLE_ONEDNN_OPTS = 0


load_dotenv(".env")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


class RedirectText:
    def __init__(self, text):
        self.text = text

    def write(self, text):
        wx.CallAfter(self.text.AppendText, text)

    def flush(self):
        pass


class FileWatcher(FileSystemEventHandler):
    def __init__(self, frame):
        self.frame = frame

    def on_created(self, event):
        """Срабатывает при создании нового файла в папке plates"""
        if not event.is_directory:
            wx.CallAfter(self.frame.process_new_file, event.src_path)


class LicensePlateRecognitionApp(wx.Frame):

    def __init__(self, parent=None):
        wx.Frame.__init__(
            self,
            parent,
            id=wx.ID_ANY,
            title=wx.EmptyString,
            pos=wx.DefaultPosition,
            size=wx.Size(1000, 800),
            style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
        )

        self.SetSizeHints(wx.Size(800, 600), wx.DefaultSize)

        # Устанавливаем иконку
        self.SetIcon(wx.Icon("app_icon.ico", wx.BITMAP_TYPE_ICO))

        # Создание основного макета
        self.create_main_layout()
        self.Centre(wx.BOTH)

    def create_img_panel(self, parent):
        """Создаёт панель для изображения."""
        panel = wx.BoxSizer(wx.VERTICAL)
        self.IMG = wx.StaticBitmap(
            parent, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0
        )
        panel.Add(self.IMG, 1, wx.ALL | wx.EXPAND, 5)
        return panel

    def create_log_panel(self, parent):
        """Создаёт панель для логов и добавляет кнопку экспорта."""
        panel = wx.BoxSizer(wx.VERTICAL)
        self.logPanel = wx.TextCtrl(
            parent, wx.ID_ANY, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2
        )
        panel.Add(self.logPanel, 1, wx.EXPAND | wx.ALL, 5)

        return panel

    def create_grid_panel(self, parent):
        """Создаёт панель для таблицы логов и добавляет кнопку экспорта."""
        panel = wx.BoxSizer(wx.VERTICAL)
        self.logGrid = wx.grid.Grid(
            parent, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0
        )

        # Настройка цветов
        self.setup_colors()

        # Настройка таблицы
        self.logGrid.CreateGrid(5, 4)
        self.logGrid.SetColLabelValue(0, "Машина")
        self.logGrid.SetColLabelValue(1, "Водитель")
        self.logGrid.SetColLabelValue(2, "Время прохождения КПП")
        self.logGrid.SetColLabelValue(3, "Статус")

        self.logGrid.EnableEditing(True)
        self.logGrid.EnableGridLines(True)
        self.logGrid.EnableDragGridSize(False)
        self.logGrid.SetMargins(0, 0)

        # Настройки столбцов
        self.logGrid.EnableDragColMove(False)
        self.logGrid.EnableDragColSize(True)
        self.logGrid.SetColLabelAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)

        # Настройки строк
        self.logGrid.EnableDragRowSize(True)
        self.logGrid.SetRowLabelAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)

        # Выравнивание ячеек
        self.logGrid.SetDefaultCellAlignment(wx.ALIGN_LEFT, wx.ALIGN_TOP)
        panel.Add(self.logGrid, 1, wx.EXPAND | wx.ALL, 5)

        # Автоматическое подстраивание ширины столбцов
        self.logGrid.AutoSizeColumns()

        # Добавление панели кнопок под таблицу
        panel.Add(self.create_buttons_panel(parent), 0, wx.EXPAND, 5)

        return panel

    def create_buttons_panel(self, parent):
        """Создаёт панель с кнопками 'Экспорт' и 'Обновить'."""
        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

        # Кнопка экспорт
        self.exportButton = wx.Button(parent, wx.ID_ANY, "Экспорт в Excel")
        buttonSizer.Add(self.exportButton, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        # Разделитель между кнопками
        buttonSizer.AddSpacer(10)

        # # Кнопка обновить
        # self.updateButton = wx.Button(parent, wx.ID_ANY, "Обновить")
        # buttonSizer.Add(self.updateButton, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        # # Разделитель между кнопками
        # buttonSizer.AddSpacer(10)

        # Кнопка выбора даты
        self.datePicker = wx.adv.DatePickerCtrl(
            self,
            wx.ID_ANY,
            wx.DefaultDateTime,
            wx.DefaultPosition,
            wx.DefaultSize,
            style=wx.adv.DP_DROPDOWN | wx.adv.DP_SHOWCENTURY,
        )
        buttonSizer.Add(self.datePicker, 0, wx.ALL, 5)

        # Установить текущую дату
        self.datePicker.SetValue(wx.DateTime.Now())

        # Установить минимальную и максимальную даты
        min_date = wx.DateTime()
        min_date.ParseDate("2025-01-01")

        max_date = wx.DateTime()
        max_date.ParseDate("2025-12-31")

        # Можно выбрать даты только в 2025 году
        self.datePicker.SetRange(min_date, max_date)

        # Привязка обработчиков событий
        self.exportButton.Bind(wx.EVT_BUTTON, self.export_to_excel)
        # self.updateButton.Bind(wx.EVT_BUTTON, self.update)
        self.datePicker.Bind(wx.adv.EVT_DATE_CHANGED, self.on_date_change)

        return buttonSizer

    def create_main_layout(self):
        """Создаёт и размещает основные сайзеры."""
        mainSizer = wx.BoxSizer(wx.VERTICAL)

        # Верхний сайзер (изображение + лог панель)
        highSizer = wx.BoxSizer(wx.HORIZONTAL)
        highSizer.Add(self.create_img_panel(self), 1, wx.EXPAND, 5)
        highSizer.Add(self.create_log_panel(self), 1, wx.EXPAND, 5)

        # Нижний сайзер (таблица логов + кнопка)
        downSizer = wx.BoxSizer(wx.VERTICAL)
        downSizer.Add(self.create_grid_panel(self), 1, wx.EXPAND, 5)

        # Добавление Верхнего и Нижнего сайзера в Главный сайзер
        mainSizer.Add(highSizer, 1, wx.EXPAND, 5)
        mainSizer.Add(downSizer, 1, wx.EXPAND, 5)

        self.SetSizer(mainSizer)
        self.Layout()

        # Перенаправление вывода
        sys.stdout = RedirectText(self.logPanel)

        # Загрузка моделей
        self.download_models()

        # Запуск наблюдателя за папкой
        self.start_file_watcher()

        # Загрузка данных в таблицу
        self.load_data_from_db()

    def setup_colors(self):
        """Настройка цветовой схемы"""
        bg_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        self.SetBackgroundColour(bg_color)

        # Настройка таблицы
        self.logGrid.SetBackgroundColour(bg_color)
        self.logGrid.SetDefaultCellBackgroundColour(bg_color)
        self.logGrid.SetLabelBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE)
        )

    # & -----------------  Обработчики  -----------------  #

    # & -----------------  Лог  -----------------  #

    # Текстовый поток
    def log_message(self, message):
        wx.CallAfter(self.logPanel.AppendText, message)

    # & -----------------  Распознование  ---------------  #

    # Загрузка моделей
    def download_models(self):
        threading.Thread(target=self.download_task, daemon=True).start()

    def download_task(self):
        """Фоновая загрузка моделей"""
        try:
            wx.CallAfter(self.log_message, "Поиск моделей...\n")
            self.Download()
            if not all(
                os.path.isfile(f)
                for f in ["model_resnet.tflite", "model_number_recognition.tflite"]
            ):
                raise Exception("Не удалось загрузить все модели")
            wx.CallAfter(self.log_message, "Модели успешно загружены!\n")
        except Exception as e:
            wx.CallAfter(self.log_message, f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {str(e)}\n")

    # Наблюдение за папкой
    def start_file_watcher(self):
        """Запуск наблюдателя за папкой plates"""
        path = "./plates"
        if not os.path.exists(path):
            os.makedirs(path)

        self.observer = Observer()
        self.observer.schedule(FileWatcher(self), path, recursive=False)
        self.observer.start()
        self.log_message(f"Наблюдение за папкой {path[2:]} запущено...\n\n")

    def process_new_file(self, file_path):
        """Обработка нового файла"""
        threading.Thread(
            target=self.recognition_task, args=(file_path,), daemon=True
        ).start()

    def recognition_task(self, file_path):
        """Фоновая обработка распознавания номера"""
        try:
            wx.CallAfter(
                self.log_message,
                f"\nОбнаружен новый файл: {os.path.basename(file_path)}\n",
            )

            # Проверка наличия моделей
            if not all(
                os.path.isfile(f)
                for f in ["model_resnet.tflite", "model_number_recognition.tflite"]
            ):
                raise FileNotFoundError(
                    "Модели не найдены! Проверьте подключение к интернету"
                )

            # Запуск распознавания
            recognized_number = self.Recognition(file_path, self)
            if recognized_number:
                wx.CallAfter(
                    self.log_message, f"Распознанный номер: {recognized_number}\n"
                )
                if self.is_number_registered(recognized_number):
                    self.load_data_to_db(recognized_number)
                    wx.CallAfter(self.log_message, "Вход разрешен\n")
                    self.on_date_change()
                else:
                    wx.CallAfter(self.log_message, "Вход запрещен\n")
            else:
                wx.CallAfter(self.log_message, "Номер не распознан\n")

        except Exception as e:
            wx.CallAfter(self.log_message, f"\nОШИБКА РАСПОЗНАВАНИЯ: {str(e)}\n")

    def is_number_registered(self, number):
        """Загружает зарегистрированные номера из базы данных."""
        try:
            # Подключение к БД
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            cursor = conn.cursor()

            cursor.execute("SELECT vehiclemark FROM vehicle;")
            numbers = {row[0] for row in cursor.fetchall()}

            cursor.close()
            conn.close()

            return number in numbers

        except Exception as e:
            print(f"ОШИБКА ПРОВЕРКИ НОМЕРА: {e}")
            return set()
        finally:
            cursor.close()
            conn.close()

    # & -----------------  Изображение  -----------------  #

    def show_image(self, img):
        """Отображает изображение в интерфейсе с масштабированием"""
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            img_pil = Image.fromarray(img)

            # Конвертация в wx.Image
            wx_img = wx.Image(img_pil.size[0], img_pil.size[1])
            wx_img.SetData(img_pil.convert("RGB").tobytes())

            # Получение размеров области отображения
            ctrl_size = self.IMG.GetSize()
            ctrl_w, ctrl_h = ctrl_size.GetWidth(), ctrl_size.GetHeight()

            # Масштабирование с сохранением пропорций
            img_ratio = w / h
            ctrl_ratio = ctrl_w / ctrl_h

            if img_ratio > ctrl_ratio:
                new_w = ctrl_w
                new_h = int(ctrl_w / img_ratio)
            else:
                new_h = ctrl_h
                new_w = int(ctrl_h * img_ratio)

            wx_img = wx_img.Scale(new_w, new_h)

            # Отображение изображения
            wx.CallAfter(self.IMG.SetBitmap, wx.Bitmap(wx_img))

    # & -----------------  Журнал  -----------------  #

    # Динамическое обновление журнала
    def on_date_change(self, date=wx.DateTime.Now().FormatISODate()):
        """Обработчик на изменение даты"""
        # Получить объект wx.DateTime
        current_date = self.datePicker.GetValue()

        # Преобразовать в строку (ISO-формат: ГГГГ-ММ-ДД)
        date = current_date.FormatISODate()  # Пример: "2025-03-12"

        self.load_data_from_db(date)

    # Динамическое обновление столбцов
    def update_grid(self):
        """Обновляет размеры столбцов после добавления данных."""
        self.logGrid.AutoSizeColumns()

    # Подгрузка журнала
    def load_data_from_db(self, date=wx.DateTime.Now().FormatISODate()):
        """Загружает данные из PostgreSQL в таблицу"""
        try:
            # Подключение к БД
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            cursor = conn.cursor()

            # Запрос данных
            query = """
            SELECT
            CONCAT_WS(' ', v.vehiclecolor, v.vehicletype, v.vehiclemark) AS vehicle,
            CONCAT_WS(' ', d.driver_firstname, d.driver_secondname, d.driver_patronymic) AS driver,
            l.transittime,
            CASE WHEN transittype THEN 'Заехал' 
            ELSE 'Выехал' 
            END AS status
            FROM "log" AS l
			JOIN "vehicle" AS v ON l.id_vehicle = v.id_vehicle
            JOIN "driver" AS d ON v.id_driver = d.id_driver
            WHERE l.transittime::date = %s
            ORDER BY l.transittime DESC
            LIMIT 50
            """
            cursor.execute(query, (date,))
            rows = cursor.fetchall()

            # Очистка сетки перед обновлением
            self.logGrid.ClearGrid()
            if self.logGrid.GetNumberRows() > 0:
                self.logGrid.DeleteRows(0, self.logGrid.GetNumberRows())

            # Получаем текущее количество строк в таблице wx.Grid
            current_rows = self.logGrid.GetNumberRows()

            # Если в БД больше строк, чем в Grid, добавляем недостающие
            if len(rows) > current_rows:
                self.logGrid.AppendRows(len(rows) - current_rows)

            # Обновляем данные в wx.Grid
            for row_index, row in enumerate(rows):
                for col_index, value in enumerate(row):
                    self.logGrid.SetCellValue(row_index, col_index, str(value))

            # Если в Grid больше строк, чем в БД, удаляем лишние
            if len(rows) < current_rows:
                self.logGrid.DeleteRows(len(rows), current_rows - len(rows))

            cursor.close()
            conn.close()

            # Автоматическое подстраивание колонок после добавления данных
            self.update_grid()

        except Exception as e:
            wx.MessageBox(f"ОШИБКА ЗАГРУЗКИ ДАННЫХ: {e}", "Ошибка", wx.ICON_ERROR)
        finally:
            cursor.close()
            conn.close()

    def load_data_to_db(self, number):
        """Добавление распознанной и идентифицированной машины в журнал"""
        try:
            # Подключение к БД
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            cursor = conn.cursor()

            query = """
            WITH get_vehicle AS (
                SELECT id_vehicle 
                FROM vehicle 
                WHERE vehiclemark = %s
                LIMIT 1
            )
            INSERT INTO log (
                id_vehicle, 
                transittime, 
                transittype
            )
            SELECT 
                id_vehicle,
                DATE_TRUNC('second', CURRENT_TIMESTAMP),
                TRUE
            FROM get_vehicle;
            """

            # Параметр передается как кортеж с одним элементом
            cursor.execute(query, (number,))
            conn.commit()

        except Exception as e:
            print(f"ОШИБКА ДОБАВЛЕНИЯ В ЖУРНАЛ: {e}")
        finally:
            cursor.close()
            conn.close()

    # Обновление журнала
    # def update(self, event):
    #     """Обновление журнала без полной перезагрузки таблицы"""
    #     pass

    # Выгрузка журнала
    def export_to_excel(self, event):
        """Экспорт данных в Excel"""
        dlg = wx.TextEntryDialog(
            self, "Введите количество строк для выгрузки:", "Выгрузка в Excel"
        )
        if dlg.ShowModal() == wx.ID_OK:
            try:
                row_count = int(dlg.GetValue())

                # Подключение к БД
                conn = psycopg2.connect(
                    host=DB_HOST,
                    port=DB_PORT,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                )

                # Запрос
                query = f"""
                SELECT
                CONCAT_WS(' ', v.vehiclecolor, v.vehicletype, v.vehiclemark) AS vehicle,
                CONCAT_WS(' ', d.driver_firstname, d.driver_secondname, d.driver_patronymic) AS driver,
                l.transittime,
                CASE WHEN transittype THEN 'Заехал' 
                ELSE 'Выехал' 
                END AS status
                FROM "log" AS l
				JOIN "vehicle" AS v ON l.id_vehicle = v.id_vehicle
                JOIN "driver" AS d ON v.id_driver = d.id_driver
                ORDER BY l.transittime DESC
                LIMIT {row_count}
                """
                df = pd.read_sql(query, conn)  # Pandas DataFrame (df)
                conn.close()

                # Сохранение Excel на рабочем столе
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
                output_file = os.path.join(desktop_path, "logbook_export.xlsx")

                df.to_excel(output_file, index=False)
                wx.MessageBox(
                    f"Данные успешно выгружены в Excel:\n{output_file}",
                    "Успех",
                    wx.ICON_INFORMATION,
                )

            except Exception as e:
                wx.MessageBox(f"ОШИБКА ЭКСПОРТА: {e}", "Ошибка", wx.ICON_ERROR)
            finally:
                conn.close()

        dlg.Destroy()

    # & -----------------  Свой функции  -----------------  #

    def Download(self):
        try:
            if not os.path.isfile("model_resnet.tflite"):
                print("Загрузка model_resnet.tflite...")
                gdown.download(
                    "https://disk.yandex.ru/d/QavLH1pvpRhLOA",
                    "model_resnet.tflite",
                    quiet=True,
                )

            if not os.path.isfile("model_number_recognition.tflite"):
                print("Загрузка model_number_recognition.tflite...")
                gdown.download(
                    "https://github.com/sovse/tflite_avto_num_recognation/blob/main/model1_nomer.tflite",
                    "model_number_recognition.tflite",
                    quiet=True,
                )
        except requests.exceptions.RequestException as e:
            raise Exception(f"ОШИБКА ПОДКЛЮЧЕНИЯ К ИНТЕРНЕТ: {str(e)}")

    def DecodeBatch(self, out):
        letters = "0 1 2 3 4 5 6 7 8 9 A B C E H K M O P T X Y".split()
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = "".join([letters[c] for c in out_best if c < len(letters)])
            ret.append(outstr)
        return ret

    def Recognition(self, file_path, frame):
        try:
            modelRecPath = "model_resnet.tflite"
            modelPath = "model_number_recognition.tflite"

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
                _, angles_peaks, _ = hough_line_peaks(
                    out, angles, distances, num_peaks=20
                )
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
                pred_texts = self.DecodeBatch(net_out_value)

                # Возврат распознанного номера
                return pred_texts[0] if pred_texts else None
            else:
                wx.CallAfter(frame.show_image, image0)
                return None

        except Exception as e:
            print(f"ОШИБКА РАСПОЗНОВАНИЯ: {str(e)}")
            return None

    def __del__(self):
        pass


if __name__ == "__main__":
    app = wx.App()
    frame = LicensePlateRecognitionApp()
    frame.Show()
    app.MainLoop()
