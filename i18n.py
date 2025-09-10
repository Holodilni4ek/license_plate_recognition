#!/usr/bin/env python3
"""
Internationalization (i18n) Module
Provides multi-language support for the license plate recognition application.
"""

import json
import os
from typing import Dict, Any


class LanguageManager:
    """Manages application translations and language switching."""

    def __init__(self):
        self.current_language = "ru"  # Default to Russian
        self.languages = {}
        self.supported_languages = ["ru", "en"]
        self.load_translations()

    def load_translations(self):
        """Load all translation files."""
        # Russian translations (default)
        self.languages["ru"] = {
            # Main Window
            "app_title": "Система Распознавания Номеров",
            "demo_mode_title": "Демо Режим",
            "demo_mode_message": "Подключение к базе данных не удалось. Запуск в демо режиме.\n\nОшибка: {}\n\nПроверьте настройки базы данных в файле .env.",
            # Common
            "username": "Имя пользователя:",
            "password": "Пароль:",
            "login": "Войти",
            "cancel": "Отмена",
            "refresh": "Обновить",
            "success": "Успех",
            "error": "Ошибка",
            "failed_load_data": "Ошибка загрузки данных: {}",
            "failed_update_grid": "Ошибка обновления таблицы: {}",
            "failed_display_image": "Ошибка отображения изображения: {}",
            # Authentication
            "auth_title": "Авторизация",
            "please_enter_username": "Пожалуйста, введите имя пользователя",
            "please_enter_password": "Пожалуйста, введите пароль",
            "invalid_credentials": "Неверное имя пользователя или пароль",
            "auth_error": "Ошибка авторизации: {}",
            # Main UI - Grid Headers
            "vehicle": "Автомобиль",
            "driver": "Водитель",
            "transit_time": "Время прохождения",
            "status": "Статус",
            # Main UI - Buttons
            "export_to_excel": "Экспорт в Excel",
            "open_journal": "Открыть журнал",
            "add_driver": "Добавить водителя",
            "add_vehicle": "Добавить автомобиль",
            "add_user": "Добавить пользователя",
            "change_language": "English",  # Shows what language it will switch TO
            # Status Messages
            "entered": "Заехал",
            "exited": "Выехал",
            "access_granted": "Доступ разрешен - записано",
            "access_denied": "Автомобиль не зарегистрирован - доступ запрещен",
            "no_plate_recognized": "Номер не распознан",
            "processing_image": "Обработка: {}",
            "recognized_plate": "Распознан номер: {}",
            "recognition_not_ready": "Система распознавания не готова",
            "processing_failed": "Ошибка обработки: {}",
            # Model Loading
            "checking_models": "Проверка моделей...",
            "downloading_resnet": "Загрузка модели ResNet...",
            "downloading_recognition": "Загрузка модели распознавания...",
            "resnet_downloaded": "Модель ResNet загружена успешно",
            "recognition_downloaded": "Модель распознавания загружена успешно",
            "models_missing": "Некоторые модели отсутствуют. Распознавание отключено.",
            "recognition_ready": "Система распознавания готова",
            "model_download_failed": "Ошибка загрузки моделей: {}",
            # File Monitoring
            "file_watcher_started": "Наблюдение за файлами запущено для {}",
            "file_watcher_failed": "Ошибка запуска наблюдения за файлами: {}",
            # Export
            "export_dialog_title": "Экспорт в Excel",
            "export_dialog_message": "Введите количество строк для экспорта:",
            "export_rows_label": "Строки",
            "export_success": "Данные успешно экспортированы в:\n{}",
            "export_complete": "Экспорт завершен",
            "export_failed": "Ошибка экспорта: {}",
            # Journal Window
            "journal_title": "Журнал",
            # Driver List Dialog
            "drivers_list_title": "Список водителей",
            "driver_id": "ID водителя",
            "delete_driver": "Удалить водителя",
            # Vehicle List Dialog
            "vehicles_list_title": "Список транспорта",
            "vehicle_id": "ID транспорта",
            "delete_vehicle": "Удалить транспорт",
            # User List Dialog
            "users_list_title": "Список пользователей",
            "user_id": "ID пользователя",
            "delete_user": "Удалить пользователя",
            # Add Driver Dialog
            "add_driver_title": "Добавление водителя",
            "first_name": "Имя:",
            "last_name": "Фамилия:",
            "middle_name": "Отчество:",
            "name": "ФИО:",
            "birth_date": "Дата рождения:",
            "nationality": "Национальность:",
            "add": "Добавить",
            "first_last_required": "Имя и фамилия обязательны",
            "driver_added": "Водитель успешно добавлен (ID: {})",
            "driver_add_failed": "Ошибка добавления водителя",
            "driver_add_error": "Ошибка при добавлении водителя: {}",
            # Add Vehicle Dialog
            "add_vehicle_title": "Добавление автомобиля",
            "license_plate": "Номер:",
            "color": "Цвет:",
            "type_model": "Тип/Модель:",
            "driver_id": "ID водителя:",
            "all_fields_required": "Все поля обязательны",
            "driver_id_number": "ID водителя должен быть числом",
            "vehicle_added": "Автомобиль успешно добавлен (ID: {})",
            "vehicle_add_failed": "Ошибка добавления автомобиля",
            "vehicle_add_error": "Ошибка при добавлении автомобиля: {}",
            # Add User Dialog
            "add_user_title": "Добавление пользователя",
            "username_password_required": "Имя пользователя и пароль обязательны",
            "password_min_length": "Пароль должен содержать минимум 6 символов",
            "user_added": "Пользователь успешно добавлен",
            "user_add_failed": "Ошибка добавления пользователя (возможно, имя пользователя уже существует)",
            "user_add_error": "Ошибка при добавлении пользователя: {}",
        }

        # English translations
        self.languages["en"] = {
            # Main Window
            "app_title": "License Plate Recognition System",
            "demo_mode_title": "Demo Mode",
            "demo_mode_message": "Database connection failed. Running in demo mode.\n\nError: {}\n\nPlease check your database settings in the .env file.",
            # Common
            "username": "Username:",
            "password": "Password:",
            "login": "Login",
            "cancel": "Cancel",
            "refresh": "Refresh",
            "success": "Success",
            "error": "Error",
            "failed_load_data": "Failed to load data: {}",
            "failed_update_grid": "Failed to update grid: {}",
            "failed_display_image": "Failed to display image: {}",
            # Authentication
            "auth_title": "Authentication",
            "please_enter_username": "Please enter username",
            "please_enter_password": "Please enter password",
            "invalid_credentials": "Invalid username or password",
            "auth_error": "Authentication error: {}",
            # Main UI - Grid Headers
            "vehicle": "Vehicle",
            "driver": "Driver",
            "transit_time": "Transit Time",
            "status": "Status",
            # Main UI - Buttons
            "export_to_excel": "Export to Excel",
            "open_journal": "Open Journal",
            "add_driver": "Add Driver",
            "add_vehicle": "Add Vehicle",
            "add_user": "Add User",
            "change_language": "Русский",  # Shows what language it will switch TO
            # Status Messages
            "entered": "Entered",
            "exited": "Exited",
            "access_granted": "Access granted - logged",
            "access_denied": "Vehicle not registered - access denied",
            "no_plate_recognized": "No license plate recognized",
            "processing_image": "Processing: {}",
            "recognized_plate": "Recognized: {}",
            "recognition_not_ready": "Recognition system not ready",
            "processing_failed": "Processing failed: {}",
            # Model Loading
            "checking_models": "Checking for models...",
            "downloading_resnet": "Downloading ResNet model...",
            "downloading_recognition": "Downloading recognition model...",
            "resnet_downloaded": "ResNet model downloaded successfully",
            "recognition_downloaded": "Recognition model downloaded successfully",
            "models_missing": "Some models are missing. Recognition disabled.",
            "recognition_ready": "Recognition system ready",
            "model_download_failed": "Model download failed: {}",
            # File Monitoring
            "file_watcher_started": "File watcher started on {}",
            "file_watcher_failed": "Failed to start file watcher: {}",
            # Export
            "export_dialog_title": "Export to Excel",
            "export_dialog_message": "Enter number of rows to export:",
            "export_rows_label": "Rows",
            "export_success": "Data exported successfully to:\n{}",
            "export_complete": "Export Complete",
            "export_failed": "Export failed: {}",
            # Journal Window
            "journal_title": "Journal",
            # Driver List Dialog
            "drivers_list_title": "Driver's List",
            "driver_id": "Driver ID",
            "delete_driver": "Delete Driver",
            # Vehicle List Dialog
            "vehicles_list_title": "Vehicle's List",
            "vehicle_id": "Vehicle ID",
            "delete_vehicle": "Delete Vehicle",
            # User List Dialog
            "users_list_title": "User's List",
            "user_id": "User ID",
            "delete_user": "Delete User",
            # Add Driver Dialog
            "add_driver_title": "Add Driver",
            "first_name": "First Name:",
            "last_name": "Last Name:",
            "middle_name": "Middle Name:",
            "name": "Full Name:",
            "birth_date": "Birth Date:",
            "nationality": "Nationality:",
            "add": "Add",
            "first_last_required": "First and last names are required",
            "driver_added": "Driver added successfully (ID: {})",
            "driver_add_failed": "Failed to add driver",
            "driver_add_error": "Error adding driver: {}",
            # Add Vehicle Dialog
            "add_vehicle_title": "Add Vehicle",
            "license_plate": "License Plate:",
            "color": "Color:",
            "type_model": "Type/Model:",
            "driver_id": "Driver ID:",
            "all_fields_required": "All fields are required",
            "driver_id_number": "Driver ID must be a number",
            "vehicle_added": "Vehicle added successfully (ID: {})",
            "vehicle_add_failed": "Failed to add vehicle",
            "vehicle_add_error": "Error adding vehicle: {}",
            # Add User Dialog
            "add_user_title": "Add User",
            "username_password_required": "Username and password are required",
            "password_min_length": "Password must be at least 6 characters",
            "user_added": "User added successfully",
            "user_add_failed": "Failed to add user (username may already exist)",
            "user_add_error": "Error adding user: {}",
        }

    def get_text(self, key: str, *args) -> str:
        """Get translated text for the current language."""
        text = self.languages.get(self.current_language, {}).get(key, key)
        if args:
            try:
                return text.format(*args)
            except:
                return text
        return text

    def set_language(self, language_code: str):
        """Change the current language."""
        if language_code in self.supported_languages:
            self.current_language = language_code

    def get_current_language(self) -> str:
        """Get the current language code."""
        return self.current_language

    def get_supported_languages(self) -> list:
        """Get list of supported language codes."""
        return self.supported_languages

    def toggle_language(self):
        """Toggle between supported languages."""
        if self.current_language == "ru":
            self.current_language = "en"
        else:
            self.current_language = "ru"

    def get_language_name(self, language_code: str) -> str:
        """Get the display name for a language."""
        names = {"ru": "Русский", "en": "English"}
        return names.get(language_code, language_code)


# Global language manager instance
_lang_manager = None


def get_lang_manager() -> LanguageManager:
    """Get the global language manager instance."""
    global _lang_manager
    if _lang_manager is None:
        _lang_manager = LanguageManager()
    return _lang_manager


def _(key: str, *args) -> str:
    """Shortcut function for getting translated text."""
    return get_lang_manager().get_text(key, *args)
