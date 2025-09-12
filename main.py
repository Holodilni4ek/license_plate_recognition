#!/usr/bin/env python3
"""
License Plate Recognition System
Main application with modular structure, proper error handling, and security fixes.
"""

import os
import sys
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
import hashlib

import cv2
import gdown
import numpy as np
import pandas as pd
import requests
import wx
import wx.adv
import wx.grid
from PIL import Image
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import my modules
from config import get_config
from database_manager import get_db_manager
from recognition import get_recognizer
from i18n import get_lang_manager, _

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MainFrame(wx.Frame):
    """Main application window with improved error handling and modular design."""

    def __init__(self, parent=None, title=None):
        self.config = get_config()
        self.db = get_db_manager()
        self.lang = get_lang_manager()

        # Set title from translation if not provided
        if title is None:
            title = self.lang.get_text("app_title")

        super().__init__(
            parent,
            id=wx.ID_ANY,
            title=title,
            pos=wx.DefaultPosition,
            size=wx.Size(self.config.ui.window_width, self.config.ui.window_height),
            style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
        )

        self.SetSizeHints(
            wx.Size(self.config.ui.min_width, self.config.ui.min_height), wx.DefaultSize
        )
        self.Centre(wx.BOTH)

        # Set icon if it exists
        if os.path.exists(self.config.paths.icon_path):
            self.SetIcon(wx.Icon(self.config.paths.icon_path, wx.BITMAP_TYPE_ICO))

        # Initialize UI components
        self.observer = None
        self.recognizer = None

        # Create UI
        self.create_ui()

        # Show login dialog
        if not self.authenticate_user():
            self.Close()
            return

        self.Show()

    def authenticate_user(self) -> bool:
        """Authenticate user with improved login dialog."""
        try:
            # Test database connection first
            self.db.execute_query("SELECT 1", ())

            dlg = LoginFrame(None)
            result = dlg.ShowModal()
            authenticated = dlg.logged_in if hasattr(dlg, "logged_in") else False
            dlg.Destroy()
            return authenticated
        except Exception as e:
            logger.warning(f"Database not available, running in demo mode: {e}")
            wx.MessageBox(
                f"Database connection failed. Running in demo mode.\n\n"
                f"Error: {e}\n\n"
                f"Please check your database settings in the .env file.",
                "Demo Mode",
                wx.OK | wx.ICON_INFORMATION,
            )
            return True  # Allow demo mode without authentication

    def create_ui(self):
        """Create the main user interface."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Top panel (image + log)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(self.create_image_panel(), 1, wx.EXPAND, 5)
        top_sizer.Add(self.create_log_panel(), 1, wx.EXPAND, 5)

        # Middle panel (data grid)
        middle_sizer = wx.BoxSizer(wx.VERTICAL)
        middle_sizer.Add(self.create_grid_panel(), 1, wx.EXPAND, 5)

        # Bottom panel (buttons)
        bottom_sizer = wx.BoxSizer(wx.VERTICAL)
        bottom_sizer.Add(self.create_buttons_panel(), 0, wx.EXPAND, 5)

        # Add all panels to main sizer
        main_sizer.Add(top_sizer, 15, wx.EXPAND, 5)
        main_sizer.Add(middle_sizer, 15, wx.EXPAND, 5)
        main_sizer.Add(bottom_sizer, 1, wx.EXPAND, 5)

        self.SetSizer(main_sizer)
        self.Layout()

        # Initialize functionality
        self.initialize_functionality()

    def create_image_panel(self) -> wx.BoxSizer:
        """Create the image display panel."""
        panel = wx.BoxSizer(wx.VERTICAL)
        self.image_display = wx.StaticBitmap(
            self, wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0
        )
        panel.Add(self.image_display, 1, wx.ALL | wx.EXPAND, 5)
        return panel

    def create_log_panel(self) -> wx.BoxSizer:
        """Create the log display panel."""
        panel = wx.BoxSizer(wx.VERTICAL)
        self.log_panel = wx.TextCtrl(
            self, wx.ID_ANY, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2
        )
        panel.Add(self.log_panel, 1, wx.EXPAND | wx.ALL, 5)
        return panel

    def create_grid_panel(self) -> wx.FlexGridSizer:
        """Create the data grid panel."""
        panel = wx.FlexGridSizer(1, 1, 0, 0)
        panel.AddGrowableRow(0)
        panel.AddGrowableCol(0)

        self.log_grid = wx.grid.Grid(
            self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0
        )
        self.setup_grid()

        panel.Add(self.log_grid, 1, wx.EXPAND | wx.ALL, 5)
        return panel

    def setup_grid(self):
        """Configure the data grid."""
        # Setup colors
        bg_color = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        self.log_grid.SetBackgroundColour(bg_color)
        self.log_grid.SetDefaultCellBackgroundColour(bg_color)
        self.log_grid.SetLabelBackgroundColour(
            wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE)
        )

        # Create grid
        self.log_grid.CreateGrid(5, 4)
        self.log_grid.SetColLabelValue(0, self.lang.get_text("vehicle"))
        self.log_grid.SetColLabelValue(1, self.lang.get_text("driver"))
        self.log_grid.SetColLabelValue(2, self.lang.get_text("transit_time"))
        self.log_grid.SetColLabelValue(3, self.lang.get_text("status"))

        # Configure grid properties
        self.log_grid.EnableEditing(False)
        self.log_grid.EnableGridLines(True)
        self.log_grid.EnableDragGridSize(False)
        self.log_grid.SetMargins(0, 0)

        # Configure columns
        self.log_grid.EnableDragColMove(False)
        self.log_grid.EnableDragColSize(True)
        self.log_grid.SetColLabelAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)

        # Configure rows
        self.log_grid.EnableDragRowSize(True)
        self.log_grid.SetRowLabelAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)
        self.log_grid.SetDefaultCellAlignment(wx.ALIGN_LEFT, wx.ALIGN_TOP)

        # Auto-size columns
        self.log_grid.AutoSizeColumns()

    def create_buttons_panel(self) -> wx.BoxSizer:
        """Create the buttons panel."""
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Export button
        self.export_button = wx.Button(
            self, wx.ID_ANY, self.lang.get_text("export_to_excel")
        )
        button_sizer.Add(self.export_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.export_button.Bind(wx.EVT_BUTTON, self.on_export_excel)

        button_sizer.AddSpacer(10)

        # Journal button
        self.journal_button = wx.Button(
            self, wx.ID_ANY, self.lang.get_text("open_journal")
        )
        button_sizer.Add(self.journal_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.journal_button.Bind(wx.EVT_BUTTON, self.on_open_journal)

        button_sizer.AddSpacer(10)

        # Date picker
        self.date_picker = wx.adv.DatePickerCtrl(
            self,
            wx.ID_ANY,
            wx.DefaultDateTime,
            wx.DefaultPosition,
            wx.DefaultSize,
            style=wx.adv.DP_DROPDOWN | wx.adv.DP_SHOWCENTURY,
        )
        button_sizer.Add(self.date_picker, 0, wx.ALL, 5)

        # Set date range
        min_date = wx.DateTime()
        min_date.ParseDate(self.config.dates.min_date)
        max_date = wx.DateTime()
        max_date.ParseDate(self.config.dates.max_date)
        self.date_picker.SetRange(min_date, max_date)
        self.date_picker.SetValue(wx.DateTime.Now())
        self.date_picker.Bind(wx.adv.EVT_DATE_CHANGED, self.on_date_change)

        button_sizer.AddSpacer(50)

        # Add driver button
        self.add_driver_button = wx.Button(
            self, wx.ID_ANY, self.lang.get_text("add_driver")
        )
        button_sizer.Add(
            self.add_driver_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5
        )
        self.add_driver_button.Bind(wx.EVT_BUTTON, self.on_driver_list)

        # Add vehicle button
        self.add_vehicle_button = wx.Button(
            self, wx.ID_ANY, self.lang.get_text("add_vehicle")
        )
        button_sizer.Add(
            self.add_vehicle_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5
        )
        self.add_vehicle_button.Bind(wx.EVT_BUTTON, self.on_vehicle_list)

        # Add user button
        self.add_user_button = wx.Button(
            self, wx.ID_ANY, self.lang.get_text("add_user")
        )
        button_sizer.Add(self.add_user_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.add_user_button.Bind(wx.EVT_BUTTON, self.on_user_list)

        # Language toggle button
        button_sizer.AddSpacer(20)
        next_lang = "English" if self.lang.current_language == "ru" else "Русский"
        self.change_language_button = wx.Button(self, wx.ID_ANY, next_lang)
        button_sizer.Add(
            self.change_language_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5
        )
        self.change_language_button.Bind(wx.EVT_BUTTON, self.on_change_language)

        return button_sizer

    def initialize_functionality(self):
        """Initialize application functionality."""
        try:
            # Redirect stdout to log panel
            sys.stdout = TextRedirector(self.log_panel)

            # Initialize recognition system
            self.initialize_recognition()

            # Start file watcher
            self.start_file_watcher()

            # Load initial data
            self.load_data_from_db()

            logger.info("Application initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            wx.MessageBox(f"Initialization failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def initialize_recognition(self):
        """Initialize the recognition system."""
        try:
            # Download models if needed
            self.download_models_async()

            # Initialize recognizer (will be done after models are downloaded)
            logger.info("Recognition system initialization started")

        except Exception as e:
            logger.error(f"Failed to initialize recognition: {e}")

    def download_models_async(self):
        """Download models in background thread."""
        threading.Thread(target=self.download_models, daemon=True).start()

    def download_models(self):
        """Download required models."""
        try:
            wx.CallAfter(self.log_message, "Checking for models...\n")

            os.makedirs(self.config.paths.models_path, exist_ok=True)

            # Download ResNet model
            if not os.path.exists(self.config.models.resnet_path):
                wx.CallAfter(self.log_message, "Downloading ResNet model...\n")
                try:
                    gdown.download(
                        self.config.models.resnet_url,
                        self.config.models.resnet_path,
                        quiet=True,
                    )
                    wx.CallAfter(
                        self.log_message, "ResNet model downloaded successfully\n"
                    )
                except Exception as e:
                    logger.error(f"Failed to download ResNet model: {e}")
                    wx.CallAfter(
                        self.log_message, f"Failed to download ResNet model: {e}\n"
                    )

            # Download recognition model
            if not os.path.exists(self.config.models.recognition_path):
                wx.CallAfter(self.log_message, "Downloading recognition model...\n")
                try:
                    gdown.download(
                        self.config.models.recognition_url,
                        self.config.models.recognition_path,
                        quiet=True,
                    )
                    wx.CallAfter(
                        self.log_message, "Recognition model downloaded successfully\n"
                    )
                except Exception as e:
                    logger.error(f"Failed to download recognition model: {e}")
                    wx.CallAfter(
                        self.log_message, f"Failed to download recognition model: {e}\n"
                    )

            # Initialize recognizer after models are ready
            if os.path.exists(self.config.models.resnet_path) and os.path.exists(
                self.config.models.recognition_path
            ):
                self.recognizer = get_recognizer()
                wx.CallAfter(self.log_message, "Recognition system ready\n")
            else:
                wx.CallAfter(
                    self.log_message, "Some models are missing. Recognition disabled.\n"
                )

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            wx.CallAfter(self.log_message, f"Model download failed: {e}\n")

    def start_file_watcher(self):
        """Start watching the plates directory for new files."""
        try:
            plates_path = self.config.paths.plates_path
            os.makedirs(plates_path, exist_ok=True)

            self.observer = Observer()
            self.observer.schedule(PlateFileHandler(self), plates_path, recursive=False)
            self.observer.start()

            wx.CallAfter(self.log_message, f"File watcher started on {plates_path}\n")

        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            wx.CallAfter(self.log_message, f"Failed to start file watcher: {e}\n")

    def process_image_file(self, file_path: str):
        """Process a new image file for license plate recognition."""
        if not self.recognizer:
            wx.CallAfter(self.log_message, "Recognition system not ready\n")
            return

        threading.Thread(
            target=self._process_image_async, args=(file_path,), daemon=True
        ).start()

    def _process_image_async(self, file_path: str):
        """Process image asynchronously."""
        try:
            wx.CallAfter(
                self.log_message, f"Processing: {os.path.basename(file_path)}\n"
            )

            # Process the image
            recognized_text, processed_image = self.recognizer.process_image(file_path)

            # Display the processed image
            if processed_image is not None:
                wx.CallAfter(self.show_image, processed_image)

            # Handle recognition results
            if recognized_text:
                wx.CallAfter(self.log_message, f"Recognized: {recognized_text}\n")

                # Check if vehicle is registered
                if self.db.is_vehicle_registered(recognized_text):
                    # Add log entry
                    if self.db.add_log_entry(recognized_text):
                        wx.CallAfter(self.log_message, "Access granted - logged\n")
                        wx.CallAfter(self.refresh_data)
                    else:
                        wx.CallAfter(self.log_message, "Failed to log entry\n")
                else:
                    wx.CallAfter(
                        self.log_message, "Vehicle not registered - access denied\n"
                    )
            else:
                wx.CallAfter(self.log_message, "No license plate recognized\n")

        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            wx.CallAfter(self.log_message, f"Processing failed: {e}\n")

    def show_image(self, image: np.ndarray):
        """Display image in the UI with proper scaling."""
        try:
            if image is None:
                return

            # Convert to PIL Image
            if len(image.shape) == 3:
                image_pil = Image.fromarray(image)
            else:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Convert to wx.Image
            wx_image = wx.Image(image_pil.size[0], image_pil.size[1])
            wx_image.SetData(image_pil.convert("RGB").tobytes())

            # Scale to fit display area
            display_size = self.image_display.GetSize()
            if display_size.width > 0 and display_size.height > 0:
                # Calculate scaling factor
                img_ratio = wx_image.GetWidth() / wx_image.GetHeight()
                display_ratio = display_size.width / display_size.height

                if img_ratio > display_ratio:
                    new_width = display_size.width
                    new_height = int(display_size.width / img_ratio)
                else:
                    new_height = display_size.height
                    new_width = int(display_size.height * img_ratio)

                wx_image = wx_image.Scale(new_width, new_height)

            # Set the image
            self.image_display.SetBitmap(wx.Bitmap(wx_image))

        except Exception as e:
            logger.error(f"Failed to display image: {e}")

    def log_message(self, message: str):
        """Add message to log panel."""
        self.log_panel.AppendText(message)

    def load_data_from_db(self, date: Optional[str] = None):
        """Load log data from database."""
        try:
            if date is None:
                date = self.date_picker.GetValue().FormatISODate()

            rows = self.db.get_log_entries(date, limit=50)

            # Update grid
            self.update_grid_data(rows)

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            wx.MessageBox(f"Failed to load data: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def update_grid_data(self, rows):
        """Update the grid with new data."""
        try:
            # Clear existing data
            if self.log_grid.GetNumberRows() > 0:
                self.log_grid.DeleteRows(0, self.log_grid.GetNumberRows())

            # Add new rows if needed
            if len(rows) > 0:
                self.log_grid.AppendRows(len(rows))

            # Populate data
            for row_idx, row_data in enumerate(rows):
                for col_idx, cell_data in enumerate(row_data):
                    self.log_grid.SetCellValue(row_idx, col_idx, str(cell_data))

            # Auto-size columns
            self.log_grid.AutoSizeColumns()

        except Exception as e:
            logger.error(f"Failed to update grid: {e}")

    def refresh_data(self):
        """Refresh the current data view."""
        self.load_data_from_db()

    # Event handlers
    def on_date_change(self, event):
        """Handle date picker change."""
        self.load_data_from_db()

    def on_export_excel(self, event):
        """Export data to Excel."""
        dlg = wx.NumberEntryDialog(
            self,
            self.lang.get_text("export_dialog_message"),
            self.lang.get_text("export_rows_label"),
            self.lang.get_text("export_dialog_title"),
            50,
            1,
            1000,
        )

        if dlg.ShowModal() == wx.ID_OK:
            try:
                row_count = dlg.GetValue()
                date = self.date_picker.GetValue().FormatISODate()

                # Get data
                rows = self.db.get_log_entries(date, limit=row_count)

                # Create DataFrame
                df = pd.DataFrame(
                    rows, columns=["Vehicle", "Driver", "Transit Time", "Status"]
                )

                # Save to Excel
                filename = f"Journal_{date}.xlsx"
                filepath = os.path.join(self.config.paths.desktop_path, filename)
                df.to_excel(filepath, index=False)

                wx.MessageBox(
                    f"Data exported successfully to:\\n{filepath}",
                    "Export Complete",
                    wx.OK | wx.ICON_INFORMATION,
                )

            except Exception as e:
                logger.error(f"Export failed: {e}")
                wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

        dlg.Destroy()

    def on_open_journal(self, event):
        """Open the full journal window."""
        journal = JournalFrame(None)
        journal.Show()

    def on_driver_list(self, event):
        """Open driver list dialog"""
        dlg = DriverListFrame(None)
        dlg.Show()

    def on_add_driver(self, event):
        """Open add driver dialog."""
        dlg = AddDriverFrame(None)
        dlg.Show()

    def on_vehicle_list(self, event):
        """Open vehicle list dialog"""
        dlg = VehicleListFrame(None)
        dlg.Show()

    def on_add_vehicle(self, event):
        """Open add vehicle dialog."""
        dlg = AddVehicleFrame(None)
        dlg.Show()

    def on_user_list(self, event):
        """Open user list dialog"""
        dlg = UserListFrame(None)
        dlg.Show()

    def on_add_user(self, event):
        """Open add user dialog."""
        dlg = AddUserFrame(None)
        dlg.Show()

    def on_change_language(self, event):
        """Handle language toggle."""
        try:
            # Toggle language
            new_lang = "en" if self.lang.current_language == "ru" else "ru"
            self.lang.set_language(new_lang)

            # Update window title
            self.SetTitle(self.lang.get_text("app_title"))

            # Update button labels
            self.export_button.SetLabel(self.lang.get_text("export_to_excel"))
            self.journal_button.SetLabel(self.lang.get_text("open_journal"))
            self.add_driver_button.SetLabel(self.lang.get_text("add_driver"))
            self.add_vehicle_button.SetLabel(self.lang.get_text("add_vehicle"))
            self.add_user_button.SetLabel(self.lang.get_text("add_user"))

            # Update grid headers
            self.log_grid.SetColLabelValue(0, self.lang.get_text("vehicle"))
            self.log_grid.SetColLabelValue(1, self.lang.get_text("driver"))
            self.log_grid.SetColLabelValue(2, self.lang.get_text("transit_time"))
            self.log_grid.SetColLabelValue(3, self.lang.get_text("status"))

            # Update language toggle button text
            next_lang = "English" if self.lang.current_language == "ru" else "Русский"
            self.change_language_button.SetLabel(next_lang)

            # Force layout update
            self.Layout()
            self.Refresh()

        except Exception as e:
            logger.error(f"Failed to change language: {e}")
            wx.MessageBox(
                f"Failed to change language: {e}", "Error", wx.OK | wx.ICON_ERROR
            )

    def __del__(self):
        """Cleanup resources."""
        if self.observer:
            self.observer.stop()
            self.observer.join()


class LoginFrame(wx.Dialog):
    """Improved login dialog with proper authentication."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("auth_title"), size=(300, 200)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create the login UI."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Username field
        username_sizer = wx.BoxSizer(wx.HORIZONTAL)
        username_label = wx.StaticText(panel, label=self.lang.get_text("username"))
        username_sizer.Add(username_label, flag=wx.RIGHT, border=8)
        self.username_ctrl = wx.TextCtrl(panel)
        username_sizer.Add(self.username_ctrl, proportion=1)
        vbox.Add(
            username_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10
        )

        vbox.Add((-1, 10))

        # Password field
        password_sizer = wx.BoxSizer(wx.HORIZONTAL)
        password_label = wx.StaticText(panel, label=self.lang.get_text("password"))
        password_sizer.Add(password_label, flag=wx.RIGHT, border=8)
        self.password_ctrl = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
        password_sizer.Add(self.password_ctrl, proportion=1)
        vbox.Add(password_sizer, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=10)

        vbox.Add((-1, 20))

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        login_btn = wx.Button(panel, label=self.lang.get_text("login"), id=wx.ID_OK)
        login_btn.SetDefault()
        cancel_btn = wx.Button(
            panel, label=self.lang.get_text("cancel"), id=wx.ID_CANCEL
        )
        button_sizer.Add(login_btn)
        button_sizer.Add(cancel_btn, flag=wx.LEFT, border=5)
        vbox.Add(button_sizer, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=10)

        panel.SetSizer(vbox)

        # Bind events
        login_btn.Bind(wx.EVT_BUTTON, self.on_login)
        cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_login(self, event):
        """Handle login attempt."""
        username = self.username_ctrl.GetValue().strip()
        password = self.password_ctrl.GetValue().strip()

        if not username:
            wx.MessageBox(
                self.lang.get_text("please_enter_username"),
                self.lang.get_text("error"),
                wx.OK | wx.ICON_ERROR,
            )
            return

        if not password:
            wx.MessageBox(
                self.lang.get_text("please_enter_password"),
                self.lang.get_text("error"),
                wx.OK | wx.ICON_ERROR,
            )
            return

        try:
            if self.db.authenticate_user(username, password):
                self.logged_in = True
                self.EndModal(wx.ID_OK)
            else:
                wx.MessageBox(
                    self.lang.get_text("invalid_credentials"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            wx.MessageBox(
                self.lang.get_text("auth_error", str(e)),
                self.lang.get_text("error"),
                wx.OK | wx.ICON_ERROR,
            )

    def on_cancel(self, event):
        """Handle cancel."""
        self.EndModal(wx.ID_CANCEL)


class JournalFrame(wx.Frame):
    """Full journal window."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("journal_title"), size=(800, 600)
        )
        self.db = get_db_manager()
        self.config = get_config()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create the journal UI."""
        # Similar to main window grid but full-featured
        # Implementation would be similar to MainFrame grid
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Add grid and controls
        self.grid = wx.grid.Grid(panel)
        self.grid.CreateGrid(10, 4)
        self.grid.SetColLabelValue(0, self.lang.get_text("vehicle"))
        self.grid.SetColLabelValue(1, self.lang.get_text("driver"))
        self.grid.SetColLabelValue(2, self.lang.get_text("transit_time"))
        self.grid.SetColLabelValue(3, self.lang.get_text("status"))

        sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(sizer)

        # Load data
        self.load_all_data()

    def load_all_data(self):
        """Load all journal data."""
        try:
            # Load recent entries
            rows = self.db.get_log_entries(wx.DateTime.Now().FormatISODate(), limit=100)

            # Update grid (similar implementation to MainFrame)
            if self.grid.GetNumberRows() > 0:
                self.grid.DeleteRows(0, self.grid.GetNumberRows())

            if len(rows) > 0:
                self.grid.AppendRows(len(rows))

                for row_idx, row_data in enumerate(rows):
                    for col_idx, cell_data in enumerate(row_data):
                        self.grid.SetCellValue(row_idx, col_idx, str(cell_data))

            self.grid.AutoSizeColumns()

        except Exception as e:
            logger.error(f"Failed to load journal data: {e}")


class DriverListFrame(wx.Frame):
    """Drivers list frame."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("drivers_list_title"), size=(500, 300)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create UI for displaying drivers list."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Drivers list
        self.list_ctrl = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.list_ctrl.InsertColumn(0, self.lang.get_text("driver_id"), width=80)
        self.list_ctrl.InsertColumn(1, self.lang.get_text("name"), width=120)
        self.list_ctrl.InsertColumn(2, self.lang.get_text("birth_date"), width=120)
        self.list_ctrl.InsertColumn(3, self.lang.get_text("nationality"), width=120)
        vbox.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 10)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        refresh_btn = wx.Button(panel, label=self.lang.get_text("refresh"))
        add_btn = wx.Button(panel, label=self.lang.get_text("add_driver"))
        delete_btn = wx.Button(panel, label=self.lang.get_text("delete_driver"))
        close_btn = wx.Button(panel, label=self.lang.get_text("cancel"))

        button_sizer.Add(refresh_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(add_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(delete_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(close_btn, 0)

        vbox.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        panel.SetSizer(vbox)

        # Bind events
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        add_btn.Bind(wx.EVT_BUTTON, self.on_add_driver)
        delete_btn.Bind(wx.EVT_BUTTON, self.on_delete_driver)
        close_btn.Bind(wx.EVT_BUTTON, self.on_close)

        # Initial load
        self.load_drivers()

    def load_drivers(self):
        """Load drivers from DB and populate list."""
        self.list_ctrl.DeleteAllItems()
        try:
            drivers = self.db.get_all_drivers()
            for driver in drivers:
                index = self.list_ctrl.InsertItem(
                    self.list_ctrl.GetItemCount(), str(driver[0])
                )
                self.list_ctrl.SetItem(index, 1, str(driver[1]))
                self.list_ctrl.SetItem(index, 2, str(driver[2]))
                self.list_ctrl.SetItem(index, 3, str(driver[3]))
        except Exception as e:
            logger.error(f"Failed to load drivers: {e}")
            wx.MessageBox(f"Error loading drivers: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_refresh(self, event):
        self.load_drivers()

    def on_add_driver(self, event):
        """Open the add udriver dialog (reuse your existing AdddDiverFrame or similar)."""
        add_frame = AddDriverFrame(self)
        add_frame.Show()
        self.load_drivers()

    def on_delete_driver(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        if selected == -1:
            wx.MessageBox("No driver selected", "Error", wx.OK | wx.ICON_ERROR)
            return

        driver_id = self.list_ctrl.GetItemText(selected)
        drivername = self.list_ctrl.GetItem(selected, 1).GetText()
        confirm = wx.MessageBox(
            f"Delete driver '{drivername}'?",
            "Confirm",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if confirm == wx.YES:
            try:
                self.db.delete_driver(int(driver_id))
                self.load_drivers()
                wx.MessageBox("Driver deleted", "Info", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                logger.error(f"Failed to delete driver: {e}")
                wx.MessageBox(
                    f"Error deleting driver: {e}", "Error", wx.OK | wx.ICON_ERROR
                )

    def on_close(self, event):
        self.Close()


class VehicleListFrame(wx.Frame):
    """Vehicles list frame."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("vehicles_list_title"), size=(500, 300)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create UI for displaying vehicles list."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Vehicles list
        self.list_ctrl = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.list_ctrl.InsertColumn(0, self.lang.get_text("vehicle_id"), width=100)
        self.list_ctrl.InsertColumn(1, self.lang.get_text("type_model")[:-1], width=80)
        self.list_ctrl.InsertColumn(1, self.lang.get_text("color")[:-1], width=80)
        self.list_ctrl.InsertColumn(
            1, self.lang.get_text("license_plate")[:-1], width=80
        )
        vbox.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 10)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        refresh_btn = wx.Button(panel, label=self.lang.get_text("refresh"))
        add_btn = wx.Button(panel, label=self.lang.get_text("add_vehicle"))
        delete_btn = wx.Button(panel, label=self.lang.get_text("delete_vehicle"))
        close_btn = wx.Button(panel, label=self.lang.get_text("cancel"))

        button_sizer.Add(refresh_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(add_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(delete_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(close_btn, 0)

        vbox.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        panel.SetSizer(vbox)

        # Bind events
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        add_btn.Bind(wx.EVT_BUTTON, self.on_add_vehicle)
        delete_btn.Bind(wx.EVT_BUTTON, self.on_delete_vehicle)
        close_btn.Bind(wx.EVT_BUTTON, self.on_close)

        # Initial load
        self.load_vehicles()

    def load_vehicles(self):
        """Load vehicles from DB and populate list."""
        self.list_ctrl.DeleteAllItems()
        try:
            vehicles = self.db.get_all_vehicles()
            for vehicle in vehicles:
                index = self.list_ctrl.InsertItem(
                    self.list_ctrl.GetItemCount(), str(vehicle[0])
                )
                self.list_ctrl.SetItem(index, 1, vehicle[1])
                self.list_ctrl.SetItem(index, 2, vehicle[2])
                self.list_ctrl.SetItem(index, 3, vehicle[3])
        except Exception as e:
            logger.error(f"Failed to load vehicles: {e}")
            wx.MessageBox(
                f"Error loading vehicles: {e}", "Error", wx.OK | wx.ICON_ERROR
            )

    def on_refresh(self, event):
        self.load_vehicles()

    def on_add_vehicle(self, event):
        """Open the add vehicle dialog (reuse your existing AddVehicleFrame or similar)."""
        add_frame = AddVehicleFrame(self)
        add_frame.Show()
        self.load_vehicles()

    def on_delete_vehicle(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        if selected == -1:
            wx.MessageBox("No vehicle selected", "Error", wx.OK | wx.ICON_ERROR)
            return

        vehicle_id = self.list_ctrl.GetItemText(selected)
        vehiclename = self.list_ctrl.GetItem(selected, 1).GetText()
        confirm = wx.MessageBox(
            f"Delete vehicle '{vehiclename}'?",
            "Confirm",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if confirm == wx.YES:
            try:
                self.db.delete_vehicle(int(vehicle_id))
                self.load_vehicles()
                wx.MessageBox("Vehicle deleted", "Info", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                logger.error(f"Failed to delete vehicle: {e}")
                wx.MessageBox(
                    f"Error deleting vehicle: {e}", "Error", wx.OK | wx.ICON_ERROR
                )

    def on_close(self, event):
        self.Close()


class UserListFrame(wx.Frame):
    """Users list frame."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("users_list_title"), size=(500, 300)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create UI for displaying users list."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Users list
        self.list_ctrl = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        self.list_ctrl.InsertColumn(0, self.lang.get_text("user_id"), width=100)
        self.list_ctrl.InsertColumn(1, self.lang.get_text("username"), width=200)
        vbox.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 10)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        refresh_btn = wx.Button(panel, label=self.lang.get_text("refresh"))
        add_btn = wx.Button(panel, label=self.lang.get_text("add_user"))
        delete_btn = wx.Button(panel, label=self.lang.get_text("delete_user"))
        close_btn = wx.Button(panel, label=self.lang.get_text("cancel"))

        button_sizer.Add(refresh_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(add_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(delete_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(close_btn, 0)

        vbox.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        panel.SetSizer(vbox)

        # Bind events
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        add_btn.Bind(wx.EVT_BUTTON, self.on_add_user)
        delete_btn.Bind(wx.EVT_BUTTON, self.on_delete_user)
        close_btn.Bind(wx.EVT_BUTTON, self.on_close)

        # Initial load
        self.load_users()

    def load_users(self):
        """Load users from DB and populate list."""
        self.list_ctrl.DeleteAllItems()
        try:
            users = self.db.get_all_users()
            for user in users:
                index = self.list_ctrl.InsertItem(
                    self.list_ctrl.GetItemCount(), str(user[0])
                )
                self.list_ctrl.SetItem(index, 1, user[1])
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
            wx.MessageBox(f"Error loading users: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_refresh(self, event):
        self.load_users()

    def on_add_user(self, event):
        """Open the add user dialog (reuse your existing AddUserFrame or similar)."""
        add_frame = AddUserFrame(self)
        add_frame.Show()
        self.load_users()

    def on_delete_user(self, event):
        selected = self.list_ctrl.GetFirstSelected()
        if selected == -1:
            wx.MessageBox("No user selected", "Error", wx.OK | wx.ICON_ERROR)
            return

        user_id = self.list_ctrl.GetItemText(selected)
        username = self.list_ctrl.GetItem(selected, 1).GetText()
        confirm = wx.MessageBox(
            f"Delete user '{username}'?",
            "Confirm",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if confirm == wx.YES:
            try:
                self.db.delete_user(int(user_id))
                self.load_users()
                wx.MessageBox("User deleted", "Info", wx.OK | wx.ICON_INFORMATION)
            except Exception as e:
                logger.error(f"Failed to delete user: {e}")
                wx.MessageBox(
                    f"Error deleting user: {e}", "Error", wx.OK | wx.ICON_ERROR
                )

    def on_close(self, event):
        self.Close()


class AddDriverFrame(wx.Frame):
    """Add driver dialog with proper database integration."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("add_driver_title"), size=(400, 300)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create the add driver UI."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Form fields
        form_sizer = wx.FlexGridSizer(5, 2, 10, 10)

        # First name
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("first_name")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.firstname_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.firstname_ctrl, 1, wx.EXPAND)

        # Last name
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("last_name")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.lastname_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.lastname_ctrl, 1, wx.EXPAND)

        # Middle name
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("middle_name")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.middlename_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.middlename_ctrl, 1, wx.EXPAND)

        # Birth date
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("birth_date")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.birthdate_ctrl = wx.adv.DatePickerCtrl(panel, style=wx.adv.DP_DROPDOWN)
        form_sizer.Add(self.birthdate_ctrl, 1, wx.EXPAND)

        # Nationality
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("nationality")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.nationality_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.nationality_ctrl, 1, wx.EXPAND)

        form_sizer.AddGrowableCol(1, 1)
        vbox.Add(form_sizer, 1, wx.EXPAND | wx.ALL, 15)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        add_btn = wx.Button(panel, label=self.lang.get_text("add"))
        cancel_btn = wx.Button(panel, label=self.lang.get_text("cancel"))
        button_sizer.Add(add_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(cancel_btn, 0)
        vbox.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        panel.SetSizer(vbox)

        # Bind events
        add_btn.Bind(wx.EVT_BUTTON, self.on_add)
        cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_add(self, event):
        """Add the driver to database."""
        try:
            firstname = self.firstname_ctrl.GetValue().strip()
            lastname = self.lastname_ctrl.GetValue().strip()
            middlename = self.middlename_ctrl.GetValue().strip()
            birthdate = self.birthdate_ctrl.GetValue().FormatISODate()
            nationality = self.nationality_ctrl.GetValue().strip()

            if not firstname or not lastname:
                wx.MessageBox(
                    self.lang.get_text("first_last_required"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )
                return

            driver_id = self.db.add_driver(
                firstname, lastname, middlename, birthdate, nationality
            )

            if driver_id:
                wx.MessageBox(
                    self.lang.get_text("driver_added", driver_id),
                    self.lang.get_text("success"),
                    wx.OK | wx.ICON_INFORMATION,
                )
                self.Close()
            else:
                wx.MessageBox(
                    self.lang.get_text("driver_add_failed"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )

        except Exception as e:
            logger.error(f"Failed to add driver: {e}")
            wx.MessageBox(
                self.lang.get_text("driver_add_error", str(e)),
                self.lang.get_text("error"),
                wx.OK | wx.ICON_ERROR,
            )

    def on_cancel(self, event):
        """Cancel and close."""
        self.Close()


class AddVehicleFrame(wx.Frame):
    """Add vehicle dialog."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("add_vehicle_title"), size=(400, 300)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create the add vehicle UI."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Form fields
        form_sizer = wx.FlexGridSizer(4, 2, 10, 10)

        # License plate
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("license_plate")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.plate_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.plate_ctrl, 1, wx.EXPAND)

        # Color
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("color")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.color_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.color_ctrl, 1, wx.EXPAND)

        # Type/Model
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("type_model")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.type_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.type_ctrl, 1, wx.EXPAND)

        # Driver (you'd typically populate this from database)
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("driver_id")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.driver_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.driver_ctrl, 1, wx.EXPAND)

        form_sizer.AddGrowableCol(1, 1)
        vbox.Add(form_sizer, 1, wx.EXPAND | wx.ALL, 15)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        add_btn = wx.Button(panel, label=self.lang.get_text("add"))
        cancel_btn = wx.Button(panel, label=self.lang.get_text("cancel"))
        button_sizer.Add(add_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(cancel_btn, 0)
        vbox.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        panel.SetSizer(vbox)

        # Bind events
        add_btn.Bind(wx.EVT_BUTTON, self.on_add)
        cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_add(self, event):
        """Add the vehicle to database."""
        try:
            plate = self.plate_ctrl.GetValue().strip().upper()
            color = self.color_ctrl.GetValue().strip()
            vehicle_type = self.type_ctrl.GetValue().strip()
            driver_id_str = self.driver_ctrl.GetValue().strip()

            if not all([plate, color, vehicle_type, driver_id_str]):
                wx.MessageBox(
                    self.lang.get_text("all_fields_required"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )
                return

            try:
                driver_id = int(driver_id_str)
            except ValueError:
                wx.MessageBox(
                    self.lang.get_text("driver_id_number"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )
                return

            vehicle_id = self.db.add_vehicle(plate, color, vehicle_type, driver_id)

            if vehicle_id:
                wx.MessageBox(
                    self.lang.get_text("vehicle_added", vehicle_id),
                    self.lang.get_text("success"),
                    wx.OK | wx.ICON_INFORMATION,
                )
                self.Close()
            else:
                wx.MessageBox(
                    self.lang.get_text("vehicle_add_failed"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )

        except Exception as e:
            logger.error(f"Failed to add vehicle: {e}")
            wx.MessageBox(
                self.lang.get_text("vehicle_add_error", str(e)),
                self.lang.get_text("error"),
                wx.OK | wx.ICON_ERROR,
            )

    def on_cancel(self, event):
        """Cancel and close."""
        self.Close()


class AddUserFrame(wx.Frame):
    """Add user dialog."""

    def __init__(self, parent):
        self.lang = get_lang_manager()
        super().__init__(
            parent, title=self.lang.get_text("add_user_title"), size=(400, 200)
        )
        self.db = get_db_manager()
        self.create_ui()
        self.Centre()

    def create_ui(self):
        """Create the add user UI."""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Form fields
        form_sizer = wx.FlexGridSizer(3, 2, 10, 10)

        # Username
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("username")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.username_ctrl = wx.TextCtrl(panel)
        form_sizer.Add(self.username_ctrl, 1, wx.EXPAND)

        # Password
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("password")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.password_ctrl = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
        form_sizer.Add(self.password_ctrl, 1, wx.EXPAND)

        form_sizer.AddGrowableCol(1, 1)
        vbox.Add(form_sizer, 1, wx.EXPAND | wx.ALL, 15)

        # Super user checkbox (optional)
        form_sizer.Add(
            wx.StaticText(panel, label=self.lang.get_text("super_user")),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        self.superuser_chk = wx.CheckBox(panel)
        form_sizer.Add(self.superuser_chk, 1, wx.EXPAND)

        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        add_btn = wx.Button(panel, label=self.lang.get_text("add_user"))
        cancel_btn = wx.Button(panel, label=self.lang.get_text("cancel"))
        button_sizer.Add(add_btn, 0, wx.RIGHT, 5)
        button_sizer.Add(cancel_btn, 0)
        vbox.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)

        panel.SetSizer(vbox)

        # Bind events
        add_btn.Bind(wx.EVT_BUTTON, self.on_add)
        cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)

    def on_add(self, event):
        """Add the user to database."""

        try:
            username = self.username_ctrl.GetValue().strip()
            password = self.password_ctrl.GetValue().strip()

            if not username or not password:
                wx.MessageBox(
                    self.lang.get_text("username_password_required"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )
                return

            if len(password) < 6:
                wx.MessageBox(
                    self.lang.get_text("password_min_length"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )
                return

            if self.superuser_chk.GetValue():
                token = self.db.get_token()
                user_id = self.db.add_superuser(username, password, token)
            else:
                user_id = self.db.add_user(username, password)

            if user_id:
                wx.MessageBox(
                    self.lang.get_text("user_added"),
                    self.lang.get_text("success"),
                    wx.OK | wx.ICON_INFORMATION,
                )
                self.Close()
            else:
                wx.MessageBox(
                    self.lang.get_text("user_add_failed"),
                    self.lang.get_text("error"),
                    wx.OK | wx.ICON_ERROR,
                )

        except Exception as e:
            logger.error(f"Failed to add user: {e}")
            wx.MessageBox(f"Error adding user: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_cancel(self, event):
        """Cancel and close."""
        self.Close()


# Helper classes
class TextRedirector:
    """Redirect text output to a wx.TextCtrl."""

    def __init__(self, text_ctrl):
        self.text_ctrl = text_ctrl

    def write(self, text):
        wx.CallAfter(self.text_ctrl.AppendText, text)

    def flush(self):
        pass


class PlateFileHandler(FileSystemEventHandler):
    """Handle new files in the plates directory."""

    def __init__(self, main_frame):
        self.main_frame = main_frame

    def on_created(self, event):
        """Handle file creation."""
        if not event.is_directory and event.src_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            wx.CallAfter(self.main_frame.process_image_file, event.src_path)


def main():
    """Main application entry point."""
    try:
        # Validate configuration
        config = get_config()
        if not config.validate():
            print("Configuration validation failed")
            return 1

        # Create and run application
        app = wx.App(False)
        frame = MainFrame()
        app.MainLoop()

        return 0

    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
