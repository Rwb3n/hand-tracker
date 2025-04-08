"""
Defines the Tkinter GUI for the application.
"""

import tkinter as tk
from tkinter import ttk # For themed widgets
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AppGUI(tk.Tk):
    """Main application window using Tkinter."""

    def __init__(self, title="Hand Mouse Control", geometry="300x150"):
        super().__init__()

        self.title(title)
        self.geometry(geometry)
        self.resizable(False, False) # Keep window size fixed

        # Placeholder callbacks - these will be set by the main application logic
        self.start_callback = lambda: logging.warning("Start callback not set!")
        self.stop_callback = lambda: logging.warning("Stop callback not set!")
        self.exit_callback = lambda: logging.warning("Exit callback not set!")

        # Configure style for themed widgets
        style = ttk.Style(self)
        style.theme_use('clam') # Using a modern theme

        # Main frame
        main_frame = ttk.Frame(self, padding="10 10 10 10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Widgets --- #

        # Status Label
        self.status_var = tk.StringVar(value="Status: Idle")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.CENTER)
        status_label.pack(pady=(0, 10), fill=tk.X)

        # Buttons Frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)

        # Start Button
        self.start_button = ttk.Button(
            buttons_frame,
            text="Start Tracking",
            command=self._handle_start,
            style='Accent.TButton' # Optional styling
        )
        self.start_button.pack(side=tk.LEFT, expand=True, padx=(0, 5))

        # Stop Button
        self.stop_button = ttk.Button(
            buttons_frame,
            text="Stop Tracking",
            command=self._handle_stop,
            state=tk.DISABLED # Initially disabled
        )
        self.stop_button.pack(side=tk.LEFT, expand=True, padx=5)

        # Exit Button
        exit_button = ttk.Button(
            buttons_frame,
            text="Exit",
            command=self._handle_exit
        )
        exit_button.pack(side=tk.LEFT, expand=True, padx=(5, 0))

        # Make the window close button call our exit handler too
        self.protocol("WM_DELETE_WINDOW", self._handle_exit)

        logging.info("App GUI initialized.")

    def set_callbacks(self, start_cb, stop_cb, exit_cb):
        """Sets the callback functions for button actions."""
        self.start_callback = start_cb
        self.stop_callback = stop_cb
        self.exit_callback = exit_cb
        logging.info("GUI callbacks set.")

    def _handle_start(self):
        logging.info("Start button clicked.")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Status: Tracking...")
        self.start_callback() # Call the actual start logic

    def _handle_stop(self):
        logging.info("Stop button clicked.")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Status: Stopped")
        self.stop_callback() # Call the actual stop logic

    def _handle_exit(self):
        logging.info("Exit requested.")
        self.update_status("Status: Exiting...")
        # Ensure stop logic runs before exiting
        if self.stop_button['state'] == tk.NORMAL:
             self.stop_callback()
        self.exit_callback() # Call the actual exit logic (which should destroy window)

    def update_status(self, message):
        """Updates the status label text."""
        self.status_var.set(message)
        self.update_idletasks() # Ensure GUI updates immediately
        logging.debug(f"GUI Status updated: {message}")

    def run(self):
        """Starts the Tkinter main event loop."""
        logging.info("Starting GUI main loop.")
        self.mainloop()

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing AppGUI...")

    # Define dummy callbacks for testing
    def test_start():
        print("Test Start Callback Triggered")
        # Simulate tracking active
        # In real app, this would be handled by the main loop
        # For testing, just update status after a delay
        gui.update_status("Status: Tracking Active (Test)")

    def test_stop():
        print("Test Stop Callback Triggered")
        gui.update_status("Status: Stopped (Test)")

    def test_exit():
        print("Test Exit Callback Triggered")
        gui.destroy() # Close the window

    try:
        gui = AppGUI()
        gui.set_callbacks(test_start, test_stop, test_exit)
        gui.run() # This blocks until the window is closed
        print("GUI main loop finished.")

    except Exception as e:
        print(f"An error occurred during GUI test: {e}")
        import traceback
        traceback.print_exc()

    print("AppGUI test complete.") 