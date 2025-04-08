"""
Handles mouse cursor movement and actions using PyAutoGUI, including smoothing.
"""

import pyautogui
import numpy as np
import time
import logging
import os
import sys

# Assuming filters are in the same parent directory structure (src/utils/)
# This import might need adjustment based on how the project is run.
# Using relative import if run as part of the package
try:
    # If run as module `python -m src.cursor_controller` or imported
    from .utils.filters import KalmanFilter2D, MovingAverageFilter2D
except ImportError:
    # Fallback for running script directly `python src/cursor_controller.py`
    try:
        from utils.filters import KalmanFilter2D, MovingAverageFilter2D
    except ImportError:
         logging.error("Failed to import filter classes. Ensure utils/filters.py exists.")
         # Provide dummy classes if filters are unavailable to allow basic functionality
         class DummyFilter:
             def filter(self, p): return p
         KalmanFilter2D = MovingAverageFilter2D = DummyFilter

# Disable PyAutoGUI failsafe if necessary (move mouse to corner to stop)
# pyautogui.FAILSAFE = False
# Disable pauses between actions if needed
pyautogui.PAUSE = 0.0 # Default is 0.1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CursorController:
    """Manages cursor movement, mapping, smoothing, and actions."""

    def __init__(self, input_width=640, input_height=480, filter_type='ma', ma_window_size=5, kf_dt=0.1, kf_process_noise=1e-2, kf_measurement_noise=1e-1, screen_margin=20):
        """Initializes the cursor controller.

        Args:
            input_width (int): Width of the input coordinate space (e.g., webcam frame).
            input_height (int): Height of the input coordinate space.
            filter_type (str): Type of filter to use: 'ma' (Moving Average) or 'kf' (Kalman).
            ma_window_size (int): Window size for Moving Average filter.
            kf_dt (float): Time step for Kalman filter.
            kf_process_noise (float): Process noise std dev for Kalman filter.
            kf_measurement_noise (float): Measurement noise std dev for Kalman filter.
            screen_margin (int): Pixels to keep away from screen edges.
        """
        self.screen_width, self.screen_height = pyautogui.size()
        self.input_width = input_width
        self.input_height = input_height
        self.screen_margin = screen_margin

        logging.info(f"Screen dimensions: {self.screen_width}x{self.screen_height}")
        logging.info(f"Input dimensions: {self.input_width}x{self.input_height}")

        # Initialize selected filter
        self.filter = None
        if filter_type == 'ma':
            self.filter = MovingAverageFilter2D(window_size=ma_window_size)
            logging.info(f"Using Moving Average filter (window={ma_window_size}).")
        elif filter_type == 'kf':
            self.filter = KalmanFilter2D(dt=kf_dt, process_noise_std=kf_process_noise, measurement_noise_std=kf_measurement_noise)
            logging.info("Using Kalman filter.")
        else:
            logging.warning(f"Unknown filter type '{filter_type}'. No filter applied.")
            # Provide a dummy filter that does nothing
            class NoFilter:
                def filter(self, p):
                    return p
            self.filter = NoFilter()

        self.last_valid_pos = (self.screen_width / 2, self.screen_height / 2)

    def _map_to_screen(self, hand_x_norm, hand_y_norm):
        """Maps normalized hand coordinates [0, 1] to screen coordinates.

        Args:
            hand_x_norm (float): Normalized x-coordinate from hand tracker.
            hand_y_norm (float): Normalized y-coordinate from hand tracker.

        Returns:
            tuple: (screen_x, screen_y) coordinates.
        """
        # Invert X if needed (depends on webcam flip state)
        # screen_x_mapped = np.interp(hand_x_norm, [0, 1], [self.screen_width - self.screen_margin, self.screen_margin])
        screen_x_mapped = np.interp(hand_x_norm, [0, 1], [self.screen_margin, self.screen_width - self.screen_margin])
        screen_y_mapped = np.interp(hand_y_norm, [0, 1], [self.screen_margin, self.screen_height - self.screen_margin])

        # Clamp values just in case to prevent pyautogui errors
        screen_x = int(np.clip(screen_x_mapped, 0, self.screen_width - 1))
        screen_y = int(np.clip(screen_y_mapped, 0, self.screen_height - 1))

        return screen_x, screen_y

    def update_position(self, hand_x_norm, hand_y_norm):
        """Updates cursor position based on normalized hand coordinates.

        Args:
            hand_x_norm (float): Normalized x-coordinate (0.0 to 1.0).
            hand_y_norm (float): Normalized y-coordinate (0.0 to 1.0).
        """
        if hand_x_norm is None or hand_y_norm is None:
             # logging.debug("Invalid hand coordinates received.")
             # Optionally, could move to last known valid pos or do nothing
             # Let's move to the last valid position smoothly if None received repeatedly
             # pyautogui.moveTo(self.last_valid_pos[0], self.last_valid_pos[1], duration=0.1)
             return # Or simply do nothing if coordinates are invalid

        screen_x_raw, screen_y_raw = self._map_to_screen(hand_x_norm, hand_y_norm)

        # Apply smoothing filter
        smoothed_pos = self.filter.filter((screen_x_raw, screen_y_raw))
        smoothed_x = int(smoothed_pos[0])
        smoothed_y = int(smoothed_pos[1])

        # Clamp smoothed values as well
        smoothed_x = int(np.clip(smoothed_x, 0, self.screen_width - 1))
        smoothed_y = int(np.clip(smoothed_y, 0, self.screen_height - 1))

        # Move the mouse
        try:
            # Use duration=0 for immediate movement
            pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)
            self.last_valid_pos = (smoothed_x, smoothed_y) # Update last valid position
            # logging.debug(f"Raw:({screen_x_raw},{screen_y_raw}), Smoothed:({smoothed_x},{smoothed_y})")
        except pyautogui.FailSafeException:
            logging.warning("PyAutoGUI fail-safe triggered!")
        except Exception as e:
             logging.error(f"Error moving mouse: {e}")

    def left_click(self):
        """Performs a left mouse click."""
        try:
            pyautogui.click(button='left')
            logging.debug("Left click performed.")
        except Exception as e:
            logging.error(f"Error performing left click: {e}")

    def right_click(self):
        """Performs a right mouse click."""
        try:
            pyautogui.click(button='right')
            logging.debug("Right click performed.")
        except Exception as e:
            logging.error(f"Error performing right click: {e}")

    def double_click(self):
        """Performs a double left click."""
        try:
            pyautogui.doubleClick()
            logging.debug("Double click performed.")
        except Exception as e:
            logging.error(f"Error performing double click: {e}")

    def press_left(self):
        """Presses and holds the left mouse button down."""
        try:
            pyautogui.mouseDown(button='left')
            logging.debug("Left mouse button pressed down.")
        except Exception as e:
            logging.error(f"Error pressing left mouse button: {e}")

    def release_left(self):
        """Releases the left mouse button."""
        try:
            pyautogui.mouseUp(button='left')
            logging.debug("Left mouse button released.")
        except Exception as e:
            logging.error(f"Error releasing left mouse button: {e}")

# Example usage (for testing this module directly)
if __name__ == '__main__':
    # Correct import path assuming script is run from project root or src/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    try:
        from src.utils.filters import KalmanFilter2D, MovingAverageFilter2D
    except ModuleNotFoundError:
         logging.error("Failed (again) to import filter classes. Ensure utils/filters.py exists in src/utils.")
         sys.exit(1)

    print("Testing CursorController...")
    controller = CursorController(filter_type='ma', ma_window_size=5, screen_margin=50)
    # controller = CursorController(filter_type='kf', screen_margin=50)

    # Simulate moving in a square
    steps = 20
    delay = 0.02
    center_x, center_y = 0.5, 0.5
    radius = 0.3

    print("Simulating movement... (Move mouse to corner to stop if failsafe enabled)")
    try:
        # Move to starting point
        start_x = center_x - radius
        start_y = center_y - radius
        controller.update_position(start_x, start_y)
        time.sleep(0.5)

        path = []
        # Top edge (left to right)
        for i in range(steps):
            x = start_x + (radius * 2 * i / steps)
            y = start_y
            path.append((x,y))
        # Right edge (top to bottom)
        for i in range(steps):
            x = start_x + (radius * 2)
            y = start_y + (radius * 2 * i / steps)
            path.append((x,y))
        # Bottom edge (right to left)
        for i in range(steps):
            x = start_x + (radius * 2) - (radius * 2 * i / steps)
            y = start_y + (radius * 2)
            path.append((x,y))
        # Left edge (bottom to top)
        for i in range(steps):
            x = start_x
            y = start_y + (radius * 2) - (radius * 2 * i / steps)
            path.append((x,y))

        for x, y in path:
            controller.update_position(x, y)
            time.sleep(delay)

        print("Movement simulation finished.")
        time.sleep(0.5)

        # Test clicks
        print("Testing clicks...")
        controller.left_click()
        time.sleep(0.5)
        controller.right_click()
        time.sleep(0.5)
        controller.double_click()
        time.sleep(0.5)

        # Test drag (press, move, release)
        print("Testing drag...")
        controller.update_position(0.5, 0.5) # Move near center
        controller.press_left()
        time.sleep(0.2)
        # Simulate dragging slightly
        for i in range(10):
             controller.update_position(0.5 + i*0.01, 0.5 + i*0.01)
             time.sleep(0.05)
        controller.release_left()
        print("Drag test finished.")

    except pyautogui.FailSafeException:
        print("Fail-safe triggered during test.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    print("CursorController test complete.") 