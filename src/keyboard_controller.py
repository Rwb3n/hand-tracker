"""
Handles keyboard actions (e.g., Alt+Tab, Ctrl+Tab) using the keyboard library.
"""

import keyboard
import time
import logging
import platform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeyboardController:
    """Manages keyboard shortcut simulation."""

    def __init__(self, delay_after_press=0.1):
        """Initializes the keyboard controller.

        Args:
            delay_after_press (float): Small delay after sending keys to ensure registration.
        """
        self.delay = delay_after_press
        # Check if running with root/admin privileges, required for keyboard hook on some OS
        self._check_privileges()

    def _check_privileges(self):
        """Warns if not running with sufficient privileges for keyboard library."""
        try:
            # This is a simple heuristic. More robust checks might be needed.
            if platform.system() == "Windows":
                import ctypes
                if not ctypes.windll.shell32.IsUserAnAdmin():
                    logging.warning("Keyboard library might require admin privileges on Windows for some features.")
            elif platform.system() == "Linux":
                import os
                if os.geteuid() != 0:
                    logging.warning("Keyboard library might require root privileges on Linux for system-wide hooks.")
            # Add check for macOS if needed
        except Exception as e:
            logging.warning(f"Could not check privileges: {e}")

    def _send_keys(self, keys):
        """Sends key combination with error handling and delay."""
        try:
            # keyboard.send() is often more reliable for combinations
            keyboard.send(keys)
            # Or using press_and_release if send doesn't work reliably
            # keyboard.press_and_release(keys)
            time.sleep(self.delay) # Ensure keys are registered by OS
            logging.debug(f"Sent keys: {keys}")
            return True
        except Exception as e:
            # Permissions errors are common if not run as admin/root
            logging.error(f"Failed to send keys '{keys}'. Error: {e}. Try running with admin/root privileges.")
            return False

    def switch_tab_next(self):
        """Simulates Ctrl+Tab to switch to the next browser/application tab."""
        logging.info("Action: Switch Tab Next (Ctrl+Tab)")
        # Note: 'ctrl+tab' might need admin privileges
        return self._send_keys('ctrl+tab')

    def switch_tab_prev(self):
        """Simulates Ctrl+Shift+Tab to switch to the previous tab."""
        logging.info("Action: Switch Tab Previous (Ctrl+Shift+Tab)")
        return self._send_keys('ctrl+shift+tab')

    def switch_window(self):
        """Simulates Alt+Tab to switch application windows."""
        logging.info("Action: Switch Window (Alt+Tab)")
        # Alt+Tab often requires special handling or privileges
        # Using press/release individually might be more reliable here
        try:
            keyboard.press('alt')
            time.sleep(0.05)
            keyboard.press_and_release('tab')
            time.sleep(0.05)
            keyboard.release('alt')
            time.sleep(self.delay)
            logging.debug("Sent keys: Alt+Tab (manual press/release)")
            return True
        except Exception as e:
            logging.error(f"Failed to send keys 'Alt+Tab'. Error: {e}. Try running with admin/root privileges.")
            # Attempt fallback with send
            # return self._send_keys('alt+tab')
            return False


# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing KeyboardController...")
    print("NOTE: This test will send actual key presses to your system!")
    print("Ensure you have an application with tabs (like a browser) open.")
    print("Starting in 5 seconds...")
    time.sleep(5)

    controller = KeyboardController()

    print("\nTesting Tab Switching...")
    print("Sending Ctrl+Tab (Next Tab)... You should see the tab change.")
    if controller.switch_tab_next():
        print("Ctrl+Tab sent.")
    else:
        print("Failed to send Ctrl+Tab.")
    time.sleep(2)

    print("\nSending Ctrl+Shift+Tab (Previous Tab)... You should see the tab change back.")
    if controller.switch_tab_prev():
        print("Ctrl+Shift+Tab sent.")
    else:
        print("Failed to send Ctrl+Shift+Tab.")
    time.sleep(2)

    print("\nTesting Window Switching...")
    print("Sending Alt+Tab... You should see the window switcher.")
    if controller.switch_window():
        print("Alt+Tab sent.")
    else:
        print("Failed to send Alt+Tab.")
    time.sleep(2)

    print("\nKeyboardController test finished.")
    print("Remember this might require running the script with admin/root privileges.") 