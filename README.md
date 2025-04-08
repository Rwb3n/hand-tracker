# Hand Mouse Control

A Python application that allows you to control your computer's mouse cursor using hand gestures detected via your webcam.

## Features

-   **Cursor Control:** Move the mouse by moving your hand.
-   **Gestures:**
    -   **Left Click:** Pinch thumb and index finger.
    -   **Right Click:** Hold a 'V' gesture (index and middle fingers).
    -   **Double Click:** Pinch twice rapidly.
    -   **Drag & Drop:** Pinch and hold to start dragging, release pinch to drop.
    -   **Switch Tab (Next):** Swipe open palm to the right.
    -   **Switch Tab (Previous):** Swipe open palm to the left.
    -   **Switch Window (Alt+Tab):** Hold an open palm steady.
-   **GUI:** Simple interface to start/stop tracking.
-   **Standalone Executable:** Packaged using PyInstaller for easy distribution (no Python installation required).

## Dependencies

-   Python 3.8+
-   OpenCV (`opencv-python`)
-   MediaPipe (`mediapipe`)
-   PyAutoGUI (`pyautogui`)
-   Keyboard (`keyboard`)
-   Tkinter (usually included with Python)

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

*(Note: `requirements.txt` should be created if not present)*

## Running from Source

```bash
python src/main.py
```

## Building the Executable

This project uses PyInstaller to create a standalone executable for Windows.

1.  **Install PyInstaller:**
    ```bash
    pip install pyinstaller
    ```
2.  **Clean previous builds (optional):**
    Delete the `build/` and `dist/` folders if they exist.
3.  **Run PyInstaller using the spec file:**
    The `hand_mouse_control.spec` file contains the necessary configuration, including bundling the required MediaPipe models.
    ```bash
p    python -m PyInstaller hand_mouse_control.spec
    ```
4.  **Find the Executable:**
    The standalone executable will be located in the `dist/hand_mouse_control/` directory (`hand_mouse_control.exe`).

### How the Build Was Achieved (PyInstaller Challenges)

Creating the executable involved several steps to correctly bundle the MediaPipe models:

1.  **Initial Attempts:** Directly adding the `mediapipe` folder or using `hook-mediapipe.py` did not reliably include the necessary `.tflite` and `.binarypb` model files.
2.  **Explicit `datas`:** The `hand_mouse_control.spec` file was modified to explicitly list the required model files from `mediapipe/modules/hand_landmark` and `mediapipe/modules/palm_detection`.
3.  **Correct `site-packages` Path:** A helper function (`get_site_packages_path`) was implemented and refined in the spec file to dynamically find the correct Python `site-packages` directory where MediaPipe was installed, resolving issues where PyInstaller looked in the wrong location (e.g., `AppData/Roaming` vs. the main Python installation).
4.  **Bundling:** The spec file now correctly adds these files to the `datas` list, ensuring they are copied to the appropriate `_internal/mediapipe/modules/...` path within the executable bundle.

## Running the Executable

Navigate to the `dist/hand_mouse_control/` directory and double-click `hand_mouse_control.exe`.

Use the GUI to start and stop tracking.

## Known Issues

-   **Landmark for Cursor Control:** The application currently uses the `INDEX_FINGER_TIP` landmark for cursor control. Attempts to use the potentially more stable `INDEX_MCP` landmark resulted in a persistent `AttributeError: INDEX_MCP` originating from `enum.py` *only* within the PyInstaller executable environment. The reason for this specific runtime error remains unclear, so `INDEX_FINGER_TIP` is used as a reliable workaround.
-   **Keyboard Library Permissions:** The `keyboard` library might require administrator privileges on Windows to function correctly for actions like switching tabs/windows.

## License

*(Consider adding a LICENSE file - e.g., MIT License)* 