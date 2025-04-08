# Design Document - Hand Mouse Control

## 1. Overview

This application allows controlling the computer mouse and basic keyboard shortcuts using hand gestures detected via a standard webcam. It is packaged as a standalone executable for Windows.

## 2. Architecture

The application follows a modular design:

- **`main.py`**: Entry point, orchestrates components, manages the main tracking thread, and handles GUI interactions.
- **`ui/app_gui.py`**: Implements the Tkinter-based graphical user interface (Start/Stop/Exit buttons, status display).
- **`utils/webcam.py`**: Handles webcam initialization, frame capture (using OpenCV), and basic FPS control.
- **`hand_tracker.py`**: Uses MediaPipe Hands to detect hand landmarks in captured frames.
- **`utils/filters.py`**: Provides smoothing filters (Kalman, Moving Average) to stabilize coordinate data.
- **`cursor_controller.py`**: Maps normalized hand coordinates to screen coordinates, applies smoothing (using filters from `utils/filters.py`), and simulates mouse movements and clicks (using PyAutoGUI).
- **`gesture_recognizer.py`**: Interprets sequences of hand landmark positions and timings to detect specific gestures (Pinch, V-Hold, Swipe, Palm Hold).
- **`keyboard_controller.py`**: Simulates keyboard shortcuts (Ctrl+Tab, Alt+Tab) associated with certain gestures (using the keyboard library).

## 3. Core Workflow (Tracking Loop in `main.py`)

1.  **Initialization**: A separate thread (`tracking_loop`) initializes `WebcamCapture`, `HandTracker`, `GestureRecognizer`, `CursorController`, and `KeyboardController`.
2.  **Frame Capture**: Reads a frame from `WebcamCapture`.
3.  **Hand Detection**: Passes the frame to `HandTracker` to get landmark results.
4.  **Landmark Extraction**: Extracts the list of landmarks (normalized coordinates).
5.  **Gesture Recognition**: Passes landmarks to `GestureRecognizer` to determine the current gesture state.
6.  **Cursor Update**: If landmarks are present, passes a stable landmark's (e.g., Index MCP) normalized coordinates to `CursorController`, which maps, smooths, and moves the mouse.
7.  **Action Execution**: Based on changes in the recognized gesture state, calls appropriate methods in `CursorController` (clicks, drag) or `KeyboardController` (shortcuts).
8.  **Loop**: Repeats steps 2-7 until the `is_tracking` flag is set to `False` (via GUI or hotkey).
9.  **Cleanup**: Releases webcam and MediaPipe resources when the loop terminates.

## 4. Key Features & Decisions

- **Threading**: The core tracking logic runs in a separate thread to prevent blocking the GUI.
- **Smoothing**: Kalman or Moving Average filters are used in `CursorController` to reduce jitter from hand tracking.
- **Gesture State**: `GestureRecognizer` maintains internal state (timing, previous gesture) to handle holds, double clicks, and swipe debouncing.
- **Coordinate Mapping**: `CursorController` maps normalized coordinates from the fixed webcam input size to the actual screen resolution, with a margin.
- **Control Point**: Using the Index MCP joint (Landmark 5) as the primary control point for cursor movement for potential stability over the fingertip.
- **Dependencies**: Uses well-established libraries (OpenCV, MediaPipe, PyAutoGUI, keyboard, Tkinter) for core functionality.
- **Packaging**: Designed for packaging into a single executable using PyInstaller.
- **Exit**: Supports exit via GUI button and `Ctrl+Shift+Q` hotkey. 