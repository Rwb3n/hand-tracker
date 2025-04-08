"""
Main application entry point.
Initializes all components, sets up threading, connects GUI callbacks,
and runs the main application loop.
"""

import threading
import time
import logging
import sys
import os
import traceback
import mediapipe as mp

# Ensure the src directory is in the Python path
# This allows running the script directly from the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import project modules with error handling
try:
    from src.utils.webcam import WebcamCapture
    from src.hand_tracker import HandTracker
    from src.gesture_recognizer import GestureRecognizer, Gesture
    from src.cursor_controller import CursorController
    from src.keyboard_controller import KeyboardController
    from src.ui.app_gui import AppGUI
    import keyboard # For hotkey
    # Import pyautogui here only for cleanup, to avoid potential import loops if modules also import it
    import pyautogui
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure all modules exist and dependencies are installed ('pip install -r requirements.txt').")
    # Attempt to provide more specific guidance based on the error message
    if "No module named 'src" in str(e):
         print("Hint: Try running this script from the project's root directory (e.g., 'python src/main.py').")
    elif "opencv" in str(e) or "cv2" in str(e):
         print("Hint: Ensure OpenCV is installed ('pip install opencv-python').")
    elif "mediapipe" in str(e):
         print("Hint: Ensure MediaPipe is installed ('pip install mediapipe').")
    elif "pyautogui" in str(e):
         print("Hint: Ensure PyAutoGUI is installed ('pip install pyautogui').")
    elif "keyboard" in str(e):
         print("Hint: Ensure keyboard is installed ('pip install keyboard'). Note: requires admin/root sometimes.")
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     traceback.print_exc()
     sys.exit(1)

# --- Global Variables --- #
is_tracking = False
tracking_thread = None
app_gui = None # Global reference to the GUI instance
webcam = None
hand_tracker = None
gesture_recognizer = None
cursor_controller = None
keyboard_controller = None

# --- Logging Setup --- #
# Configure logging level and format
log_level = logging.INFO # Change to logging.DEBUG for more verbose output
log_format = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format)


# --- Configuration --- #
# These could be loaded from a config file later
INPUT_WIDTH = 640
INPUT_HEIGHT = 480
TARGET_FPS = 30 # Adjust based on performance
FILTER_TYPE = 'ma' # 'ma' or 'kf'
MA_WINDOW_SIZE = 5
SCREEN_MARGIN = 30 # Pixels margin from screen edges

# Gesture timing/thresholds (can override recognizer defaults here if needed)
PINCH_THRESHOLD = 0.06
V_HOLD_DURATION = 0.4
PALM_HOLD_DURATION = 0.8
DOUBLE_CLICK_INTERVAL = 0.3
SWIPE_THRESHOLD = 0.07
SWIPE_DEBOUNCE = 0.5

# --- Main Tracking Logic --- #
def tracking_loop():
    """The core loop that captures frames, detects gestures, and controls the mouse/keyboard."""
    global is_tracking, webcam, hand_tracker, gesture_recognizer, cursor_controller, keyboard_controller, app_gui

    logging.info("Tracking loop thread started.")
    try:
        # Initialize components
        webcam = WebcamCapture(width=INPUT_WIDTH, height=INPUT_HEIGHT, target_fps=TARGET_FPS)
        hand_tracker = HandTracker(max_hands=1) # Track only one hand for simplicity
        gesture_recognizer = GestureRecognizer(
             pinch_threshold=PINCH_THRESHOLD,
             v_hold_duration=V_HOLD_DURATION,
             palm_hold_duration=PALM_HOLD_DURATION,
             double_click_interval=DOUBLE_CLICK_INTERVAL,
             swipe_threshold_x=SWIPE_THRESHOLD,
             swipe_debounce=SWIPE_DEBOUNCE
        )
        cursor_controller = CursorController(
            input_width=INPUT_WIDTH,
            input_height=INPUT_HEIGHT,
            filter_type=FILTER_TYPE,
            ma_window_size=MA_WINDOW_SIZE,
            screen_margin=SCREEN_MARGIN
        )
        keyboard_controller = KeyboardController() # Already logs privilege warnings

        logging.info("All components initialized successfully within tracking thread.")
        if app_gui:
            # Update GUI status from the main thread if possible, or queue update
            # For simplicity here, direct update (may cause issues if Tkinter not thread-safe)
            # A queue-based approach is safer for cross-thread GUI updates.
            try:
                 app_gui.update_status("Status: Components Initialized")
                 time.sleep(0.5) # Brief pause to show init message
                 app_gui.update_status("Status: Tracking Active...")
            except Exception as gui_e:
                 logging.warning(f"Error updating GUI status from thread: {gui_e}")

    except IOError as e:
         logging.error(f"Webcam initialization failed: {e}")
         if app_gui: app_gui.update_status("Error: Webcam Failed")
         is_tracking = False # Ensure loop terminates
         if app_gui: app_gui._handle_stop() # Update GUI state if start failed
         return # Stop thread execution
    except Exception as e:
        logging.error(f"Error initializing components: {e}")
        traceback.print_exc()
        if app_gui: app_gui.update_status("Error: Initialization Failed")
        is_tracking = False # Ensure loop terminates
        if app_gui: app_gui._handle_stop() # Update GUI state if start failed
        return # Stop thread execution

    # --- Main Loop --- #
    last_gesture = Gesture.NONE
    consecutive_no_hand_frames = 0

    while is_tracking:
        try:
            start_time = time.time() # For FPS calculation/debugging

            success, frame = webcam.read_frame()
            if not success or frame is None:
                logging.warning("Failed to get frame from webcam.")
                consecutive_no_hand_frames += 1
                if consecutive_no_hand_frames > 10: # Increased tolerance
                     logging.error("Webcam failed multiple times, stopping tracking.")
                     if app_gui: app_gui.update_status("Error: Webcam Stopped")
                     is_tracking = False # Signal stop
                     # Schedule GUI update from main thread if possible
                     if app_gui: app_gui.after(0, app_gui._handle_stop)
                time.sleep(0.1)
                continue

            consecutive_no_hand_frames = 0 # Reset counter on success

            # Hand Tracking
            results, _ = hand_tracker.process_frame(frame) # We don't need annotated frame here
            landmarks = hand_tracker.get_landmarks(results)

            # Gesture Recognition
            current_gesture = gesture_recognizer.recognize(landmarks)

            if landmarks:
                # Cursor Movement (using index finger tip for now, could change)
                # Using INDEX_MCP (joint 5) might be more stable than tip
                # control_point = hand_tracker.get_specific_landmark(landmarks, mp.solutions.hands.HandLandmark.INDEX_MCP)
                # Using INDEX_FINGER_TIP due to persistent runtime AttributeError with INDEX_MCP in packaged executable.
                control_point = hand_tracker.get_specific_landmark(landmarks, mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP)
                if control_point:
                    cursor_controller.update_position(control_point[0], control_point[1])
                # else: # Fallback to index tip if MCP somehow fails
                #     index_tip = hand_tracker.get_specific_landmark(landmarks, hand_tracker.INDEX_TIP)
                #     if index_tip:
                #          cursor_controller.update_position(index_tip[0], index_tip[1])

                # Perform Actions based on Gesture State Changes
                if current_gesture != last_gesture:
                    logging.info(f"Gesture changed: {last_gesture.name} -> {current_gesture.name}")

                    # --- Handle Discrete Actions (triggered once on state change) ---
                    if current_gesture == Gesture.LEFT_CLICK:
                        cursor_controller.left_click()
                    elif current_gesture == Gesture.DOUBLE_CLICK:
                         cursor_controller.double_click()
                    elif current_gesture == Gesture.RIGHT_CLICK:
                        cursor_controller.right_click()
                    elif current_gesture == Gesture.DRAG_START:
                        cursor_controller.press_left()
                    elif current_gesture == Gesture.DRAG_END:
                         cursor_controller.release_left()
                    elif current_gesture == Gesture.TAB_NEXT:
                        keyboard_controller.switch_tab_next()
                    elif current_gesture == Gesture.TAB_PREV:
                        keyboard_controller.switch_tab_prev()
                    elif current_gesture == Gesture.SWITCH_WINDOW:
                        keyboard_controller.switch_window()

            else: # No landmarks detected
                 # If hand disappears while pinching, release the drag
                 if last_gesture == Gesture.PINCHING or last_gesture == Gesture.DRAG_START:
                      cursor_controller.release_left()
                      logging.info("Drag ended due to hand loss.")

            last_gesture = current_gesture # Update last gesture for next iteration

            # Optional: FPS Calculation/Logging
            # end_time = time.time()
            # fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            # logging.debug(f"FPS: {fps:.2f}")

            # Short sleep managed by webcam.read_frame based on target_fps
            # time.sleep(0.001) # Can add a tiny sleep if CPU usage is still high

        except Exception as e:
            logging.error(f"Error in tracking loop: {e}")
            traceback.print_exc()
            # Decide whether to stop tracking on error
            # is_tracking = False
            # if app_gui: app_gui.after(0, lambda: app_gui.update_status("Error: Tracking Loop Failed"))
            time.sleep(1) # Pause briefly after an error before continuing

    # --- Cleanup --- #
    logging.info("Exiting tracking loop thread.")
    if hand_tracker: hand_tracker.close()
    if webcam: webcam.release()
    # Ensure mouse button isn't stuck down if loop exited unexpectedly
    try:
        # Check if pyautogui was successfully imported before using
        if 'pyautogui' in sys.modules:
             pyautogui.mouseUp(button='left', _pause=False) # Use _pause=False for cleanup
             pyautogui.mouseUp(button='right', _pause=False)
    except NameError:
         logging.warning("PyAutoGUI not imported, cannot perform cleanup mouseUp.")
    except Exception as cleanup_e:
        logging.warning(f"Ignoring error during cleanup mouseUp: {cleanup_e}")
    logging.info("Tracking components released.")

# --- GUI Callbacks --- #
def start_tracking():
    global is_tracking, tracking_thread
    if not is_tracking:
        is_tracking = True
        logging.info("Starting tracking thread...")
        # Use daemon=True so thread exits automatically if main program exits unexpectedly
        tracking_thread = threading.Thread(target=tracking_loop, name="TrackingThread", daemon=True)
        tracking_thread.start()
    else:
        logging.warning("Tracking is already active.")

def stop_tracking():
    global is_tracking, tracking_thread
    if is_tracking:
        is_tracking = False
        logging.info("Stop signal sent to tracking thread...")
        if tracking_thread and tracking_thread.is_alive():
            # Wait briefly for the thread to exit on its own
            tracking_thread.join(timeout=2.0)
            if tracking_thread.is_alive():
                logging.warning("Tracking thread did not terminate gracefully within timeout.")
            else:
                 logging.info("Tracking thread terminated gracefully.")
        else:
             logging.info("Tracking thread was not running or already finished.")
        tracking_thread = None
    else:
        logging.info("Tracking is not currently active.")

def exit_application():
    global app_gui
    logging.info("Exit application requested.")
    stop_tracking() # Ensure tracking stops before exiting

    # Give threads a moment to potentially clean up
    time.sleep(0.2)

    if app_gui:
        logging.info("Destroying GUI window.")
        app_gui.destroy() # Close the Tkinter window

    logging.info("Application exited via exit_application.")
    # Use os._exit for a more forceful exit if sys.exit doesn't work from threads/callbacks
    os._exit(0)
    # sys.exit(0) # Might not work reliably from all contexts

def setup_hotkey():
    """Sets up the global hotkey to exit the application."""
    try:
        # Use a lambda or functools.partial if arguments needed for callback
        keyboard.add_hotkey('ctrl+shift+q', exit_application)
        logging.info("Hotkey Ctrl+Shift+Q registered for exiting.")
    except NameError:
         logging.warning("'keyboard' module not imported successfully. Hotkey disabled.")
    except Exception as e:
        logging.error(f"Failed to set up hotkey: {e}. Try running with admin/root privileges.")

# --- Main Execution --- #
if __name__ == "__main__":
    logging.info("Application starting...")
    setup_hotkey()

    gui_thread = None
    try:
        # Run the GUI in the main thread as Tkinter prefers
        app_gui = AppGUI()
        app_gui.set_callbacks(start_tracking, stop_tracking, exit_application)

        logging.info("Starting GUI... Application will run until GUI is closed or hotkey pressed.")
        app_gui.run() # This starts the Tkinter main loop and blocks here

        # Code here will run after the GUI window is closed
        logging.info("GUI main loop finished.")
        # Ensure tracking is stopped if GUI closed manually
        stop_tracking()

    except Exception as e:
        logging.error(f"Fatal error during application execution: {e}")
        traceback.print_exc()
        # Attempt cleanup even on error
        stop_tracking()
        sys.exit(1)

    logging.info("Application finished cleanly.")
    sys.exit(0)
