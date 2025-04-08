"""
Handles hand detection and landmark extraction using MediaPipe.
"""

import cv2
import mediapipe as mp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HandTracker:
    """Wraps MediaPipe Hands for detection and landmark extraction."""

    def __init__(self, static_mode=False, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initializes the MediaPipe Hands model.

        Args:
            static_mode (bool): Whether to treat images as static or video stream.
            max_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence for initial detection.
            min_tracking_confidence (float): Minimum confidence for tracking.
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        # Context manager ensures resources are properly managed
        # Note: Initialization happens here, not in a separate method to use context manager
        # If persistent object needed across many frames, consider not using context manager on init
        # For this class structure, we'll initialize it within process_frame or similar
        # Or manage the context outside this class if HandTracker is long-lived.
        # Let's assume HandTracker is instantiated once and process_frame is called repeatedly.
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        # MediaPipe drawing utility
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        logging.info("MediaPipe Hands initialized.")

    def process_frame(self, frame):
        """Processes a single frame to detect hands and landmarks.

        Args:
            frame (np.ndarray): The input image frame (BGR format from OpenCV).

        Returns:
            tuple: (results, annotated_frame)
                   - results: MediaPipe Hands results object (contains landmarks).
                   - annotated_frame: Frame with landmarks drawn (or original if no hands).
        """
        # Convert the BGR image to RGB, required by MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True # Back to writeable if needed later

        # Prepare the annotated frame (copy of original)
        annotated_frame = frame.copy()

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        # else: # Optional: Log if no hands are detected
            # logging.debug("No hands detected in the frame.")

        return results, annotated_frame

    def get_landmarks(self, results, hand_index=0):
        """Extracts landmark data for a specific hand.

        Args:
            results: The MediaPipe Hands results object from process_frame.
            hand_index (int): The index of the hand to get landmarks for (default 0).

        Returns:
            list or None: A list of landmark dictionaries [{ 'id': id, 'x': x, 'y': y, 'z': z }, ...]
                          or None if no hand or specified index not found.
        """
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_index:
            hand_landmarks = results.multi_hand_landmarks[hand_index]
            landmark_list = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                # Note: x, y are normalized to [0.0, 1.0] by image width/height
                # z represents landmark depth, smaller is closer
                landmark_list.append({
                    'id': idx,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z
                })
            return landmark_list
        return None

    def get_specific_landmark(self, landmarks, landmark_id):
        """Gets coordinates of a specific landmark ID from the extracted list.

        Args:
            landmarks (list): List of landmark dictionaries from get_landmarks.
            landmark_id (int): The ID of the landmark to retrieve (e.g., 4 for Thumb Tip).

        Returns:
            tuple or None: (x, y) coordinates of the landmark, or None if not found.
        """
        if landmarks:
            for lm in landmarks:
                if lm['id'] == landmark_id:
                    return lm['x'], lm['y']
        return None

    def close(self):
        """Releases MediaPipe Hands resources."""
        # Check if hands object was initialized before closing
        if hasattr(self, 'hands') and self.hands:
             self.hands.close()
             logging.info("MediaPipe Hands resources released.")
        else:
             logging.info("MediaPipe Hands resources were already closed or not initialized.")

    def __del__(self):
        """Ensure resources are released when the object is destroyed."""
        self.close()


# Example usage (for testing this module directly)
if __name__ == '__main__':
    # Need WebcamCapture from utils for testing
    # Assuming utils directory is accessible
    import sys
    import os
    # Simple way to allow importing from parent directory if utils is there
    # Adjust path as necessary based on your project structure
    # This assumes hand_tracker.py is in src/ and utils/ is also in src/
    # If hand_tracker.py is in the root, this needs adjustment.
    # Let's assume the structure: project_root/src/hand_tracker.py and project_root/src/utils/webcam.py
    # current_dir = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(current_dir) # This would be project_root if __file__ is src/hand_tracker.py
    # utils_path = os.path.join(parent_dir, 'src', 'utils') # If run from root
    # If running src/hand_tracker.py directly, '.' should refer to src/
    utils_path = os.path.join(os.path.dirname(__file__), 'utils')
    # A more robust way if running script from anywhere:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)

    # Use try-except for cleaner error handling during import
    try:
        from src.utils.webcam import WebcamCapture
    except ModuleNotFoundError:
        print("Error: Could not import WebcamCapture.")
        print("Ensure WebcamCapture is in src/utils/webcam.py and PYTHONPATH is correct.")
        sys.exit(1)
    except ImportError as e:
        print(f"ImportError: {e}")
        sys.exit(1)

    webcam = None
    tracker = None
    try:
        webcam = WebcamCapture()
        tracker = HandTracker()
        print("HandTracker initialized. Press 'q' to quit.")

        while True:
            success, frame = webcam.read_frame()
            if not success or frame is None:
                print("Failed to capture frame.")
                time.sleep(0.5) # Avoid busy-looping if webcam fails
                continue # Try next frame

            # Process the frame
            results, annotated_frame = tracker.process_frame(frame)

            # Get landmarks for the first detected hand
            landmarks = tracker.get_landmarks(results)

            if landmarks:
                # Example: Get Thumb Tip (ID 4) and Index Finger Tip (ID 8)
                thumb_tip = tracker.get_specific_landmark(landmarks, 4)
                index_tip = tracker.get_specific_landmark(landmarks, 8)

                # Print coordinates (normalized)
                # Check if tips are found before accessing coordinates
                thumb_coords = f"x={thumb_tip[0]:.2f}, y={thumb_tip[1]:.2f}" if thumb_tip else "Not found"
                index_coords = f"x={index_tip[0]:.2f}, y={index_tip[1]:.2f}" if index_tip else "Not found"
                print(f"Thumb Tip (4): {thumb_coords}\t Index Tip (8): {index_coords}")
            else:
                 print("No landmarks detected for hand 0.")


            # Display the annotated frame
            cv2.imshow('Hand Tracking Test', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting.")
                break

    except IOError as e:
        print(f"Webcam Error: {e}")
    except ImportError as e:
        print(f"Import error during execution: {e}")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() # Print stack trace for debugging
    finally:
        print("Cleaning up resources...")
        if tracker is not None:
            tracker.close()
        if webcam is not None:
            webcam.release()
        cv2.destroyAllWindows()
        print("Resources released, exiting HandTracker test.") 