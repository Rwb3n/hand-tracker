"""
Handles webcam access and frame capturing using OpenCV.
"""
import cv2
import time
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebcamCapture:
    """Manages webcam initialization, frame reading, and resource release."""

    def __init__(self, device_index=0, width=640, height=480, target_fps=20):
        """Initializes the webcam capture.

        Args:
            device_index (int): Index of the webcam device.
            width (int): Desired frame width.
            height (int): Desired frame height.
            target_fps (int): Target frames per second (used for delay calculation).
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps if target_fps > 0 else 0
        self.cap = None
        self._initialize_capture()

    def _initialize_capture(self):
        """Initializes or re-initializes the VideoCapture object."""
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            logging.error(f"Error: Could not open webcam device index {self.device_index}.")
            self.cap = None # Ensure cap is None if failed
            # Consider raising an exception or specific handling
            raise IOError(f"Cannot open webcam {self.device_index}")

        # Set desired frame properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Note: Setting CAP_PROP_FPS is often unreliable; we control FPS via sleep
        logging.info(f"Webcam {self.device_index} initialized ({self.width}x{self.height}).")

    def read_frame(self):
        """Reads a single frame from the webcam.

        Returns:
            tuple: (success_flag, frame) where success_flag is boolean
                   and frame is the captured image (numpy array) or None.
        """
        if not self.cap or not self.cap.isOpened():
            logging.warning("Attempted to read from uninitialized or closed webcam.")
            # Try to re-initialize in case it was disconnected
            try:
                self._initialize_capture()
                if not self.cap: return False, None # Still failed
            except IOError:
                 return False, None # Re-initialization failed

        # Simple FPS control using sleep
        time.sleep(self.frame_delay)

        success, frame = self.cap.read()
        if not success:
            logging.warning("Failed to read frame from webcam.")
            return False, None

        # Optional: Flip frame horizontally for a more natural mirror effect
        frame = cv2.flip(frame, 1)
        return True, frame

    def release(self):
        """Releases the webcam resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logging.info(f"Webcam {self.device_index} released.")
        self.cap = None

    def __del__(self):
        """Ensure webcam is released when the object is destroyed."""
        self.release()

# Example usage (for testing this module directly)
if __name__ == '__main__':
    try:
        webcam = WebcamCapture(target_fps=30)
        print("Webcam initialized. Press 'q' to quit.")

        while True:
            success, frame = webcam.read_frame()
            if not success or frame is None:
                print("Failed to capture frame, exiting.")
                break

            # Display the frame (optional)
            cv2.imshow('Webcam Test', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except IOError as e:
        print(f"Initialization Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'webcam' in locals():
            webcam.release() # Explicitly release
        cv2.destroyAllWindows()
        print("Resources released, exiting test.") 