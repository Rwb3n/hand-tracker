"""
Recognizes specific hand gestures based on MediaPipe landmark data.
"""

import numpy as np
import math
import time
from enum import Enum, auto
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Gesture(Enum):
    NONE = auto()
    MOVE = auto() # Default action when a hand is detected but no specific gesture
    LEFT_CLICK = auto() # Pinch released quickly
    RIGHT_CLICK = auto() # V gesture held
    DOUBLE_CLICK = auto() # Two quick pinches
    DRAG_START = auto() # Pinch started
    DRAG_END = auto() # Pinch released after holding
    TAB_NEXT = auto() # Swipe Right (open palm)
    TAB_PREV = auto() # Swipe Left (open palm)
    SWITCH_WINDOW = auto() # Static Open Palm held
    # Intermediate states might be useful internally
    PINCHING = auto()
    HOLDING_V = auto()
    HOLDING_PALM = auto()

class GestureRecognizer:
    """Recognizes gestures from hand landmarks."""

    # Landmark IDs
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    def __init__(self,
                 pinch_threshold=0.05, # Normalized distance for pinch
                 v_hold_duration=0.5, # Seconds to hold V for right click
                 palm_hold_duration=1.0, # Seconds to hold open palm for Alt+Tab
                 double_click_interval=0.3, # Max seconds between clicks for double click
                 swipe_threshold_x=0.08, # Min horizontal normalized distance change for swipe
                 swipe_debounce=0.5 # Min seconds between swipes
                 ):
        self.pinch_threshold = pinch_threshold
        self.v_hold_duration = v_hold_duration
        self.palm_hold_duration = palm_hold_duration
        self.double_click_interval = double_click_interval
        self.swipe_threshold_x = swipe_threshold_x
        self.swipe_debounce = swipe_debounce

        # State variables
        self.last_gesture = Gesture.NONE
        self.pinch_start_time = None
        self.last_click_time = None
        self.is_pinching = False
        self.v_gesture_start_time = None
        self.is_holding_v = False
        self.palm_start_time = None
        self.is_holding_palm = False
        self.last_wrist_pos = None
        self.last_swipe_time = 0
        self.last_landmarks = None # Store previous frame's landmarks for velocity/swipe

    def _get_landmark_pos(self, landmarks, landmark_id):
        """Safely get the (x, y) position of a landmark."""
        if landmarks and 0 <= landmark_id < len(landmarks):
            return landmarks[landmark_id]['x'], landmarks[landmark_id]['y']
        return None

    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points (tuples or lists)."""
        if p1 is None or p2 is None:
            return float('inf')
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _is_finger_up(self, landmarks, finger_tip_id, finger_pip_id):
        """Check if a finger is likely extended (tip is further from wrist than pip)."""
        # A simple heuristic: check if tip's y is lower (higher on screen) than pip's y
        # Assumes standard hand orientation (fingers pointing up/away)
        tip_pos = self._get_landmark_pos(landmarks, finger_tip_id)
        pip_pos = self._get_landmark_pos(landmarks, finger_pip_id)
        # wrist_pos = self._get_landmark_pos(landmarks, self.WRIST)

        if tip_pos and pip_pos:
             # Use Y coordinate: smaller Y means higher up on the image
             return tip_pos[1] < pip_pos[1]
            # Alternative: Distance based (tip further from wrist than pip)
            # if wrist_pos:
            #    dist_tip_wrist = self._calculate_distance(tip_pos, wrist_pos)
            #    dist_pip_wrist = self._calculate_distance(pip_pos, wrist_pos)
            #    return dist_tip_wrist > dist_pip_wrist * 1.1 # Add tolerance
        return False

    def recognize(self, landmarks):
        """Recognizes the gesture from the current landmarks.

        Args:
            landmarks (list or None): List of landmark dictionaries or None if no hand.

        Returns:
            Gesture: The recognized gesture enum.
        """
        current_time = time.time()
        gesture = Gesture.NONE

        if not landmarks:
            # Reset states if hand is lost
            if self.is_pinching: # End drag if hand lost while pinching
                gesture = Gesture.DRAG_END
            self.is_pinching = False
            self.pinch_start_time = None
            self.is_holding_v = False
            self.v_gesture_start_time = None
            self.is_holding_palm = False
            self.palm_start_time = None
            self.last_wrist_pos = None
            self.last_landmarks = None
            return gesture # Return DRAG_END or NONE

        # --- Get Key Landmark Positions --- (Ensure landmarks is not None here)
        thumb_tip = self._get_landmark_pos(landmarks, self.THUMB_TIP)
        index_tip = self._get_landmark_pos(landmarks, self.INDEX_TIP)
        middle_tip = self._get_landmark_pos(landmarks, self.MIDDLE_TIP)
        ring_tip = self._get_landmark_pos(landmarks, self.RING_TIP)
        pinky_tip = self._get_landmark_pos(landmarks, self.PINKY_TIP)
        wrist_pos = self._get_landmark_pos(landmarks, self.WRIST)

        # --- Basic Finger States --- (Used by multiple gestures)
        index_up = self._is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_up = self._is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_up = self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_up = self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        all_fingers_up = index_up and middle_up and ring_up and pinky_up

        # --- Gesture Recognition Logic --- #

        # 1. Pinch Detection (Thumb Tip 4, Index Tip 8)
        pinch_distance = self._calculate_distance(thumb_tip, index_tip)
        currently_pinching = pinch_distance < self.pinch_threshold

        if currently_pinching and not self.is_pinching:
            # Pinch Start
            self.is_pinching = True
            self.pinch_start_time = current_time
            gesture = Gesture.DRAG_START # Signal potential drag start
            logging.debug("Gesture: Pinch Start (Drag Start)")
        elif not currently_pinching and self.is_pinching:
            # Pinch End
            self.is_pinching = False
            pinch_duration = current_time - (self.pinch_start_time if self.pinch_start_time else current_time)
            self.pinch_start_time = None

            # Decide between Click and Drag End based on duration (optional, simple click for now)
            # We handle double click check first
            is_double_click = False
            if self.last_click_time and (current_time - self.last_click_time < self.double_click_interval):
                 gesture = Gesture.DOUBLE_CLICK
                 logging.debug("Gesture: Double Click")
                 self.last_click_time = None # Consume the first click for double click
                 is_double_click = True
            else:
                 gesture = Gesture.LEFT_CLICK
                 logging.debug("Gesture: Left Click")
                 self.last_click_time = current_time # Record time for potential double click

            # If it wasn't a double click, it could be the end of a drag
            if not is_double_click:
                 # We already sent DRAG_START, now send DRAG_END if pinch ended
                 gesture = Gesture.DRAG_END
                 logging.debug("Gesture: Pinch End (Drag End)")

        elif self.is_pinching:
             # Still pinching - could be considered PINCHING state if needed downstream
             gesture = Gesture.PINCHING # Internal state or could be used for hold

        # 2. V-Gesture (Index 8 up, Middle 12 up, Ring 16 down, Pinky 20 down)
        # Use PIP joints as reference to avoid curl confusion
        ring_down = not self._is_finger_up(landmarks, self.RING_TIP, self.RING_PIP)
        pinky_down = not self._is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        is_v_gesture = index_up and middle_up and ring_down and pinky_down and not self.is_pinching

        if is_v_gesture and not self.is_holding_v:
            self.is_holding_v = True
            self.v_gesture_start_time = current_time
            logging.debug("Gesture: V Start")
        elif is_v_gesture and self.is_holding_v:
            hold_duration = current_time - (self.v_gesture_start_time if self.v_gesture_start_time else current_time)
            if hold_duration >= self.v_hold_duration:
                gesture = Gesture.RIGHT_CLICK
                logging.debug("Gesture: Right Click (V Held)")
                # Reset start time to prevent continuous firing? Or let go/change gesture?
                # Resetting allows single trigger per hold:
                self.v_gesture_start_time = None # Consume the hold
                self.is_holding_v = False # Require releasing V to trigger again
            else:
                 gesture = Gesture.HOLDING_V # Internal state
        elif not is_v_gesture and self.is_holding_v:
            self.is_holding_v = False
            self.v_gesture_start_time = None
            logging.debug("Gesture: V End")

        # 3. Open Palm (All fingers up) - Static for Alt+Tab, Swipe for Tab Next/Prev
        is_open_palm = all_fingers_up and not self.is_pinching and not is_v_gesture

        if is_open_palm and not self.is_holding_palm:
            # Palm Start
            self.is_holding_palm = True
            self.palm_start_time = current_time
            self.last_wrist_pos = wrist_pos # Record position at start of palm
            logging.debug("Gesture: Palm Start")
        elif is_open_palm and self.is_holding_palm:
            # Still Holding Palm - Check for Swipe or Static Hold
            gesture = Gesture.HOLDING_PALM # Default state while holding
            hold_duration = current_time - (self.palm_start_time if self.palm_start_time else current_time)

            # Swipe Detection
            if wrist_pos and self.last_wrist_pos and (current_time - self.last_swipe_time > self.swipe_debounce):
                dx = wrist_pos[0] - self.last_wrist_pos[0]
                # dy = wrist_pos[1] - self.last_wrist_pos[1] # Could use dy for up/down swipes

                if dx > self.swipe_threshold_x:
                    gesture = Gesture.TAB_NEXT # Swipe Right
                    logging.debug("Gesture: Swipe Right (Tab Next)")
                    self.last_swipe_time = current_time
                    # Reset palm state after swipe?
                    self.is_holding_palm = False
                    self.palm_start_time = None
                    self.last_wrist_pos = None
                elif dx < -self.swipe_threshold_x:
                    gesture = Gesture.TAB_PREV # Swipe Left
                    logging.debug("Gesture: Swipe Left (Tab Prev)")
                    self.last_swipe_time = current_time
                    # Reset palm state after swipe?
                    self.is_holding_palm = False
                    self.palm_start_time = None
                    self.last_wrist_pos = None
                # else: # No significant horizontal movement, could be static hold
                 #   pass

            # Static Hold Detection (only if not swiped)
            if gesture == Gesture.HOLDING_PALM and hold_duration >= self.palm_hold_duration:
                gesture = Gesture.SWITCH_WINDOW # Alt+Tab
                logging.debug("Gesture: Static Palm Held (Switch Window)")
                # Reset state to prevent continuous firing
                self.is_holding_palm = False
                self.palm_start_time = None
                self.last_wrist_pos = None

            # Update last position for next frame's swipe check if still holding palm
            if self.is_holding_palm:
                 self.last_wrist_pos = wrist_pos

        elif not is_open_palm and self.is_holding_palm:
            # Palm End
            self.is_holding_palm = False
            self.palm_start_time = None
            self.last_wrist_pos = None
            logging.debug("Gesture: Palm End")

        # --- Default/Fallback Gesture --- #
        # If no other gesture detected but hand is present
        if gesture == Gesture.NONE and landmarks:
             gesture = Gesture.MOVE # Indicate hand is present and can move cursor

        # --- Update State --- #
        self.last_gesture = gesture
        self.last_landmarks = landmarks # Store for next frame

        return gesture

# Example usage (for testing this module directly)
if __name__ == '__main__':
    # This test requires simulated landmark data or integration with HandTracker
    print("Testing GestureRecognizer (requires simulated data)...")

    recognizer = GestureRecognizer(pinch_threshold=0.1, v_hold_duration=0.5, palm_hold_duration=1.0)

    # --- Test Case 1: Simulate Pinch Click --- #
    print("\nTest 1: Pinch Click")
    # Far apart
    landmarks_far = [{ 'id': 4, 'x': 0.4, 'y': 0.5, 'z': 0 }, { 'id': 8, 'x': 0.6, 'y': 0.5, 'z': 0 }]
    # Close together (pinch)
    landmarks_pinch = [{ 'id': 4, 'x': 0.5, 'y': 0.5, 'z': 0 }, { 'id': 8, 'x': 0.52, 'y': 0.5, 'z': 0 }]

    print(f"Step 1 (No Pinch): {recognizer.recognize(landmarks_far)}") # MOVE
    print(f"Step 2 (Pinch Start): {recognizer.recognize(landmarks_pinch)}") # DRAG_START
    print(f"Step 3 (Still Pinching): {recognizer.recognize(landmarks_pinch)}") # PINCHING
    time.sleep(0.1)
    print(f"Step 4 (Pinch End): {recognizer.recognize(landmarks_far)}") # DRAG_END / LEFT_CLICK

    # --- Test Case 2: Simulate Double Click --- #
    print("\nTest 2: Double Click")
    print(f"Step 1 (No Pinch): {recognizer.recognize(landmarks_far)}") # MOVE
    print(f"Step 2 (Pinch 1 Start): {recognizer.recognize(landmarks_pinch)}") # DRAG_START
    print(f"Step 3 (Pinch 1 End): {recognizer.recognize(landmarks_far)}") # DRAG_END/LEFT_CLICK
    time.sleep(0.1) # Within double click interval
    print(f"Step 4 (Pinch 2 Start): {recognizer.recognize(landmarks_pinch)}") # DRAG_START
    print(f"Step 5 (Pinch 2 End): {recognizer.recognize(landmarks_far)}") # DOUBLE_CLICK

    # --- Test Case 3: Simulate V Hold (Right Click) --- #
    # Need more landmarks for V-gesture simulation
    print("\nTest 3: V-Gesture Right Click (Requires more landmarks - Skipped in simple test)")
    # landmarks_v = [...] # Define landmarks for V
    # print(f"Step 1 (V Start): {recognizer.recognize(landmarks_v)}")
    # time.sleep(recognizer.v_hold_duration + 0.1)
    # print(f"Step 2 (V Held): {recognizer.recognize(landmarks_v)}") # RIGHT_CLICK

    # --- Test Case 4: Simulate Palm Hold (Alt+Tab) --- #
    print("\nTest 4: Palm Hold (Requires more landmarks - Skipped in simple test)")
    # landmarks_palm = [...] # Define landmarks for open palm
    # print(f"Step 1 (Palm Start): {recognizer.recognize(landmarks_palm)}")
    # time.sleep(recognizer.palm_hold_duration + 0.1)
    # print(f"Step 2 (Palm Held): {recognizer.recognize(landmarks_palm)}") # SWITCH_WINDOW

    print("\nGestureRecognizer test finished (basic cases).") 