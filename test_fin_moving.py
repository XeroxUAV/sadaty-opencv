import numpy as np
import cv2
import time

class WindowPassing:
    def __init__(self, kp=0, kd=0, ki=0, cent_threshold=20, tilt_threshold=15):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.cent_threshold = cent_threshold
        self.tilt_threshold = tilt_threshold
        self.center_coords = None
        self.cols = 0
        self.rows = 0

    def get_frame_shape(self, frame):
        self.rows, self.cols = frame.shape[:2]

    def get_hsv_from_trackbar(self):
        h_min = cv2.getTrackbarPos("H Min", "HSV Controls")
        s_min = cv2.getTrackbarPos("S Min", "HSV Controls")
        v_min = cv2.getTrackbarPos("V Min", "HSV Controls")
        h_max = cv2.getTrackbarPos("H Max", "HSV Controls")
        s_max = cv2.getTrackbarPos("S Max", "HSV Controls")
        v_max = cv2.getTrackbarPos("V Max", "HSV Controls")
        return (np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

    def detect_window_center(self, frame):
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get HSV from trackbars
        lower, upper = self.get_hsv_from_trackbar()

        # Create mask
        mask = cv2.inRange(imgHSV, lower, upper)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        imgResult = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 50)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if 4 == len(approx):
                pts = approx.reshape(-1, 2)
                win_cx = int((pts[:, 0].min() + pts[:, 0].max()) / 2)
                win_cy = int((pts[:, 1].min() + pts[:, 1].max()) / 2)
                cv2.circle(frame, (win_cx, win_cy), 20, (0, 255, 0), 3)
                cv2.putText(frame, f"Center", (win_cx + 10, win_cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.center_coords = (win_cx, win_cy)
                return (win_cx, win_cy), area, mask, edges, imgResult

        self.center_coords = None
        return None, 0, mask, edges, imgResult

    def process_frame(self, frame):
        self.get_frame_shape(frame)
        center, area, mask, edges, result = self.detect_window_center(frame)
        return center, area, mask, edges, result
