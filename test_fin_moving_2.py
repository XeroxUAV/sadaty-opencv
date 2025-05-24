import numpy as np
import cv2

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
        self.controller = None
        # ✅ Two predefined HSV ranges
        self.lower1 = np.array([110, 55, 82])
        self.upper1 = np.array([179, 130, 146])

        self.lower2 = np.array([110, 69, 102])
        self.upper2 = np.array([179, 255, 141])

        self.lower3 = np.array([108, 45, 90])
        self.upper3 = np.array([179, 255, 255])

        self.lower4 = np.array([106, 49, 81])
        self.upper4 = np.array([179, 255, 198])

        self.lower5 = np.array([105, 42, 79])
        self.upper5 = np.array([179, 255, 255])

        self.lower6 = np.array([104, 42, 73])
        self.upper6 = np.array([179, 255, 176])

        self.lower14 = np.array([110, 47, 88])
        self.upper14 = np.array([179, 255, 255])

        self.lower151 = np.array([113, 64, 77])
        self.upper151 = np.array([179, 255, 255])

        self.lower16 = np.array([116, 64, 77])
        self.upper16 = np.array([179, 255, 255])

    def get_frame_shape(self, frame):
        self.rows, self.cols = frame.shape[:2]

    def detect_window_center(self, frame):
        imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ✅ Create combined mask using 2 HSV ranges
        if self.controller.mask == 0:
            # mask1 = cv2.inRange(imgHSV, self.lower1, self.upper1)
            # mask2 = cv2.inRange(imgHSV, self.lower2, self.upper2)
            # mask3 = cv2.inRange(imgHSV, self.lower3, self.upper3)
            # mask = cv2.bitwise_and(mask1, mask2)
            # mask = cv2.bitwise_and(mask3, mask)
            mask1 = cv2.inRange(imgHSV, self.lower14, self.upper14)
            mask2 = cv2.inRange(imgHSV, self.lower151, self.upper151)
            mask3 = cv2.inRange(imgHSV, self.lower16, self.upper16)
            mask = cv2.bitwise_and(mask1, mask2)
            mask = cv2.bitwise_and(mask3, mask)

        elif self.controller.mask == 1:
            mask4 = cv2.inRange(imgHSV, self.lower4, self.upper4)
            mask5 = cv2.inRange(imgHSV, self.lower5, self.upper5)
            mask6 = cv2.inRange(imgHSV, self.lower6, self.upper6)
            mask = cv2.bitwise_and(mask4, mask5)

        # mask = cv2.bitwise_and(mask, mask6)

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
            if len(approx) == 4:
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
