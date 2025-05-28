import cv2
import numpy as np
from djitellopy import Tello

# Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()
print(tello.get_battery())
#########
kernel = np.ones((3, 3), np.uint8)



# Initialize Kalman filter
class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self):
        prediction = self.kalman.predict()
        return prediction

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)

kf = KalmanFilter()
alpha = 0.2
smoothed_center = None
last_confidence = 1.0
decay_factor = 0.95

def nothing(x):
    pass

# First HSV window
cv2.namedWindow("HSV1")
cv2.resizeWindow("HSV1", 300, 300)
cv2.createTrackbar("LH", "HSV1", 114, 179, nothing)
cv2.createTrackbar("LS", "HSV1", 89, 255, nothing)
cv2.createTrackbar("LV", "HSV1", 177, 255, nothing)
cv2.createTrackbar("UH", "HSV1", 179, 179, nothing)
cv2.createTrackbar("US", "HSV1", 255, 255, nothing)
cv2.createTrackbar("UV", "HSV1", 255, 255, nothing)

# Second HSV window
cv2.namedWindow("HSV2")
cv2.resizeWindow("HSV2", 300, 300)
cv2.createTrackbar("LH", "HSV2", 116, 179, nothing)
cv2.createTrackbar("LS", "HSV2", 84, 255, nothing)
cv2.createTrackbar("LV", "HSV2", 75, 255, nothing)
cv2.createTrackbar("UH", "HSV2", 128, 179, nothing)
cv2.createTrackbar("US", "HSV2", 146, 255, nothing)
cv2.createTrackbar("UV", "HSV2", 255, 255, nothing)

# Third HSV window
cv2.namedWindow("HSV3")
cv2.resizeWindow("HSV3", 300, 300)
cv2.createTrackbar("LH", "HSV3", 116, 179, nothing)
cv2.createTrackbar("LS", "HSV3", 36, 255, nothing)
cv2.createTrackbar("LV", "HSV3", 153, 255, nothing)
cv2.createTrackbar("UH", "HSV3", 145, 179, nothing)
cv2.createTrackbar("US", "HSV3", 128, 255, nothing)
cv2.createTrackbar("UV", "HSV3", 255, 255, nothing)

cv2.namedWindow("MinArea")
cv2.resizeWindow("MinArea", 300, 50)
cv2.createTrackbar("MinArea", "MinArea", 10000, 50000, nothing)

def process_frame(frame):
    global smoothed_center, last_confidence
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # First HSV range
    lh1 = cv2.getTrackbarPos("LH", "HSV1")
    ls1 = cv2.getTrackbarPos("LS", "HSV1")
    lv1 = cv2.getTrackbarPos("LV", "HSV1")
    uh1 = cv2.getTrackbarPos("UH", "HSV1")
    us1 = cv2.getTrackbarPos("US", "HSV1")
    uv1 = cv2.getTrackbarPos("UV", "HSV1")
    lower1 = np.array([lh1, ls1, lv1])
    upper1 = np.array([uh1, us1, uv1])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Second HSV range
    lh2 = cv2.getTrackbarPos("LH", "HSV2")
    ls2 = cv2.getTrackbarPos("LS", "HSV2")
    lv2 = cv2.getTrackbarPos("LV", "HSV2")
    uh2 = cv2.getTrackbarPos("UH", "HSV2")
    us2 = cv2.getTrackbarPos("US", "HSV2")
    uv2 = cv2.getTrackbarPos("UV", "HSV2")
    lower2 = np.array([lh2, ls2, lv2])
    upper2 = np.array([uh2, us2, uv2])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # Third HSV range
    lh3 = cv2.getTrackbarPos("LH", "HSV3")
    ls3 = cv2.getTrackbarPos("LS", "HSV3")
    lv3 = cv2.getTrackbarPos("LV", "HSV3")
    uh3 = cv2.getTrackbarPos("UH", "HSV3")
    us3 = cv2.getTrackbarPos("US", "HSV3")
    uv3 = cv2.getTrackbarPos("UV", "HSV3")
    lower3 = np.array([lh3, ls3, lv3])
    upper3 = np.array([uh3, us3, uv3])
    mask3 = cv2.inRange(hsv, lower3, upper3)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask1, mask2)
    combined_mask = cv2.bitwise_or(combined_mask, mask3)

    min_area = cv2.getTrackbarPos("MinArea", "MinArea")
    dialated = cv2.dilate(combined_mask,kernel , iterations=2)
    # opening = cv2.morphologyEx(dialated, cv2.MORPH_OPEN,
    #                            kernel, iterations=1)
    contours, _ = cv2.findContours(dialated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_window = None
    best_area = 0

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)
        if 4 <= len(approx) <= 6 and area > min_area:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.3 < aspect_ratio < 3.0 and area > best_area:
                best_window = approx
                best_area = area

    if best_window is not None:
        M = cv2.moments(best_window)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if smoothed_center is None:
                smoothed_center = (cX, cY)
            else:
                smoothed_center = (int(alpha * cX + (1 - alpha) * smoothed_center[0]),
                                   int(alpha * cY + (1 - alpha) * smoothed_center[1]))

            kf.correct(smoothed_center[0], smoothed_center[1])
            cv2.drawContours(frame, [best_window], 0, (0, 255, 0), 3)
            cv2.circle(frame, smoothed_center, 5, (255, 255 , 0), -1)
            last_confidence = 1.0
            cv2.putText(frame, f"Center: {smoothed_center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "confidence : {last_confidence}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    else:
        # If no detection, decay confidence and keep last smoothed_center
        last_confidence *= decay_factor
        cv2.putText(frame, f"Confidence: {last_confidence:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    predicted = kf.predict()
    predicted = (int(predicted[0]), int(predicted[1]))
    cv2.circle(frame, predicted, 5, (255, 0, 0), -1)
    cv2.putText(frame, f"Predicted: {predicted}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Mask1", mask1)
    cv2.imshow("Mask2", mask2)
    cv2.imshow("Mask3", mask3)
    cv2.imshow("Combined Mask", combined_mask)
    return frame

# Main loop
while True:
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))

    output = process_frame(frame)
    cv2.imshow("Window Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
tello.streamoff()