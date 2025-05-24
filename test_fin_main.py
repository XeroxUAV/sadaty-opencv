import cv2
import time
from djitellopy import Tello
from test_fin_moving import WindowPassing


class TelloWrapper:
    def __init__(self):
        self.tello = Tello()
        self.tello.connect()
        print(f"Battery: {self.tello.get_battery()}%")
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()

    def get_frame(self):
        return self.frame_read.frame

    def stop(self):
        self.tello.streamoff()
        self.tello.end()


class Main:
    def __init__(self):
        self.drone = TelloWrapper()
        self.WindowPassing = WindowPassing()
        self.last_time = time.time()

    def run(self):
        try:
            while True:
                frame = self.drone.get_frame()
                if frame is None:
                    continue

                frame = cv2.resize(frame, (640, 480))

                now = time.time()
                fps = 1 / (now - self.last_time)
                self.last_time = now
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Get mask too
                _, _, mask, _, _ = self.WindowPassing.process_frame(frame)

                #  Show both original and mask
                cv2.imshow("Tello Camera - Window Detection", frame)
                cv2.imshow("Window Detection Mask", mask)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

                time.sleep(0.03)
        finally:
            print("Closing connection...")
            self.drone.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Main()
    app.run()
