from djitellopy import Tello
import cv2
import time
from controller import Controller
from test_fin_moving_2 import WindowPassing

# width = 640  # WIDTH OF THE IMAGE
# height = 480  # HEIGHT OF THE IMAGE
move_power = 50



def save_image_on_click(event, x, y, flags, param):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param["frame"]

        filename = f"./images/screenshot_{int(cv2.getTickCount())}-{counter}.png"
        cv2.imwrite(filename, frame)
        counter += 1
        print(f"[INFO] Image saved as {filename}")


def stream():
    frame_read = tello.get_frame_read()
    frame = frame_read.frame

    if not frame is None:
        # img = cv2.resize(myFrame, (width, height))
            # describe the type of font
        # to be used.
        center, area, mask, edges, result = frame_processor.process_frame(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use putText() method for
        # inserting text on video
        cv2.putText(frame,
                    f'{tello.get_battery()}%',
                    (50, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

        cv2.imshow("Mask", mask)
        # cv2.imshow("Result", result)
        cv2.imshow('Original', frame)
        # cv2.setMouseCallback("Original", save_image_on_click, {"frame": frame})

        # WAIT FOR THE 'Q' BUTTON TO STOP
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True



if __name__ == "__main__":
    controller = Controller()
    tello = Tello()
    counter = 0
    tello.connect()
    frame_processor = WindowPassing()
    frame_processor.controller = controller
    # tello.streamoff()
    # tello.streamon()

    while True:
        # print("Switche States:\n", controller.switches)
        controller.update_states()
        # Turn the video stream on and off
        if controller.switches['stream-on']:
            tello.streamoff()
            tello.streamon()
            time.sleep(0.5)
            controller.allow_stream = True
            controller.switches['stream-on'] = False
            print('Stream is on ...')

        elif controller.switches['stream-off']:
            tello.streamoff()
            controller.allow_stream = False
            controller.switches['stream-off'] = False
            print('Stream is off ...')

        # Take-off and Land of Tello
        if controller.switches['take-off']:
            tello.takeoff()
            controller.allow_movment = True
            controller.switches['take-off'] = False

        elif controller.switches['land']:
            tello.land()
            controller.allow_movment = False
            controller.switches['take-off'] = False

        if controller.allow_stream:
            try:
                stream()
            except Exception as e:
                print("This is a fucking error", e)
                tello.streamoff()
                tello.land()
                break
        # try:
        #     stream()
        # except Exception as e:
        #     print(e)
        #     tello.streamoff()
        #     break
        if controller.allow_movment:
            tello.send_rc_control(
                int(controller.movment['pitch'] * move_power),
                int(controller.movment['roll'] * -move_power),
                int(controller.movment['throttle'] * -move_power),
                int(controller.movment['yaw'] * move_power)
            )
