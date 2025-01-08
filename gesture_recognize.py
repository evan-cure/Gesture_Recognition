import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

model_path = '/Users/evancureton/Desktop/gesture recognition/gesture_recognizer.task'

#base_options = BaseOptions(model_asset_path = model_path)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamps_ms: int): 
    print('gesture recognition result: {result} at timestamp: {timestamps_ms}')
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, frame = cam.read()
        frame_timestamp = cam.get(cv2.CAP_PROP_POS_MSEC)

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_frame, frame_timestamp)
        print(GestureRecognizerOptions.result_callback)
        

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break


cam.release()
out.release()
cv2.destroyAllWindows()



