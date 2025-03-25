import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
import pygame

class GestureRecognition:
    def __init__(self, model_path, video_source=0, output_video='output.mp4'):
        # Initialize MediaPipe and OpenCV
        self.model_path = model_path
        self.cam = cv2.VideoCapture(video_source)
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video, fourcc, 20.0, (self.frame_width, self.frame_height))

        self.current_frame = None
        self.timestamp = 0
        self.current_gesture = "None"
        self.played = False

        self.thumb_tip_local = (0,0)
        self.index_tip_local = (0,0)
        self.middle_tip_local = (0,0)
        self.ring_tip_local = (0,0)
        self.pinky_tip_local = (0,0)
        

        # MediaPipe and Gesture Recognizer setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Gesture Recognizer setup
        self.base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.print_result
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        MARGIN = 20  # increased margin
        FONT_SIZE = 1.5  # increased font size
        FONT_THICKNESS = 2  # increased thickness
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        test_index = 0

        # Loop through the detected hands to visualize
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Ensure the landmarks are correctly accessed for drawing
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

            # Add landmarks to the Proto object
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks   # Ensure correct access to landmarks
            ])
            
            # Draw the hand landmarks
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            

            # Get the top left corner of the detected hand's bounding box
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            z_coordinates = [landmark.z for landmark in hand_landmarks]

            self.thumb_tip_local = (x_coordinates[4], y_coordinates[4])
            self.index_tip_local = (x_coordinates[8], y_coordinates[8])
            self.middle_tip_local = (x_coordinates[12], y_coordinates[12])
            self.ring_tip_local = (x_coordinates[16], y_coordinates[16])
            self.pinky_tip_local = (x_coordinates[20], y_coordinates[20])

            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def apply_wacky_filter(self, image: np.ndarray, magnitude: float) -> np.ndarray:
        

        # Convert the image to HSV to manipulate colors
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Randomly modify the hue, saturation, and value (brightness)
        hue_shift = int (magnitude * random.randint(-30, 30))  # Small random hue shift for color variation
        saturation_shift = int (magnitude * random.randint(-30, 30))  # Random saturation shift
        value_shift = int(magnitude * random.randint(-30, 30))  # Random brightness shift

        hsv_image[..., 0] = cv2.add(hsv_image[..., 0], hue_shift)  # Modify Hue
        hsv_image[..., 1] = cv2.add(hsv_image[..., 1], saturation_shift)  # Modify Saturation
        hsv_image[..., 2] = cv2.add(hsv_image[..., 2], value_shift)  # Modify Brightness

        # Convert back to BGR format
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Apply some distortion - create a wave-like effect
        height, width, _ = image.shape
        for y in range(height):
            # Create a sinusoidal wave pattern to distort the image
            distortion = int(magnitude * int(30 * np.sin(y / 10.0 + random.random())))
            image[y, :] = np.roll(image[y, :], distortion, axis=0)

        # Add some noise to the image
        #noise = int(magnitude * np.random.randint(0, 50, (height, width, 3), dtype=np.uint8))
        #image = cv2.add(image, noise)

        # Optional: Add a bit of Gaussian blur for extra wackiness
        #image = cv2.GaussianBlur(image, (5, 5), 0)

        return image

    def distance_between_fingers(self, finger_tip_1, finger_tip_2):
        x1, y1 = finger_tip_1
        x2, y2 = finger_tip_2

        # Compute the Euclidean distance between the two landmarks
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        

        return distance

    def print_result(self, result, output_image, timestamps_ms):
        gesture_label = "No gesture detected"
        distance = 0.0
        multi = 0

        # Check if any gestures are detected
        if len(result.gestures) > 0:
            gesture_label = result.gestures[0][0].category_name
            distance = self.distance_between_fingers(self.index_tip_local, self.thumb_tip_local)
            pygame.mixer.init()
            audio_file_path = "/Users/evancureton/Desktop/gesture_recognition/best_sound.mp3"
            if gesture_label == "Closed_Fist":
                multi = 100
                if not self.played:
                    pygame.mixer.music.load(audio_file_path)
                    pygame.mixer.music.play()
                    self.played = True

            elif gesture_label == "Pointing_Up":
                multi = 0
                self.played = False
                pygame.mixer.stop()
            else:
                multi = 10
                pygame.mixer.stop()

        # Convert the image from mp.Image to OpenCV format for further processing
        frame = output_image.numpy_view()
        frame = self.apply_wacky_filter(frame, distance * multi)

        # Draw landmarks for all detected hands
        annotated_frame = self.draw_landmarks_on_image(frame, result)

        # Update the current frame and gesture info
        self.current_frame = annotated_frame
        self.current_gesture = gesture_label
        self.current_distance = distance  # Add this line to store the distance

    def start_recognition(self, mode = 'o', stop_gesture = "", duration_sec = float("inf")):
        timestamp = 0
        total_time = 0

        while True:
            ret, frame = self.cam.read()
            start_time = time.time()

            if not ret:
                break

            

            # Write the frame to the output file
            # self.out.write(frame)

            # Convert frame to MediaPipe format
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.recognizer.recognize_async(mp_frame, timestamp)

            # Display the annotated frame
            if self.current_frame is not None:
                cv2.imshow('Gesture Recognition', self.current_frame)

            timestamp += 1
            total_time += time.time() - start_time

            # Exit the loop when 'q' is pressed
            if mode == 'o': 
                if cv2.waitKey(1) == ord('q'):
                    break
            elif mode == 't':
                if total_time >= duration_sec:
                    break
            elif mode == 'g':
                if self.current_gesture == stop_gesture:
                    break
            else:
                raise Exception("Uknown Mode")

            

        # Release resources
        self.cam.release()
        self.out.release()
        cv2.destroyAllWindows()
    
    

        

# Usage
if __name__ == "__main__":
    model_path = '/Users/evancureton/Desktop/gesture_recognition/gesture_recognizer.task'  # Path to your gesture recognizer model
    gesture_recognition = GestureRecognition(model_path)
    gesture_recognition.start_recognition()

    
    
    # # Start the first recognition (open mode)
    # thread1 = threading.Thread(target=gesture_recognition.start_recognition, args=('o',))
    # thread1.start()

    # # Start the second recognition (timed mode)
    # thread2 = threading.Thread(target=gesture_recognition.start_recognition, args=('t', 5.0))
    # thread2.start()

    # # Wait for threads to complete
    # thread1.join()
    # thread2.join()

