from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
from gesture_recognize import GestureRecognition
import cv2
import numpy as np
import base64
import time

app = Flask(__name__)

# Initialize gesture recognition
model_path = '/Users/evancureton/Desktop/gesture_recognition/gesture_recognizer.task'
gesture_recognition = GestureRecognition(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    # Decode base64 image
    img_bytes = base64.b64decode(image_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Process frame through gesture recognition
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    gesture_recognition.timestamp += 1
    gesture_recognition.recognizer.recognize_async(mp_frame, gesture_recognition.timestamp)
    
    # Wait briefly for the async recognition to complete
    time.sleep(0.01)
    
    if gesture_recognition.current_frame is not None:
        frame = gesture_recognition.current_frame
    
    # Encode the processed frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_image = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{processed_image}',
        'gesture': gesture_recognition.current_gesture,
        'distance': f'{gesture_recognition.current_distance:.2f}',
        'timestamp': gesture_recognition.timestamp
    })

@app.route('/set_volume', methods=['POST'])
def set_volume():
    data = request.get_json()
    volume = data.get('volume', 0.5)
    gesture_recognition.set_volume(volume)
    return jsonify({'volume': gesture_recognition.volume})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        ret, frame = gesture_recognition.cam.read()
        if not ret:
            break
        else:
            # Process frame through gesture recognition
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            gesture_recognition.recognizer.recognize_async(mp_frame, gesture_recognition.timestamp)
            
            if gesture_recognition.current_frame is not None:
                frame = gesture_recognition.current_frame
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)