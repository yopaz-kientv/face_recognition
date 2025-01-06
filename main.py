import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO

# Khởi tạo Flask và SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# HTML để hiển thị camera (client-side)
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý video được truyền từ client (camera điện thoại)
@socketio.on('video_frame')
def handle_video_frame(frame_data):
    # Chuyển đổi dữ liệu frame từ base64 thành ảnh
    img_data = base64.b64decode(frame_data)
    np_array = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Chuyển sang không gian màu RGB để xử lý bởi MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    results = face_detection.process(rgb_frame)

    # Vẽ hình chữ nhật xung quanh khuôn mặt
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Chuyển đổi lại frame thành base64 để gửi về client
    _, buffer = cv2.imencode('.jpg', frame)
    img_byte = BytesIO(buffer)
    img_base64 = base64.b64encode(img_byte.getvalue()).decode('utf-8')

    # Gửi frame đã xử lý về client
    emit('processed_frame', img_base64)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
