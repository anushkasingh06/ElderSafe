import cv2
import numpy as np
import tensorflow as tf
import requests
import base64
import time

interpreter = tf.lite.Interpreter(model_path="C:\\Users\\cool\\OneDrive\\Desktop\\thunder.tflite")
interpreter.allocate_tensors()

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

username = "adi12345"
password = "32479DA3-FA60-6329-86C5-4D6EAF89A0D0"
base_url = "https://rest.clicksend.com/v3"
sender_id = "adi12345"
auth_header = {
    "Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
}

def send_sms(message, recipient):
    payload = {
        "messages": [
            {
                "source": "sdk",
                "from": sender_id,
                "body": message,
                "to": recipient
            }
        ]
    }
    endpoint = "/sms/send"
    response = requests.post(base_url + endpoint, headers=auth_header, json=payload)
    if response.status_code == 200:
        print(f"SMS sent successfully to {recipient}.")
    else:
        print(f"Failed to send SMS to {recipient}. Error:", response.text)

def detect_pose(keypoints):
    for person_keypoints in keypoints:
        nose_keypoint = person_keypoints[0]
        nose_confidence = nose_keypoint[2]
        if nose_confidence > 0.4:
            if person_keypoints[15][2] > 0.4 and person_keypoints[16][2] > 0.4:
                if person_keypoints[5][2] < 0.1 and person_keypoints[6][2] < 0.1 and person_keypoints[11][2] < 0.1 and person_keypoints[12][2] < 0.1:
                    return "Running", np.mean([person_keypoints[i][2] for i in range(17)])
            if (person_keypoints[5][2] > 0.4 and person_keypoints[6][2] > 0.4) and (person_keypoints[11][2] < 0.1 and person_keypoints[12][2] < 0.1):
                if person_keypoints[15][2] < 0.1 and person_keypoints[16][2] < 0.1:
                    return "Falling", np.mean([person_keypoints[i][2] for i in range(17)])
            return "Standing", np.mean([person_keypoints[i][2] for i in range(17)])
    return "Unknown", 0.0

sms_sent = False
fps_start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(img, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    pose, accuracy = detect_pose(keypoints_with_scores[0])
    cv2.putText(frame, f"Pose: {pose}, Accuracy: {accuracy:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if pose == "Falling" and not sms_sent:
        emergency_number = "+919301799607"
        message = "Emergency: Person detected falling!"
        send_sms(message, emergency_number)
        sms_sent = True

    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - fps_start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_count = 0
        fps_start_time = time.time()

    cv2.imshow('MoveNet Thunder', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
