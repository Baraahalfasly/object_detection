import cv2
import os
import subprocess
from datetime import datetime

classnames = []
classfile = 'files/coco.names'
with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')

p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
net = cv2.dnn_DetectionModel(p, v)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

KNOWN_WIDTHS = {
    "person": 50.0,
    "car": 180.0,
    "bicycle": 60.0,
}
KNOWN_DISTANCE = 200.0
focal_length = None

def calculate_focal_length(measured_width_in_image, known_width):
    return (measured_width_in_image * KNOWN_DISTANCE) / known_width

def calculate_distance(focal_length, object_width_in_image, known_width):
    return (known_width * focal_length) / object_width_in_image

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
video_filename = os.path.join(output_folder, f'output_video_{timestamp}.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

print(f"Saving video to: {video_filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")
        break

    classids, confs, bbox = net.detect(frame, confThreshold=0.5)

    if focal_length is None and len(bbox) > 0:
        for classid, _, box in zip(classids.flatten(), confs.flatten(), bbox):
            label = classnames[classid - 1]
            if label in KNOWN_WIDTHS:
                object_width_in_image = box[2]
                focal_length = calculate_focal_length(object_width_in_image, KNOWN_WIDTHS[label])
                print(f"Calculated focal length for '{label}': {focal_length}")
                break

    for classid, confidence, box in zip(classids.flatten(), confs.flatten(), bbox):
        x, y, w, h = box
        label = classnames[classid - 1]

        if label in KNOWN_WIDTHS and focal_length:
            distance = calculate_distance(focal_length, w, KNOWN_WIDTHS[label])
            distance_text = f"{distance:.2f} cm"
        else:
            distance_text = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {distance_text}",
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), thickness = 0)

    cv2.imshow('Object Detection with Distance', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Opening video: {video_filename}")
subprocess.run(['start', video_filename], shell=True)
