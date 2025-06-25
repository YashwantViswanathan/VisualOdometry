import torch
import cv2
import numpy as np
import os

# Load MiDaS v2.1 Small
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# MiDaS transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load YOLOv5
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo.eval()

# Create output directory
output_dir = 'distance_angle_frames_2'
os.makedirs(output_dir, exist_ok=True)

# Open video
video_path = '/Users/yashwant/Documents/DashVid2.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
scale = None  # Calibration scale
known_distance = 10.0  # Known distance for calibration in meters

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Approximate focal length in pixels (based on 24mm focal length on 36mm sensor)
    fx = w / (2 * np.tan(np.radians(24 / 2)))
    cx = w / 2

    # MiDaS depth estimation
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()

    # YOLO detection
    results = yolo(img_rgb)
    detections = results.xyxy[0].cpu().numpy()
    vehicle_classes = [1, 2, 3, 5, 7]
    vehicle_detections = [d for d in detections if int(d[5]) in vehicle_classes]

    # Calibrate using first vehicle in first frame
    if frame_count == 0 and vehicle_detections:
        x1, y1, x2, y2, _, _ = vehicle_detections[0]
        xm, ym = int((x1 + x2) / 2), int(y2)
        scale = known_distance * prediction[ym, xm]

    # Process detections
    for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(vehicle_detections):
        xm, ym = int((x1 + x2) / 2), int(y2)
        if 0 <= xm < w and 0 <= ym < h:
            depth_val = prediction[ym, xm]
            distance = scale / depth_val if scale else 0.0

            # Angle of incidence calculation
            angle_rad = np.arctan((xm - cx) / fx)
            angle_deg = np.degrees(angle_rad)

            # Annotate
            label = f"{results.names[int(cls_id)]}: {distance:.1f}m, {angle_deg:.1f}°"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Save annotated frame
    out_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"Saved: {out_path}")

    frame_count += 1

cap.release()
print("Finished processing with distance and angle annotations.")

