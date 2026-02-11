import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def angle_with_vertical(p1, p2):
    ''' calculem l'angle entre 2 keypoints del cos '''
    vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    vertical = np.array([0, 1])
    cos_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
    angle = np.degrees(np.arccos(cos_angle))
    return angle

cap = cv2.VideoCapture(0)  # 0 = webcam

bye = False
while not bye:
    ret, frame = cap.read()
    if not ret:
        bye = True

    results = model(frame, verbose=False)[0]
    annotated_frame = results.plot()

    if results.keypoints is not None:
        kpts = results.keypoints.xy[0].cpu().numpy()

        left_shoulder = kpts[5]
        right_shoulder = kpts[6]
        left_eye = kpts[1]
        right_eye = kpts[2]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        eye_center = (kpts[1] + kpts[2]) / 2

        neck_angle = angle_with_vertical(shoulder_center, eye_center)
        head_forward = eye_center[0] - shoulder_center[0]

        posture_text = "GOOD POSTURE"
        color = (0, 255, 0)
        
        if neck_angle > 20 or head_forward > 30:
            posture_text = "BAD POSTURE"
            color = (0, 0, 255)

        cv2.putText(annotated_frame, posture_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(annotated_frame, f"Neck angle: {int(neck_angle)}",
            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(annotated_frame, f"Head forward: {int(head_forward)}",
            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    cv2.imshow("Posture Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        bye = True

cap.release()
cv2.destroyAllWindows()
