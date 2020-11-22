import cv2
import numpy as np
import matplotlib.pyplot as plt

from yolo_v3 import detect_cars

import time


video = cv2.VideoCapture('/media/bonilla/HDD_2TB_basura/databases/road_traffic_test.mp4')
video_frame_duration = 1 / video.get(cv2.CAP_PROP_FPS)

previous_gray_frame = None
hsv = None

classes = [2, 7]
TOO_FAR_Y = None

pt_1 = None
pt_2 = None
M = None

while True:
    t_start_frame = time.time()
    ret, frame = video.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, None, fx=0.7, fy=0.7)
    if previous_gray_frame is None:
        TOO_FAR_Y = int(frame_resized.shape[0] * 0.45)
        pt_1 = np.float32([[304, 227], [592, 227],
                           [frame_resized.shape[1], 372], [0, 372]])
        pt_2 = np.float32([[0, 0], [frame_resized.shape[1], 0],
                           [frame_resized.shape[1], TOO_FAR_Y], [0, TOO_FAR_Y]])
        M = cv2.getPerspectiveTransform(pt_1, pt_2)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        previous_gray_frame = frame_gray
        continue

    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous_gray_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    warped = cv2.warpPerspective(frame_resized, M, (frame_resized.shape[1], frame_resized.shape[0]))

    for x1, y1, x2, y2, cls in detect_cars(frame_resized):
        if y1 < TOO_FAR_Y or x2 < frame_resized.shape[1] / 2:
            break
        if cls not in classes:
            break
        avg_mag = np.mean(mag[y1: y2, x1: x2])
        avg_ang = np.mean(ang[y1: y2, x1: x2])

        if np.isnan(avg_mag) or np.isnan(avg_ang):
            break

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        delta_x = int(center_x + np.cos(avg_ang) * avg_mag * 15)
        delta_y = int(center_y + np.sin(avg_ang) * avg_mag * 15)

        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.arrowedLine(frame_resized, (center_x, center_y), (delta_x, delta_y), (0, 255, 0), 1)

        speed = np.cos(avg_ang) * avg_mag * 2000000 / (y1 ** 2)
        cv2.putText(frame_resized, str(int(speed)) + "Km/h", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # warped points
        a_center_x2, a_center_y2, a = np.matmul(M, [center_x, center_y, 1])
        cv2.circle(warped, (int(a_center_x2 / a), int(a_center_y2 / a)), 10, (0, 255, 0), -1)

    cv2.line(frame_resized, (0, TOO_FAR_Y), (frame_resized.shape[1], TOO_FAR_Y), (255, 0, 170), 1)
    cv2.imshow('Result', frame_resized)
    cv2.imshow('Warped', warped)

    t_end_frame = time.time()
    frame_duration = t_end_frame - t_start_frame
    delay_need = int(video_frame_duration - frame_duration) * 1000
    if delay_need <= 0:
        delay_need = 1
    if cv2.waitKey(delay_need) == ord('q'):
        break
    previous_gray_frame = frame_gray
    print("FPS:", 1 / (time.time() - t_start_frame))

cv2.destroyAllWindows()
video.release()
