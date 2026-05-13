import cv2
import mediapipe as mp
import pandas as pd
import math

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = dot_product / (mag_ba * mag_bc)
    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))


def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    rows = []
    frame_num = 0
    detected_frames = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            row = {"frame": frame_num, "detected": 0}

            if results.pose_landmarks:
                detected_frames += 1
                row["detected"] = 1

                landmarks = results.pose_landmarks.landmark

                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                rs = (right_shoulder.x * width, right_shoulder.y * height)
                re = (right_elbow.x * width, right_elbow.y * height)
                rw = (right_wrist.x * width, right_wrist.y * height)

                ls = (left_shoulder.x * width, left_shoulder.y * height)
                le = (left_elbow.x * width, left_elbow.y * height)
                lw = (left_wrist.x * width, left_wrist.y * height)

                row.update({
                    "right_shoulder_x": rs[0],
                    "right_shoulder_y": rs[1],
                    "right_elbow_x": re[0],
                    "right_elbow_y": re[1],
                    "right_wrist_x": rw[0],
                    "right_wrist_y": rw[1],
                    "right_elbow_angle": calculate_angle(rs, re, rw),

                    "left_shoulder_x": ls[0],
                    "left_shoulder_y": ls[1],
                    "left_elbow_x": le[0],
                    "left_elbow_y": le[1],
                    "left_wrist_x": lw[0],
                    "left_wrist_y": lw[1],
                    "left_elbow_angle": calculate_angle(ls, le, lw),
                })

            rows.append(row)

            if frame_num % 30 == 0:
                print(f"{video_path}: processed frame {frame_num}, detected so far: {detected_frames}")

            frame_num += 1

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"Saved: {output_csv}")
    print(f"Total frames: {frame_num}")
    print(f"Detected frames: {detected_frames}")


if __name__ == "__main__":
    process_video("call_2-6/fulltest1/output3.mp4", "correct_pose.csv")
    process_video("call_2-6/fulltest1/incorrect_form.mp4", "incorrect_pose.csv")