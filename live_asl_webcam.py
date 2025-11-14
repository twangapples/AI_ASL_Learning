import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(
    "/Users/thomaswang/Documents/python/American-Sign-Language-Detection/model/keypoint_classifier/keypoint_classifier.tflite"
)
LABEL_PATH = Path(
    "/Users/thomaswang/Documents/python/American-Sign-Language-Detection/model/keypoint_classifier/keypoint_classifier_label.csv"
)


def load_tflite_classifier():
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(LABEL_PATH, encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f)]

    print("✅ Keypoint TFLite classifier loaded for live inference.")
    return interpreter, input_details, output_details, labels


def calc_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([x, y])
    return landmark_point


def pre_process_landmarks(landmark_list):
    if not landmark_list:
        return []

    temp = landmark_list.copy()
    base_x, base_y = temp[0]
    for point in temp:
        point[0] -= base_x
        point[1] -= base_y

    flattened = np.array(temp, dtype=np.float32).flatten()
    max_value = np.max(np.abs(flattened))
    if max_value == 0:
        return flattened.tolist()
    return (flattened / max_value).tolist()


def calc_bounding_rect(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = []
    for lm in hand_landmarks.landmark:
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        landmark_array.append([x, y])
    points = np.array(landmark_array)
    x, y, w, h = cv2.boundingRect(points)
    return [x, y, x + w, y + h]


def draw_bounding_rect(frame, brect):
    x1, y1, x2, y2 = brect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    interpreter, input_details, output_details, labels = load_tflite_classifier()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam (index 0).")

    dwell_frames = 15
    cooldown_seconds = 1.2
    frame_stability = 0
    last_prediction = None
    last_confirmed_letter = None
    cooldown_until = 0.0
    typed_text = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                prediction_text = "No hand detected"

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    processed = pre_process_landmarks(landmark_list)

                    if processed:
                        input_tensor = np.array([processed], dtype=np.float32)
                        interpreter.set_tensor(input_details[0]["index"], input_tensor)
                        interpreter.invoke()
                        preds = interpreter.get_tensor(output_details[0]["index"])[0]
                        top_idx = int(np.argmax(preds))
                        confidence = float(np.max(preds))
                        predicted_letter = labels[top_idx]
                        prediction_text = f"{predicted_letter} ({confidence:.0%})"

                        if predicted_letter == last_prediction:
                            frame_stability += 1
                        else:
                            frame_stability = 1
                            last_prediction = predicted_letter

                        now = time.time()
                        if (
                            frame_stability >= dwell_frames
                            and now >= cooldown_until
                            and (last_confirmed_letter != predicted_letter or now - cooldown_until >= cooldown_seconds)
                        ):
                            typed_text.append(predicted_letter)
                            last_confirmed_letter = predicted_letter
                            cooldown_until = now + cooldown_seconds

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )
                    draw_bounding_rect(frame, calc_bounding_rect(frame, hand_landmarks))

                cv2.putText(
                    frame,
                    prediction_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.rectangle(frame, (15, 70), (625, 125), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    "Text: " + "".join(typed_text[-50:]),
                    (25, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("ASL Live Prediction (press 'q' to quit)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    typed_text.clear()
                if key == ord("b") and typed_text:
                    typed_text.pop()
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

