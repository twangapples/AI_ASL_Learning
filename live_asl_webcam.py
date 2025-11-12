import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "sign_language_model.keras"

CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]


def load_sid220():
    model = load_model(MODEL_PATH)
    print("✅ SID220 model loaded for live inference.")
    return model


def extract_landmarks(results: mp.solutions.hands.Hands, image_rgb):
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    return np.expand_dims(np.asarray(coords, dtype=np.float32), axis=0), hand_landmarks


def main():
    model = load_sid220()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam (index 0).")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame from webcam.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                prediction_text = "No hand detected"

                if results.multi_hand_landmarks:
                    landmark_tensor, hand_landmarks = extract_landmarks(results, frame_rgb)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

                    if landmark_tensor is not None:
                        preds = model.predict(landmark_tensor, verbose=0)[0]
                        top_idx = int(np.argmax(preds))
                        confidence = float(np.max(preds))
                        prediction_text = f"{CLASS_LABELS[top_idx]} ({confidence:.0%})"

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

                cv2.imshow("ASL Live Prediction (press 'q' to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

