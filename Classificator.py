# KIT

import os
import cv2

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

# from fer import Video
from fer import FER  # noqa: E402


def classificator():
    # Initialize the FER detector
    emo_detector = FER(mtcnn=True)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera (webcam)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()

            # Add text to the frame
            text = "Press 'q' to quit"
            cv2.putText(
                frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Call the FER top_emotion function on the current frame
            # dominant_emotion, emotion_score = emo_detector.top_emotion(frame)

            # Detect faces in the frame
            faces = emo_detector.detect_emotions(frame)

            for face in faces:
                x, y, w, h = face["box"]  # Get the face bounding box coordinates
                dominant_emotion, emotion_score = sorted(
                    face["emotions"].items(), key=lambda t: t[1], reverse=True
                )[0]

                # Draw a rectangle around the detected face
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (255, 0, 0), 2
                )  # Red color

                # Display the dominant emotion and its score near the face
                emotion_text = f"{dominant_emotion} (Score: {emotion_score:.2f})"
                cv2.putText(
                    frame,
                    emotion_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            # Display the frame
            cv2.imshow("Webcam", frame)

            # Break the loop when the 'q' key is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    classificator()
