import cv2
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights, EfficientNet
import torchvision
import re

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # For Macbook, use mps


def draw_valence_bar(frame, valence, x, y, w, bar_height=20):
    if valence >= 0.25:
        color = (0, 255, 0)  # Green for positive valence
    elif valence <= -0.25:
        color = (0, 0, 255)  # Red for negative valence
    else:
        color = (255, 255, 0)  # Blue for neutral valence

    bar_width = int(w)
    cv2.rectangle(frame, (x, y - bar_height), (x + bar_width, y), (100, 100, 100), -1)
    cursor_x = x + int(bar_width * (valence + 1) / 2)
    cv2.line(frame, (cursor_x, y - bar_height), (cursor_x, y), color, 2)

    valence_text = f"Valence: {valence:.2f}"
    cv2.putText(
        frame,
        valence_text,
        (x, y - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_arousal_bar(frame, arousal, x, y, h, bar_width=20):
    if arousal >= 0.25:
        color = (0, 255, 0)  # Green for positive valence
    elif arousal <= -0.25:
        color = (0, 0, 255)  # Red for negative valence
    else:
        color = (255, 255, 0)  # Blue for neutral valence

    bar_height = int(h)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), -1)
    cursor_y = y + int(bar_height * (-arousal + 1) / 2)
    cv2.line(frame, (x, cursor_y), (x + bar_width, cursor_y), color, 2)

    arousal_text = f"Arousal: {arousal:.2f}"
    cv2.putText(
        frame,
        arousal_text,
        (x + 30, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def get_emotion(outputs_cls):
    emotions = [
        "Neutral",
        "Happy",
        "Sad",
        "Suprise",
        "Fear",
        "Disgust",
        "Angry",
        "Contempt",  # AffectNet8 has 8 classes, when using the AffectNet7 model, remove this class
    ]

    max_indices = outputs_cls.argmax(dim=1)
    emotions_batch = [emotions[idx.item()] for idx in max_indices]
    return emotions_batch

    valence_text = f"Valence: {valence:.2f}"
    cv2.putText(
        frame,
        valence_text,
        (x, y - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


cap = cv2.VideoCapture(0)  # 0 is usually the default camera (webcam)

# Load the model
MODEL = models.maxvit_t(weights="DEFAULT")
block_channels = MODEL.classifier[3].in_features
MODEL.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.LayerNorm(block_channels),
    nn.Linear(block_channels, block_channels),
    nn.Tanh(),
    nn.Linear(
        block_channels, 10, bias=False
    ),  # Change the number of output classes, e.g. for AffectNet7 combined use 9 output neurons
)
MODEL.load_state_dict(
    torch.load(
        "models/AffectNet8_Maxvit_Combined/model.pt", map_location=torch.device(DEVICE)
    )
)
MODEL.eval()
MODEL.to(DEVICE)

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# Inititalize the face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ***** Access the webcam *****

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        text = "Press 'q' to quit"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        faces = face_classifier.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        # Loop over multiple detected faces
        for x, y, w, h in faces:
            # Cut out the face from the frame
            face_roi = frame[y : y + h, x : x + w]

            img = test_transform(face_roi)
            img = img.unsqueeze(0)  # Add a batch dimension for the model
            outputs = MODEL(img.to(DEVICE))
            outputs_cls = outputs[:, :7]
            valence = outputs[:, 7:8].item()
            arousal = outputs[:, 8:].item()

            # Draw the valence bar over the face
            draw_valence_bar(frame, valence, x, y, w)
            draw_arousal_bar(frame, arousal, x + w, y, h)

            emotion = get_emotion(outputs_cls)
            emotion_text = f"Emotion: {emotion}"
            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
                0
            ]
            cv2.putText(
                frame,
                emotion_text,
                (x - text_size[0] - 10, y + text_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
