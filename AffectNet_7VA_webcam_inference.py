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


def draw_valence_bar(frame, valence, x, y, w, bar_height=20):
    # Define the color based on valence value
    if valence >= 0.25:
        color = (0, 255, 0)  # Green for positive valence
    elif valence <= -0.25:
        color = (0, 0, 255)  # Red for negative valence
    else:
        color = (255, 255, 0)  # Blue for neutral valence

    # Calculate the width of the bar based on the valence
    bar_width = int(w)

    # Draw the base bar
    cv2.rectangle(frame, (x, y - bar_height), (x + bar_width, y), (100, 100, 100), -1)

    # Calculate the cursor position within the bar
    cursor_x = x + int(bar_width * (valence + 1) / 2)

    # Draw the cursor
    cv2.line(frame, (cursor_x, y - bar_height), (cursor_x, y), color, 2)

    # Write the valence value as text
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
    # Define the color based on arousal value
    if arousal >= 0.25:
        color = (0, 255, 0)  # Green for positive valence
    elif arousal <= -0.25:
        color = (0, 0, 255)  # Red for negative valence
    else:
        color = (255, 255, 0)  # Blue for neutral valence

    # Calculate the height of the bar based on the arousal
    bar_height = int(h)

    # Draw the base bar
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), -1)

    # Calculate the cursor position within the bar
    cursor_y = y + int(bar_height * (-arousal + 1) / 2)

    # Draw the cursor
    cv2.line(frame, (x, cursor_y), (x + bar_width, cursor_y), color, 2)

    # Write the arousal value as text
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
    ]
    max_indices = outputs_cls.argmax(
        dim=1
    )  # Get the index of the highest value for each sample in the batch
    emotions_batch = [
        emotions[idx.item()] for idx in max_indices
    ]  # Map the indices to emotion labels
    return emotions_batch

    # Write the valence value as text
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
    nn.Linear(block_channels, 9, bias=False),
)
MODEL.load_state_dict(
    torch.load("best_model_affectnet_improved7VA.pt", map_location=torch.device("cpu"))
)
MODEL.eval()

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

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Add text to the frame
        text = "Press 'q' to quit"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Call the FER top_emotion function on the current frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_classifier.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        # Loop over multiple detected faces
        for x, y, w, h in faces:
            # Cut out the face from the frame
            face_roi = frame[y : y + h, x : x + w]

            # Process the face
            img = test_transform(face_roi)  # Apply the transformations
            img = img.unsqueeze(0)  # Add a batch dimension for the model
            outputs = MODEL(img)
            outputs_cls = outputs[:, :7]
            valence = outputs[:, 7:8].item()
            arousal = outputs[:, 8:].item()

            # Draw the valence bar over the face
            draw_valence_bar(frame, valence, x, y, w)

            # Draw the arousal bar to the right of the face
            draw_arousal_bar(frame, arousal, x + w, y, h)

            # Write the emotion label as text
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

        # Display the frame
        cv2.imshow("Webcam", frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
