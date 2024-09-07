import warnings
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary with Gujarati characters
labels_dict = {0: 'અ', 1: 'ઇ', 2: 'ઉ'}

# Font size and path
font_size = 40
font_path = './NotoSansGujarati.ttf'  # Update this path

# Check font path and load font
try:
    font = ImageFont.truetype(font_path, font_size)
except OSError as e:
    print(f"Error loading font: {e}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark coordinates
            x_ = []
            y_ = []
            data_aux = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Ensure data_aux length matches the model's expected input length
            expected_features = model.n_features_in_  # This assumes your model has this attribute
            if len(data_aux) != expected_features:
                print(f"Warning: Feature length mismatch. Expected {expected_features}, got {len(data_aux)}.")
                continue

            # Prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Convert frame to PIL Image
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)

            # Calculate text position
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            text_position = (x1, y1 - 10)

            # Calculate text size using textbbox for newer Pillow versions
            bbox = draw.textbbox(text_position, predicted_character, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw text with background for visibility
            background_position = (text_position[0] - 5, text_position[1] - text_height - 15)
            draw.rectangle(
                [background_position, (background_position[0] + text_width + 10, background_position[1] + text_height + 10)],
                fill=(0, 0, 0)  # Black background
            )
            draw.text(text_position, predicted_character, font=font, fill=(255, 255, 255))  # White text

            # Convert back to OpenCV format
            frame = np.array(pil_image)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
