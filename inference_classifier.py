from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import pickle
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import collections

app = FastAPI()

# Load the model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary with Gujarati characters
labels_dict = {0: 'અ', 1: 'ઇ', 2: 'ઉ'}

# Font settings
font_size = 40
font_path = './NotoSansGujarati.ttf'  # Update this path
font = ImageFont.truetype(font_path, font_size)

# Maintain a frame counter to limit to 500 frames
MAX_FRAMES = 500

# WebSocket for real-time communication with the frontend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    frame_count = 0  # Counter for how many frames are processed

    try:
        while frame_count < MAX_FRAMES:  # Stop after 500 frames
            data = await websocket.receive_text()
            
            # Decode the base64 image sent from the client
            image_data = base64.b64decode(data.split(",")[1])
            np_img = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Process the frame using MediaPipe Hands
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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
                    expected_features = model.n_features_in_
                    if len(data_aux) != expected_features:
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

                    # Draw text with background
                    bbox = draw.textbbox(text_position, predicted_character, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    background_position = (text_position[0] - 5, text_position[1] - text_height - 15)
                    draw.rectangle(
                        [background_position, (background_position[0] + text_width + 10, background_position[1] + text_height + 10)],
                        fill=(0, 0, 0)
                    )
                    draw.text(text_position, predicted_character, font=font, fill=(255, 255, 255))

                    # Convert back to OpenCV format
                    frame = np.array(pil_image)

            # Encode the frame to send it back
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f"data:image/jpeg;base64,{frame_base64}")
            
            frame_count += 1  # Increment frame count after processing each frame

    except WebSocketDisconnect:
        print("Client disconnected")

    print(f"Processed {frame_count} frames.")
