import os
import cv2

# Directory where images will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3 
dataset_size = 100

# Use default camera index 0, change to 1 or 2 if needed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

for j in range(number_of_classes):
    # Create a directory for each class
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display instructions and wait for user to press 'Q' to start
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        cv2.putText(frame, 'Ready? Press "Q" to start collecting images.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Collect and save images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
