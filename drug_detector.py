import torch
import cv2
import os
import torchvision
from telegram_sender import send_msg

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace with path to your model
import time

# Create a folder to store detected images if it doesn't exist
output_folder = "detected_images"
if not os.path.exists(output_folder):
   os.makedirs(output_folder)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Adjust the video source if needed

frame_count = 0  # Counter to track frames
last_save_time = time.time()  # Time of last image save

while True:
   # Capture frame-by-frame
   ret, frame = cap.read()

   # Perform object detection
   results = model(frame)
   print(type(results))

   # Check for detections with confidence above threshold
   for detection in results.pandas().xyxy[0].itertuples():
       confidence = detection[5]
       if confidence > 0.45:
           frame_count += 1

           # Check if it's time to save the image
           if frame_count % 10 == 0 and time.time() - last_save_time >= 10:
               last_save_time = time.time()
               filename = f"detection_{frame_count}.jpg"  # Adjust filename format if needed
               output_path = os.path.join(output_folder, filename)
               result = model(frame)
               result.render()
               cv2.imwrite(output_path, frame)
               res = send_msg(output_path)

   # Draw bounding boxes and labels
   results.render()  # Render results directly on frame

   # Display the resulting frame
   cv2.imshow('YOLOv5 Real-Time Detection', frame)

   # Exit if 'q' key is pressed
   if cv2.waitKey(1) == ord('q'):
       break

# Release resources
cap.release()
cv2.destroyAllWindows()