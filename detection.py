import torch
import cv2
import os

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace with path to your model

# Initialize video capture
input_video_path = 'path_to_input_video.mp4'  # Replace 'path_to_input_video.mp4' with the path to your input video file
cap = cv2.VideoCapture(input_video_path)

# Get video information
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
output_video_path = 'output_video.avi'  # Set the desired output video path
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels
    results.render()  # Render results directly on frame

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
