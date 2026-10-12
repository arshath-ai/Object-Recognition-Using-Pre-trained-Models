# Import the OS module (used to work with file paths)
import os

# Import OpenCV library for computer vision
import cv2

# Import time module
import time

# Import imutils for image resizing
import imutils

# Import NumPy for mathematical operations
import numpy as np


# Get the current folder where this Python file is stored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the path of the MobileNet SSD prototxt file
prototxt = os.path.join(BASE_DIR, "MobileNetSSD_deploy.prototxt.txt")

# Set the path of the trained Caffe model
model = os.path.join(BASE_DIR, "MobileNetSSD_deploy.caffemodel")

# Set the minimum confidence threshold (20%)
confThresh = 0.2


# Check whether the prototxt file exists
if not os.path.exists(prototxt):

    # Print error message
    print("ERROR: Prototxt file not found!")

    # Print the file location
    print(prototxt)

    # Stop the program
    exit()


# Check whether the model file exists
if not os.path.exists(model):

    # Print error message
    print("ERROR: Caffe model file not found!")

    # Print the file location
    print(model)

    # Stop the program
    exit()


# List of object classes supported by MobileNet SSD
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


# Generate a random RGB color for every class
COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))


# Print loading message
print("Loading model...")


# Load the MobileNet SSD model
net = cv2.dnn.readnetfromcaffe(prototxt,model)


# Print success message
print("Model Loaded Successfully")

# Print camera message
print("Starting Camera...")


# Open the webcam
vs = cv2.VideoCapture(2)


# Check whether the camera opened successfully
if not vs.isOpened():

    # Print error
    print("Cannot open webcam!")

    # Stop program
    exit()


# Wait for 2 seconds
time.sleep(2)


# Start infinite loop
while True:

    # Read one frame from webcam
    ret, frame = vs.read()

    # Check whether frame is captured
    if not ret:

        # Print error
        print("Failed to read frame.")

        # Exit loop
        break


    # Resize frame to width 800
    frame = imutils.resize(frame, width=800)


    # Get image height and width
    (h, w) = frame.shape[:2]


    # Convert image into Blob format
    blob = cv2.dnn.blobFromImage(

        # Resize image to 300x300
        cv2.resize(frame,(300,300)),

        # Scale factor
        0.007843,

        # Required input size
        (300,300),

        # Mean subtraction value
        127.5
    )


    # Send blob to the neural network
    net.setInput(blob)


    # Perform object detection
    detections = net.forward()


    # Loop through all detected objects
    for i in range(detections.shape[2]):

        # Get confidence score
        confidence = detections[0,0,i,2]


        # Check confidence threshold
        if confidence > confThresh:

            # Get class ID
            idx = int(detections[0,0,i,1])


            # Ignore invalid class IDs
            if idx >= len(CLASSES):
                continue


            # Get bounding box coordinates
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])


            # Convert coordinates to integers
            (startX,startY,endX,endY)=box.astype("int")


            # Create label with class name and confidence
            label="{}: {:.2f}%".format(
                CLASSES[idx],
                confidence*100
            )


            # Draw rectangle around detected object
            cv2.rectangle(
                frame,
                (startX,startY),
                (endX,endY),
                COLORS[idx],
                2
            )


            # Decide where to display the label
            y = startY-15 if startY>30 else startY+20


            # Display object name and confidence
            cv2.putText(
                frame,
                label,
                (startX,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[idx],
                2
            )


    # Display output window
    cv2.imshow("Object Recognition",frame)


    # Wait for keyboard input
    key=cv2.waitKey(1)&0xFF


    # Press ESC to exit
    if key==27:
        break


# Release the webcam
vs.release()


# Close all OpenCV windows
cv2.destroyAllWindows()

