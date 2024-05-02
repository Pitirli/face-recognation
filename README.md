# face-recognation
import cv2

# Create a camera connection
cm = cv2.VideoCapture(0)

# Identifying the red area
show_red_area = True

# Coordinates of the red area (x= width, y= height)
red_area = (0, 0, 640, 640)

# Installing Haar feature-based classifier for face recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables for time measurement
startTime = None
endTime = None

while True:
    # Reading camera image
    ret, frame = cm.read()
    if not ret:
        break

    # Show red area
    if show_red_area:
        x, y, c, v = red_area
        cv2.rectangle(frame, (x, y), (x + c, y + v), (0, 0, 255), 2)

    # Converting the image within the red area to grayscale
    gray = cv2.cvtColor(frame[y:y + v, x:x + c], cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Calculate the time the face is detected
    if len(faces) > 0:
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + x2, y + y1 + y2), (0, 255, 0), 2)

        # Save start time
        if startTime is None:
            startTime = cv2.getTickCount()
        endTime = None
        show_red_area = False
    else:
        # Calculate processing time when no face detected
        if endTime is None and startTime is not None:
            endTime = cv2.getTickCount()
            elapsed_time = (endTime - startTime) / cv2.getTickFrequency()
            print("Yüz algılanma süresi: {:.2f} saniye".format(elapsed_time))

            # Reset time and show red area again
            startTime = None
            endTime = None
            show_red_area = True

        # Print processing time on image
        cv2.putText(frame, f'Time: {elapsed_time:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    # Showing the image on the screen
    cv2.imshow('frame', frame)

    # Use q key to release camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera connection
cm.release()

# Close all windows
cv2.destroyAllWindows()
