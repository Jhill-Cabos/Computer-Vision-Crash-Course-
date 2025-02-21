import cv2

cameraCapture0 = cv2.VideoCapture(0)  
cameraCapture1 = cv2.VideoCapture(1)  # Second camera (or video file)

success0 = cameraCapture0.grab()
success1 = cameraCapture1.grab()

if success0 and success1:
    success0, frame0 = cameraCapture0.retrieve()
    success1, frame1 = cameraCapture1.retrieve()
    
    if success0 and success1:
        cv2.imshow('Frame 0', frame0)
        cv2.imshow('Frame 1', frame1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Failed to grab frames.")

cameraCapture0.release()
cameraCapture1.release()