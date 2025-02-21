import cv2

# Open video file
video = cv2.VideoCapture(r"C:\Users\Ben\Downloads\ACT 3\input.mp4")  

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define different output formats
fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 format
fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')  # AVI format

out_mp4 = cv2.VideoWriter('output.mp4', fourcc_mp4, fps, (frame_width, frame_height))
out_avi = cv2.VideoWriter('output.avi', fourcc_avi, fps, (frame_width, frame_height))

while True:
    success, frame = video.read()
    if not success:
        break
    out_mp4.write(frame)
    out_avi.write(frame)
    cv2.imshow('Video Playback', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

video.release()
out_mp4.release()
out_avi.release()
cv2.destroyAllWindows()
