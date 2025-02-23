import cv2
import numpy as np
import os

def read_images(folder, size):
    X, y = [], []
    label_map = {"Jhill": 0, "Unknown": 1} 

    for label_name, label in label_map.items():
        person_folder = os.path.join(folder, label_name)
        if not os.path.exists(person_folder):
            print(f"Warning: Folder {person_folder} does not exist.")
            continue

        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Unable to read {img_path}")
                continue

            img_resized = cv2.resize(img, (200, 200))  # Resize for consistency
            X.append(img_resized)
            y.append(label)

    if not X or not y:
        return None  

    return [np.array(X), np.array(y)]

def face_rec():
    names = ['Jhill', 'Unknown']  
    [X, y] = read_images("Faces", 1)
    print(y)
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, y)

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                params = model.predict(roi)
                label = names[params[0]]
                cv2.putText(img, label + ", " + str(params[1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except:
                continue

        cv2.imshow("EigenFace", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()