import cv2
import numpy as np
import os

def load(folder_path):
    X, y = [], []
    label_dict = {}  
    label_id = 0
    for p_name in os.listdir(folder_path):
        p_path = os.path.join(folder_path, p_name)
        if not os.path.isdir(p_path):
            continue
        if p_name not in label_dict:
            label_dict[p_name] = label_id
            label_id += 1
        for image_name in os.listdir(p_path):
            image_path = os.path.join(p_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            X.append(img)
            y.append(label_dict[p_name])

    return np.asarray(X, dtype=np.float32), np.array(y), label_dict  

train_path = "face_train"
X_train, y_train, label_dict = load(train_path)

# Create and train model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(X_train, y_train)
model.save("model_f.yml")
np.save("label_dict.npy", label_dict)

print("Model trained and saved successfully!")
