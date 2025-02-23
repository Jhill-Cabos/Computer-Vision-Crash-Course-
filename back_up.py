import numpy as np
import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk

def preprocess_image(img, sz=(200, 200)):
    img = cv2.resize(img, sz)
    img = cv2.equalizeHist(img)  
    return img

def read_images(path, sz=(200, 200)):
    X, y = [], []
    label = 0
    names = []
    for subdir, dirs, files in os.walk(path):
        for subdirname in dirs:
            names.append(subdirname) 
            subject_path = os.path.join(subdir, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    filepath = os.path.join(subject_path, filename)
                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    img = preprocess_image(img, sz)
                    X.append(np.asarray(img, dtype=np.uint8))
                    y.append(label)
                except Exception as e:
                    print("Error:", e)
            label += 1
    return [X, np.asarray(y, dtype=np.int32)], names
(X, y), names = read_images("Faces")
eigen_model = cv2.face.EigenFaceRecognizer_create()
fisher_model = cv2.face.FisherFaceRecognizer_create()
lbph_model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0)
eigen_model.train(X, y)
fisher_model.train(X, y)
lbph_model.train(X, y)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
root = tk.Tk()
root.title("Face Recognition GUI")
root.geometry("1200x500")

frame_results = tk.Frame(root)
frame_results.pack()

def create_frame(title):
    frame = tk.Frame(frame_results)
    frame.pack(side=tk.LEFT, padx=10)
    label = tk.Label(frame, text=title, font=("Arial", 12))
    label.pack()
    img_label = tk.Label(frame)
    img_label.pack()
    return label, img_label

eigen_label, eigen_image_label = create_frame("EigenFace:")
fisher_label, fisher_image_label = create_frame("FisherFace:")
lbph_label, lbph_image_label = create_frame("LBPH:")

camera = cv2.VideoCapture(0)

def update_frame():
    ret, frame = camera.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eigen_img, fisher_img, lbph_img = frame.copy(), frame.copy(), frame.copy()
    
    for (x, y, w, h) in faces:
        roi = preprocess_image(gray[y:y + h, x:x + w])
        
        eigen_pred, eigen_conf = eigen_model.predict(roi)
        fisher_pred, fisher_conf = fisher_model.predict(roi)
        lbph_pred, lbph_conf = lbph_model.predict(roi)
        
        predicted_name = names[lbph_pred] 
        color = (0, 0, 255) if predicted_name == "Papa" else (255, 0, 0)
        
        cv2.rectangle(eigen_img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(fisher_img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(lbph_img, (x, y), (x + w, y + h), color, 2)
        
        eigen_label.config(text=f"EigenFace: {names[eigen_pred]} (Conf: {eigen_conf:.2f})")
        fisher_label.config(text=f"FisherFace: {names[fisher_pred]} (Conf: {fisher_conf:.2f})")
        lbph_label.config(text=f"LBPH: {names[lbph_pred]} (Conf: {lbph_conf:.2f})")
    
    def update_tk_image(cv_img, label):
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        label.configure(image=img)
        label.image = img

    update_tk_image(eigen_img, eigen_image_label)
    update_tk_image(fisher_img, fisher_image_label)
    update_tk_image(lbph_img, lbph_image_label)

    root.after(10, update_frame)

def on_closing():
    camera.release()
    cv2.destroyAllWindows()
    root.quit()

root.protocol("WM_DELETE_WINDOW", on_closing)
update_frame()
root.mainloop()
