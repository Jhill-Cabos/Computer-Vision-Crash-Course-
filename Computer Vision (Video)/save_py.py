import numpy as np
import csv
import os
import cv2

def read_images(path, sz=None):
    X_jhill, y_jhill = [], []
    X_frank, y_frank = [], []
    label_map = {"Jhill": 0, "Papa": 1}  
    
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            label = label_map.get(subdirname, -1)  
            if label == -1:
                continue
            for filename in os.listdir(subject_path):
                try:
                    if filename == ".directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if sz is not None:
                        im = cv2.resize(im, (200, 200))
                    if label == 0:
                        X_jhill.append(np.asarray(im, dtype=np.uint8))
                        y_jhill.append(label)
                    elif label == 1:
                        X_frank.append(np.asarray(im, dtype=np.uint8))
                        y_frank.append(label)
                except Exception as e:
                    print("Error loading image:", str(e))
    
    return [(X_jhill, y_jhill, "jhill.csv"), (X_frank, y_frank, "papa.csv")]

def save_images_and_labels_to_csv(images, labels, csv_filename):
    assert len(images) == len(labels), "Mismatch between number of images and labels"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        num_pixels = images[0].size
        header = ["label"] + [f"pixel{i}" for i in range(num_pixels)]
        writer.writerow(header)
        for img, label in zip(images, labels):
            img_flattened = img.flatten()
            row = np.insert(img_flattened, 0, label)
            writer.writerow(row)
    print(f"Images and labels saved to {csv_filename}")

if __name__ == "__main__":
    datasets = read_images("Faces", 1)
    for images, labels, filename in datasets:
        if images:  # Ensure we only save if images exist
            save_images_and_labels_to_csv(images, labels, filename)