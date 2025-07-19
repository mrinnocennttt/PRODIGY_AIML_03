# PRODIGY_AIML_03
SVM for Cats vs Dogs Classification
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tqdm import tqdm

# Define paths (example path, adjust accordingly)
CAT_PATH = '/path/to/cats/'
DOG_PATH = '/path/to/dogs/'

# Load and preprocess data
data = []
labels = []

IMG_SIZE = 64

for img_name in tqdm(os.listdir(CAT_PATH)):
    img = cv2.imread(os.path.join(CAT_PATH, img_name))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data.append(img.flatten())
    labels.append(0)

for img_name in tqdm(os.listdir(DOG_PATH)):
    img = cv2.imread(os.path.join(DOG_PATH, img_name))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data.append(img.flatten())
    labels.append(1)

X = np.array(data)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
