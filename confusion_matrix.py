import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
import re
from tensorflow import keras

CLASS_NUM = 6
TEST_PATH = './test'
MODEL_SAVED_FILE = './models/DenseNet201_128.hdf5'

model_resnet50_conv = keras.applications.DenseNet201(weights='imagenet', include_top=False)
image_input = keras.layers.Input(shape=(128, 128, 3), name='input')
x = model_resnet50_conv(image_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(CLASS_NUM, activation='softmax', name='predictions')(x)

model = keras.models.Model(inputs=image_input, outputs=x)
model.summary()

model.load_weights(MODEL_SAVED_FILE)
X_pred, gt_label = [], []
for image in sorted(os.listdir(TEST_PATH)):
    image_path = os.path.join(TEST_PATH, image)
    if image.endswith('.jpg'):
        img = cv2.resize(cv2.imread(image_path), (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255
        X_pred.append(img)
        gt_label.append(''.join(re.findall('cardboard|glass|metal|paper|plastic|trash', image[:-4])))
X_pred = np.array(X_pred)

print(gt_label)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

y_pred = model.predict(X_pred)

pred_label = []
for i, item in enumerate(y_pred):
    label = classes[np.argmax(item)]
    pred_label.append(label)

plt.figure(figsize=(8, 8))
conf_matrix = confusion_matrix(gt_label, pred_label)
classs = list(classes)
plt.imshow(conf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classs))
plt.xticks(tick_marks, classs)
plt.yticks(tick_marks, classs)
plt.savefig('conf_matrix.pdf')
plt.show()

count = 0
for i in range(len(y_pred)):
    if pred_label[i] == gt_label[i]:
        count+=1
print('Accuracy is ', count / len(y_pred))

df_submit = pd.DataFrame({'gt': gt_label, 'pred': pred_label})
df_submit.to_csv('result.csv', index=False)
