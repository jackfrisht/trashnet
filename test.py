import os
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras

CLASS_NUM = 6
TEST_PATH = './test'
MODEL_SAVED_FILE = './models/resnet50.hdf5'

model_resnet50_conv = keras.applications.ResNet50V2(weights='imagenet', include_top=False)
image_input = keras.layers.Input(shape=(128, 128, 3), name='input')
x = model_resnet50_conv(image_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(CLASS_NUM, activation='softmax', name='predictions')(x)

model = keras.models.Model(inputs=image_input, outputs=x)
model.summary()

model.load_weights(MODEL_SAVED_FILE)

X_pred, ids = [], []
for image in sorted(os.listdir(TEST_PATH)):
    image_path = os.path.join(TEST_PATH, image)
    print(image_path)
    if image.endswith('.jpg'):
        img = cv2.resize(cv2.imread(image_path), (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255
        X_pred.append(img)
        ids.append(image[:-4])
X_pred = np.array(X_pred)
ids = np.array(ids)

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

y_pred = model.predict(X_pred)

for i, item in enumerate(y_pred):
    _id = ids[i]
    food = classes[np.argmax(item)]
    # label.append(character)
    print(food)

# df_submit = pd.DataFrame({'Id': ids, 'character': label})
# df_submit.to_csv('submission.csv', index=False)


