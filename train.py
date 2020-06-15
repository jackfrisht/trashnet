from tensorflow import keras
import argparse
import os

MODEL_MAP = {'resnet50': keras.applications.ResNet50V2,
             'resnet152': keras.applications.ResNet152V2,
             'inceptionresnet': keras.applications.InceptionResNetV2,
             'densenet121': keras.applications.DenseNet121,
             'densenet201': keras.applications.DenseNet201}

"""Parses arguments."""
parser = argparse.ArgumentParser(description='Train TrashNet')
parser.add_argument('-n', '--model_name', type=str, default='resnet50', help='Name of model.')
parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch Size.')
parser.add_argument('-res', '--input_size', type=int, default=128, help='Input image resolution')
parser.add_argument('-epoch', '--epoch', type=int, default=50, help='Maximum Epoch')

args = parser.parse_args()


CLASS_NUM = 6
train_data = './train/'
val_data = './val/'
inputsize = args.input_size
MODEL_SAVED_FILE = os.path.join('./models/', args.model_name + '.hdf5')

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(inputsize, inputsize),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        val_data,
        target_size=(inputsize, inputsize),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True)

model_ = MODEL_MAP[args.model_name](weights='imagenet', include_top=False)
image_input = keras.layers.Input(shape=(inputsize, inputsize, 3), name='input')
x = model_(image_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(CLASS_NUM, activation='softmax', name='predictions')(x)

model = keras.models.Model(inputs=image_input, outputs=x)
model.summary()

# opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(MODEL_SAVED_FILE,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

def lr_scheduler(epoch):
    if epoch < 10:
        lr = 0.001
    elif epoch < 20:
        lr = 0.0001
    else:
        lr = 0.00001
    print('lr: %f' % lr)
    return lr

scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler)
cb_list = [checkpoint, scheduler]

history = model.fit(train_generator,
                    epochs=args.epoch,
                    shuffle=True,
                    validation_data=(validation_generator),
                    callbacks=cb_list)
