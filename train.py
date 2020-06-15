from tensorflow import keras

CLASS_NUM = 6
train_data = './train/'
val_data = './val/'
MODEL_SAVED_FILE = './models/DenseNet201_128.hdf5'

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(128, 128),
        batch_size=128,
        class_mode='categorical',
        shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        val_data,
        target_size=(128, 128),
        batch_size=128,
        class_mode='categorical',
        shuffle=True)

model_resnet50_conv = keras.applications.DenseNet121(weights='imagenet', include_top=False)
image_input = keras.layers.Input(shape=(128, 128, 3), name='input')
x = model_resnet50_conv(image_input)
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
                    epochs=50,
                    shuffle=True,
                    validation_data=(validation_generator),
                    callbacks=cb_list)
