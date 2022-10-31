from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_data = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
test_data = ImageDataGenerator(rescale=1./255)

train_set = train_data.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )
test_set = test_data.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

# model.fit(train_set,
#         epochs=3,
#     )
model.fit_generator(
        train_set,
        # steps_per_epoch=800,
        epochs=10,
        validation_data=test_set,
        # validation_steps=200
        )
model.save('model.h5')
# img = cv2.imread('3.jpg')
# img1 = cv2.resize(img, (64, 64))

# data = np.array(img1, dtype="float32") / 255.0
# data = data.reshape(1, 64, 64 ,3)
# pre = model.predict(data)[0][0]
# print(pre)
# if pre > 0.5:
#     print('dog')
# else:
#     print('cat')