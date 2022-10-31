from keras.models import load_model
import numpy as np
import cv2
cat_model = load_model('model.h5')
img = cv2.imread('9.jpg')
img1 = cv2.resize(img, (64, 64))

data = np.array(img1, dtype="float32") / 255.0
data = data.reshape(1, 64, 64 ,3)
pre = cat_model.predict(data)[0][0]
print(pre)
if pre > 0.5:
    print('dog')
else:
    print('cat')