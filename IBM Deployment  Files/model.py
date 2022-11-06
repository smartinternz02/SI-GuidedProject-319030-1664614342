from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


img1 = image.load_img(r"/home/wsuser/work/test/circle/0.png", target_size = (64,64))

type(img1)

x = image.img_to_array(img1)

x.shape

x = np.expand_dims(x, axis = 0)

x.shape

xpred = np.argmax(model.predict(x))

xpred

img2 = image.load_img(r"/home/wsuser/work/test/square/1.png", target_size = (64,64))

y = image.img_to_array(img2)

y.shape

y = np.expand_dims(y, axis = 0)

y.shape

ypred = np.argmax(model.predict(y))

ypred

index = ['square', 'circle', 'triangle']

prediction = index[xpred]

prediction

prediction = index[ypred]

prediction

