from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


model = load_model("shape.h5") 

def predict(InputImg):
    
    img=image.load_img(InputImg,target_size=(64,64))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    pred=model.predict(x)
    pred=np.argmax(model.predict(x), axis=-1)
    print(pred)
    index=['circle', 'square', 'triangle']
 
    result = str(index[pred[0]])


    return result