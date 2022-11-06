# Importing the Libraries

pwd


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='SW_rFlhYMnrPkeBKApVZmABknriA18QF20zMou6MY0l7',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'smartmathematicstutor-donotdelete-pr-qs8pw1zdcmk7gn'
object_key = 'Dataset.zip'

streaming_body_2 = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/


from io import BytesIO
import zipfile
unzip=zipfile.ZipFile(BytesIO(streaming_body_2.read()),'r')
file_paths=unzip.namelist()
for path in file_paths:
    unzip.extract(path)

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout

from tensorflow.keras.preprocessing .image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

pwd

ls

cd /home/wsuser/work/test

# Loading the data and performing data agumentation

x_train = train_datagen.flow_from_directory(directory=r"/home/wsuser/work/test",target_size = (64,64),batch_size = 32,class_mode = 'categorical')
x_test = test_datagen.flow_from_directory(directory=r'/home/wsuser/work/train',target_size = (64,64),batch_size = 32,class_mode = 'categorical')

x_train.class_indices

# Building the Model

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape = (64,64,3)))

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(units = 128, kernel_initializer= "random_uniform",activation = "relu"))

model.add(Dense(units = 3 , kernel_initializer= "random_uniform",activation = "softmax"))

# Compiling the model

model.compile(optimizer= "adam",loss = "categorical_crossentropy" , metrics =["accuracy"])

# Fitting the model

model.fit_generator(x_train,steps_per_epoch = 342 , epochs = 100 , validation_data = x_test,validation_steps = 10)

model.summary()

# Saving our model

model.save("shapes.h5")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("shapes.h5")

!tar -zcvf dataset.tgz shapes.h5

ls -1

!pip install watson-machine-learning-client --upgrade

from ibm_watson_machine_learning import APIClient
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey":"PaGStrzZMY89uAASG0GiDBKP6yQ7qy8AMH8zGCmCJMp-"
}
client = APIClient(wml_credentials)

client = APIClient(wml_credentials)

def guid_from_space_name(client, space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']["name"] == space_name)['metadata']['id'])

space_uid = guid_from_space_name(client, 'imageclassification')
print("Space UID = " + space_uid)

client.set.default_space(space_uid)

client.software_specifications.list()

software_spec_uid = client.software_specifications.get_uid_by_name('tensorflow_rt22.1-py3.9')
software_spec_uid

ls

 model_details=client.repository.store_model(model='dataset.tgz',meta_props={
 client.repository.ModelMetaNames.NAME:"CNN",
 client.repository.ModelMetaNames.TYPE:"tensorflow_2.7",
 client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_spec_uid})

model_id=client.repository.get_model_id(model_details)

model_id

client.repository.download('a4bc1ab7-6234-4287-8971-e135056903dd','datasetsmt.tgz')

ls

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



