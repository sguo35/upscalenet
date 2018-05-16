import keras
from losses import pixelwise_crossentropy
from keras.models import load_model

model = load_model('./model.h5')
model.compile(optimizer='adam', loss=pixelwise_crossentropy)
