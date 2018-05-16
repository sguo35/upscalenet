import keras
from losses import pixelwise_crossentropy
from keras.models import load_model

from dataGenerator import DataGenerator

model = load_model('./model.h5')
model.compile(optimizer='adam', loss=pixelwise_crossentropy, metrics=['acc'])

dataGen = DataGenerator()
model.fit_generator(dataGen, epochs=100, workers=1, use_multiprocessing=False)