from keras.models import load_model
from load_data_test import load_data

model = load_model('./model.h5')
pred = model.predict(load_data())
for p in pred:
    print(p[0])