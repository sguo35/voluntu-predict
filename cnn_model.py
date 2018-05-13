# Example usage
from keras import regularizers
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Conv1D, GlobalMaxPool1D,SpatialDropout1D,CuDNNGRU,Bidirectional,PReLU,GRU, BatchNormalization
from keras.layers import GlobalAveragePooling1D
from resnet import residual_network
from keras.models import Model
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

# changeable parameters
MULTIPLIER = int(1)
L2_regularizer = 0.01

def get_model():
    input_subject_category = Input((200,99))
    subject_category_network = residual_network(input_subject_category)

    predictions = Dense(2, activation="softmax")(subject_category_network)

    model = Model(inputs=[input_subject_category], outputs=predictions)
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

    return model

model = get_model()

print(model.summary())

model.save('./model.h5')