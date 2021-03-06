# Stacked LSTM with residual connections in depth direction.
#
# Naturally LSTM has something like residual connections in time.
# Here we add residual connection in depth.
#
# Inspired by Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144).
# They observed that residual connections allow them to use much deeper stacked RNNs.
# Without residual connections they were limited to around 4 layers of depth.
#
# It uses Keras 2 API.

from keras.layers import CuDNNLSTM, Lambda, Dense, Input
from keras.models import Model
from keras.layers.merge import add
from keras import regularizers

def make_residual_lstm_layers(input, rnn_width, rnn_depth):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = CuDNNLSTM(rnn_width, return_sequences=return_sequences, kernel_regularizer=regularizers.l2(0.01))(x)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
    return x


def lstm_model(input_layer):
    m = make_residual_lstm_layers(input_layer, 128, 4)
    m = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(m)
    model = Model(inputs=[input_layer], outputs=[m])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
    
    return model
input_subject_category = Input((200,99))
model = lstm_model(input_subject_category)

print(model.summary())

model.save('./model.h5')