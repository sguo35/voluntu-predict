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

from keras.layers import LSTM, Lambda, Bidirectional, BatchNormalization
from keras.layers.merge import add
import keras

def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = Bidirectional(LSTM(rnn_width, 
        recurrent_dropout=rnn_dropout, dropout=rnn_dropout, 
        return_sequences=return_sequences, implementation=2, 
        kernel_regularizer=keras.regularizers.l2(0.02)))(x)
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

if __name__ == '__main__':
    # Example usage
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model
    from keras.callbacks import ReduceLROnPlateau
    from keras.optimizers import SGD
    from load_data import load_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    x, y = load_data()    
    
    input = Input(shape=(200, 84))
    output = make_residual_lstm_layers(input, rnn_width=128, rnn_depth=4, rnn_dropout=0.4)
    output = Dropout(0.4)(output)
    output = Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.02))(output)

    model = Model(inputs=input, outputs=output)

    model.compile(optimizer=SGD(0.01, nesterov=True), loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    model.fit(x, y, batch_size=128, epochs=200, verbose=1, validation_split=0.2)#, callbacks=[reduce_lr])
    model.save('./model.h5')
    import gc
    gc.collect()