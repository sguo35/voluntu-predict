# Voluntu-Predict
Predicts the validity of volunteer hours. Trained on a cleaned subset of ~5k Voluntu hour submissions from 2017.

# Model
- `cnn_model.py` is the latest and highest performing model
- ResNeXT-like architecture w/ 1D convolutions for textual processing (see `resnet.py` for paper link)
- `resnet.py` contains CNN model definition and hyperparameters
- CNN outperforms LSTM (~0.87 F-score)
- Run `cnn_model.py` first to generate the model, then run `cnn_train.py` to train.
- Prediction/inference code is outdated, do not use

# Data
- `out.csv` contains hour entries
- Each entry has text description, start/end times, organization UUID, supervisor UUID
- Dataset is highly unbalanced, so we duplicate negative data 20x

# Miscellaneous
- `clr_callback.py` contains cyclic learn rate callback, do not use unless you're using regular SGD instead of Adam
