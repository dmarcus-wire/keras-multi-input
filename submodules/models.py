# import packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# create MLP
# operate on house attributes
def create_mlp(dim, regress=False):
    # define it
    model = Sequential()
    # 8 nodes fully conntected with relu
    model.add(Dense(8, input_dim=dim, activation="relu"))
    # 4 nodes fully conntected with relu
    model.add(Dense(4, activation="relu"))

    # check to see if regression should be added
    if regress:
        # goal is multi-input design, so regress is set to false
        # thus, output is 4 nodes
        model.add(Dense(1, activation="linear"))

    # return the model
    return model

# create CNN
def create_cnn(width, height, depth, filters=(16,32,64), regress=False):
    # initialize the input shape and channel dimensions
    # assume TensorFlow channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer, set the input
        if i == 0:
            x = inputs

        # CONV > RELU > BN > POOL
        # f = 16, 32 or 64
        x = Conv2D(f, (3, 3), padding="same")(x)
        # relu activation
        x = Activation("relu")(x)
        # batch normalization
        x = BatchNormalization(axis=chanDim)(x)
        # reduce spatial dimensions
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC > RELU > BN > DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    # help reduce overfitting
    x = Dropout(0.5)(x)

    # apply another FC layer, to match the number of
    # nodes coming from the MLP
    # we want to concatenate the MLP and CNN for a single predictor
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model