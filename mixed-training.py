# USAGE
# python mixed-training.py -d dataset

# import packages
from submodules import datasets # for house attr and iamges from disk
from submodules import models # cnn implementation + MLP
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense # fully connected layer node
from tensorflow.keras.models import Model # multiple inputs and mixed data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate # concate output house attr + house images
import numpy as np
import argparse
import locale # format output to terminal
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct path to .txt file
print("[INFO] loading house attributes...")
# file is raw with bathrooms, bedrooms, area, zipcode and price
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
# load the file from disk
df = datasets.load_house_attributes(inputPath)

# load the images
print("[INFO] loading house images...")
# construct the image montages
images = datasets.load_house_images(df, args["dataset"])
# scale intensities from 0 to 1
images = images / 255.0

# split the data 75% : 25%
print("[INFO] processing data...")
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largest house price
# reduce degress of freedom from model
# scale 0 to 1
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# process the house attr. setting 0 to 1
(trainAttrX, testAttrX) = datasets.process_house_attributes(df,
	trainAttrX, testAttrX)

# create MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

# combine the output
# will have 4 entries in it
combinedInput = concatenate([mlp.output, cnn.output])

# set regression head
# 4 FC nodes
x = Dense(4, activation="relu")(combinedInput)
# 1 FC nodes - actual regression head = predicting price
x = Dense(1, activation="linear")(x)

# set the model
model = Model(inputs=[mlp.input, cnn.input], outputs=x)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(
	x=[trainAttrX, trainImagesX], y=trainY,
	# set valiation data
	validation_data=([testAttrX, testImagesX], testY),
	epochs=200, batch_size=8)

# predict with the model on testing data
print("[INFO] predicting house prices...")
# make predictions on the homes on 2 types of inputs we supply
preds = model.predict([testAttrX, testImagesX])

# compute the diff beetween actual
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))

# MAE off by about 31% +/-
# std off by about 22% +/-
# running off just MLP outperforms
# not taking into consideration area
# overall, CNN not efficient to predict home prices
# healthcare can be valuable as it involves numerical, categorical and imagery
