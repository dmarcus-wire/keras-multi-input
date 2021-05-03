# import necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

# load house attributes function for parsing the tct file
# model should predict the final column
def load_house_attributes(inputPath):
    # init the list of column names in the CSV
    cols = ["bedrooms","bathrooms","area","zipcode","price"]
    # load using pandas
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

    # data preprocessing
    # determine the unique zip codes
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each unique zip and corresponding count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zip code counts for the dataset is unbalanced
        # sanitize data by removing houses with less than 25
        # in the same zip
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

        # return the data frome
        return df

# process house attributes
# uses the previous data frame
def process_house_attributes(df, train, test):
    # init the colum names of the continuous data
    continuous = ["bedrooms","bathrooms","area"]

    # perform min-max scaling each continuous column in the range 0 to 1
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # one hot encode the zip categorical data vector from 0 to 1
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    # construct train/test data concatenating categorical  with cont. features
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    # return concatenated data
    return( trainX, testX)

# load house images
# each home has 4 images of the same areas
# plan to train the model on a montage of these images
def load_house_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []

	# loop over the indexes of the houses
	for i in df.index.values:
		# find the four images for the house and sort the file paths,
		# ensuring the four are always in the *same order*
		basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
		housePaths = sorted(list(glob.glob(basePath)))

		# initialize our list of input images along with the output image
		# after *combining* the four input images
		inputImages = []
        # allocate memory for the output 2x2 montage image
		outputImage = np.zeros((64, 64, 3), dtype="uint8")

		# loop over the input house paths
		for housePath in housePaths:
			# load the input image, resize it to be 32 32, and then
			# update the list of input images
			image = cv2.imread(housePath)
			image = cv2.resize(image, (32, 32))
			inputImages.append(image)

		# tile the four input images in the output image such the first
		# image goes in the top-right corner, the second image in the
		# top-left corner, the third image in the bottom-right corner,
		# and the final image in the bottom-left corner
		outputImage[0:32, 0:32] = inputImages[0]
		outputImage[0:32, 32:64] = inputImages[1]
		outputImage[32:64, 32:64] = inputImages[2]
		outputImage[32:64, 0:32] = inputImages[3]

		# add the tiled image to our set of images the network will be
		# trained on
		images.append(outputImage)

	# return our set of images
	return np.array(images)