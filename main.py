from tkinter import *
from tkinter import ttk
from tkinter.ttk import Separator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "gender",
    "body_mass_g",
]

species = [
    "Adelie",
    "Gentoo",
    "Chinstrap",
]


# Visualize Penguins dataset
# and draw all possible combinations between them
def plotGraph(originalDF, xAxis, yAxis):
    plt.figure('Graph')
    plt.cla()

    # Separate each class in different dataframe
    # Then Draw the data frame on xAxis and yAxis
    class1DataFrame = originalDF.loc[originalDF['species'].isin(['Adelie'])]
    plt.scatter(class1DataFrame[xAxis], class1DataFrame[yAxis], color='red')

    class2DataFrame = originalDF.loc[originalDF['species'].isin(['Gentoo'])]
    plt.scatter(class2DataFrame[xAxis], class2DataFrame[yAxis], color='blue')

    class3DataFrame = originalDF.loc[originalDF['species'].isin(['Chinstrap'])]
    plt.scatter(class3DataFrame[xAxis], class3DataFrame[yAxis], color='green')

    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.show()


# Remove null values from gender column and convert it to numerical values
# Normalize all values of all features to range between 0 and 1
def dataNormalization(dataFrame):
    # Get number of male and females
    numberOfMales = dataFrame.gender.value_counts().male
    numberOfFemales = dataFrame.gender.value_counts().female
    # Merge the Null values with the class with the highest number
    if numberOfMales > numberOfFemales:
        dataFrame.gender.replace({np.NAN: 'male'}, inplace=True)
    else:
        dataFrame.gender.replace({np.NAN: 'female'}, inplace=True)

    dataFrame.gender.replace({'male': 1, 'female': 0}, inplace=True)

    # No normalize for strings
    # So, remove species column from dataframe to apply normalization
    speciesDF = dataFrame[['species']]
    dataFrame = dataFrame.drop(columns=['species'])

    # normalization
    for column in dataFrame.columns:
        dataFrame[column] = dataFrame[column] / dataFrame[column].abs().max()

    # Merge species column again to dataframe
    frames = [speciesDF, dataFrame]
    dataFrame = pd.concat(frames, axis=1)
    return dataFrame


# get all data from gui
def getDataFromGUI():
    feature1 = feature1Value.get()
    feature2 = feature2Value.get()
    class1 = class1Value.get()
    class2 = class2Value.get()
    etaValue = float(learningRateTextField.get())
    epochsValue = int(numberOfEpochsTextField.get())
    weightMatrix = np.random.rand(2, 1)
    biasValue = np.zeros((1, 1))
    if biasCheckBox.get() == 0:
        biasValue = np.zeros((1, 1))
        bias = 0
    else:
        biasValue = np.ones((1, 1))
        bias = 1
    weightMatrix = np.concatenate([biasValue, weightMatrix])
    class1train, class1test, class2train, class2test = dataSplitter(class1, class2, feature1, feature2,
                                                                    originalDataframe)

    # To merge train sets together the shuffle them
    trainData = pd.concat([class1train, class2train])
    trainData = shuffle(trainData)
    testData = pd.concat([class1test, class2test])
    testData = shuffle(testData)

    weightMatrix = train(trainData, weightMatrix, feature1, feature2, bias, etaValue, epochsValue)
    test(weightMatrix, testData, feature1, feature2, bias)


# Replace selected 2 species with numerical values and drop the 3rd class
# Split train and test dataframes and shuffle them
def dataSplitter(class1, class2, feature1, feature2, dataframe):
    # Replace first class with -1 and second with 1
    dataframe.species.replace({class1: -1, class2: 1}, inplace=True)
    # remove the unwanted third class
    unwantedClass = dataframe[dataframe['species'] != -1].index & dataframe[dataframe['species'] != 1].index
    dataframe.drop(unwantedClass, inplace=True)
    # Remove the unwanted features
    dataframe = dataframe[['species', feature1, feature2]]  # remove unwanted columns

    # Split the two classes into two separate data frames
    class1DataFrame = dataframe.loc[dataframe['species'].isin([-1])]
    class1DataFrame = shuffle(class1DataFrame)
    class2DataFrame = dataframe.loc[dataframe['species'].isin([1])]
    class2DataFrame = shuffle(class2DataFrame)

    # Split data frames into train test data frames
    class1train, class1test = train_test_split(class1DataFrame, test_size=0.4)
    class2train, class2test = train_test_split(class2DataFrame, test_size=0.4)
    return class1train, class1test, class2train, class2test


def train(trainSet, weightMatrix, feature1, feature2, bias, etaValue, epochs):
    weightMatrix = weightMatrix.transpose()
    for x in range(epochs):
        for i in trainSet.index:
            # Select the wanted row from the dataframe
            selectedFeatures = [[bias, trainSet[feature1][i], trainSet[feature2][i]]]
            actualClass = trainSet['species'][i]

            # Multiply the features by weight matrix to get the predicted class
            yi = np.dot(np.array(selectedFeatures), weightMatrix.T)
            predictedClass = signum(yi)
            # Calculate error
            if predictedClass != actualClass:
                error = actualClass - predictedClass
                arr = np.asarray(selectedFeatures)
                weightMatrix = ((etaValue * error) * arr) + weightMatrix

    return weightMatrix


def signum(yi):
    return 1 if yi >= 0 else -1


def test(weightMatrix, testSet, feature1, feature2, bias):
    truePos = trueNeg = falsePos = falseNeg = 0

    for i in testSet.index:
        # Select the wanted row from the dataframe
        selectedFeatures = [[bias, testSet[feature1][i], testSet[feature2][i]]]
        actualClass = testSet['species'][i]
        # Multiply the features by weight matrix to get the predicted class
        yi = np.dot(selectedFeatures, weightMatrix.T)
        predictedClass = signum(yi)

        if actualClass == 1 and predictedClass == 1:
            truePos = truePos + 1
        elif actualClass == 1 and predictedClass == -1:
            falseNeg = falseNeg + 1
        elif actualClass == -1 and predictedClass == 1:
            falsePos = falsePos + 1
        elif actualClass == -1 and predictedClass == -1:
            trueNeg = trueNeg + 1

    accuracy = ((truePos + trueNeg) / len(testSet.index)) * 100
    print('accuracy: ', accuracy)
    print('True Positive: ', truePos)
    print('True Negative: ', trueNeg)
    print('False Positive: ', falsePos)
    print('False Negative: ', falseNeg)
    plotTestGraph(testSet, feature1, feature2, weightMatrix)


def plotTestGraph(testSet, xAxis, yAxis, weightMatrix):
    bias = weightMatrix[0][0]
    w1 = weightMatrix[0][1]
    w2 = weightMatrix[0][2]

    # Straight line equation to draw decision boundary
    x = np.linspace(testSet.loc[:, xAxis].min(), testSet.loc[:, xAxis].max())
    y = -(w1 * x + bias) / w2

    # Split each class in different dataframe
    class1DataFrame = testSet.loc[testSet['species'].isin([-1])]
    class2DataFrame = testSet.loc[testSet['species'].isin([1])]

    plt.figure('Graph')
    plt.cla()
    plt.scatter(class1DataFrame[xAxis], class1DataFrame[yAxis], color='red')
    plt.scatter(class2DataFrame[xAxis], class2DataFrame[yAxis], color='blue')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.plot(x, y, linestyle='solid', color='orange')
    plt.show()


if __name__ == '__main__':
    originalDataframe = pd.read_csv(r'penguins.csv')
    originalDataframe = dataNormalization(originalDataframe)

    main_window = Tk()
    main_window.title('Task One')
    main_window.geometry("512x512")

    # Select Features
    featureHeader = Label(main_window, text="Select 2 Features").pack()
    feature1Value = StringVar()
    feature1Value.set(features[0])
    feature1DropMenu = OptionMenu(main_window, feature1Value, *features).pack()
    feature2Value = StringVar()
    feature2Value.set(features[1])
    feature2DropMenu = OptionMenu(main_window, feature2Value, *features).pack()

    # Select Classes
    classHeader = Label(main_window, text='Select 2 Classes').pack()
    class1Value = StringVar()
    class1Value.set(species[0])
    class1DropMenu = OptionMenu(main_window, class1Value, *species).pack()
    class2Value = StringVar()
    class2Value.set(species[1])
    class2DropMenu = OptionMenu(main_window, class2Value, *species).pack()

    # Add Learning Rate
    learningRateHeader = Label(main_window, text="Add Learning Rate").pack()
    learningRateTextField = ttk.Entry(main_window, width=20)
    learningRateTextField.pack()

    # Add Epochs
    numberOfEpochsHeader = Label(main_window, text="Add Number Of Epochs").pack()
    numberOfEpochsTextField = ttk.Entry(main_window, width=20)
    numberOfEpochsTextField.pack()

    # Select Bias
    biasCheckBox = IntVar()
    checkbox = Checkbutton(main_window, text='Bias', variable=biasCheckBox).pack()

    # Start Classification
    button = Button(main_window, text="Start", command=getDataFromGUI).pack()

    sep = Separator(main_window, orient='horizontal')
    sep.pack(fill='x')

    # Select feature to plot graph
    plotHeader = Label(main_window, text='Select Features to compare').pack()
    xAxisValue = StringVar()
    xAxisValue.set(features[0])  # Default Value
    xAxisDropMenu = OptionMenu(main_window, xAxisValue, *features).pack()
    yAxisValue = StringVar()
    yAxisValue.set(features[1])  # Default Value
    yAxisDropMenu = OptionMenu(main_window, yAxisValue, *features).pack()
    plotGraphBtn2 = Button(main_window, text='Plot Graph',
                           command=lambda: plotGraph(originalDataframe, xAxisValue.get(), yAxisValue.get())).pack()

    main_window.mainloop()
