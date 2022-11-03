import random
import tkinter
from tkinter import *
import csv
from tkinter import ttk
from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# ToDO: remove first choice form second dropDownMenu
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


def plotGraph(originalDF, xAxis, yAxis):
    class1DataFrame = originalDF.loc[originalDF['species'].isin(['Adelie'])]
    class2DataFrame = originalDF.loc[originalDF['species'].isin(['Gentoo'])]
    class3DataFrame = originalDF.loc[originalDF['species'].isin(['Chinstrap'])]

    plt.figure('Graph')
    plt.cla()
    plt.scatter(class1DataFrame[xAxis], class1DataFrame[yAxis], color='red')
    plt.scatter(class2DataFrame[xAxis], class2DataFrame[yAxis], color='blue')
    plt.scatter(class3DataFrame[xAxis], class3DataFrame[yAxis], color='green')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.show()


# Remove null values from gender column and convert it to numerical values
def dataCleaning(dataFrame):
    numberOfMales = dataFrame.gender.value_counts().male
    numberOfFemales = dataFrame.gender.value_counts().female
    if numberOfMales > numberOfFemales:
        dataFrame.gender.replace({np.NAN: 'male'}, inplace=True)
    else:
        dataFrame.gender.replace({np.NAN: 'female'}, inplace=True)

    dataFrame.gender.replace({'male': 1, 'female': 0}, inplace=True)
    return dataFrame


def getDataFromGUI():
    feature1 = feature1Value.get()
    feature2 = feature2Value.get()
    class1 = class1Value.get()
    class2 = class2Value.get()
    etaValue = int(learningRateTextField.get())
    epochsValue = int(numberOfEpochsTextField.get())
    weightMatrix = np.random.rand(3, 1)
    if biasCheckBox.get() == 0:
        bias = 0
    else:
        bias = 1

    class1train, class1test, class2train, class2test = dataSplitter(class1, class2, feature1, feature2,
                                                                    originalDataframe)

    # To merge train sets together the shuffle them
    trainData = pd.concat([class1train, class2train])
    trainData = shuffle(trainData)
    testData = pd.concat([class1test, class2test])
    testData = shuffle(testData)

    weightMatrix = singleNeuron(trainData, weightMatrix, feature1, feature2, bias, etaValue, epochsValue)
    test(weightMatrix, testData, feature1, feature2, bias)


# Replace selected classes with numerical values and drop the 3rd class
# Split train and test dataframes
def dataSplitter(class1, class2, feature1, feature2, originalDF):
    dataframe = originalDF
    dataframe.species.replace({class1: -1, class2: 1}, inplace=True)
    unwantedClass = dataframe[dataframe['species'] != -1].index & dataframe[dataframe['species'] != 1].index
    dataframe.drop(unwantedClass, inplace=True)
    dataframe = dataframe[['species', feature1, feature2]]  # remove unwanted columns
    class1DataFrame = dataframe.loc[dataframe['species'].isin([-1])]
    class1DataFrame = shuffle(class1DataFrame)
    class2DataFrame = dataframe.loc[dataframe['species'].isin([1])]
    class2DataFrame = shuffle(class2DataFrame)
    # print(class1DataFrame)
    class1train, class1test = train_test_split(class1DataFrame, test_size=0.4)
    class2train, class2test = train_test_split(class2DataFrame, test_size=0.4)
    return class1train, class1test, class2train, class2test


def singleNeuron(trainSet, weightMatrix, feature1, feature2, bias, etaValue, epochs):
    weightMatrix = weightMatrix.transpose()
    for x in range(epochs):
        for i in trainSet.index:
            selectedRow = [[bias, trainSet[feature1][i], trainSet[feature2][i]]]
            selectedClass = trainSet['species'][i]
            yi = np.dot(np.array(selectedRow), weightMatrix.T)
            # print("\nFirst matrix {}".format(selectedRow))
            # print("Second matrix {}".format(weightMatrix))
            # print("Result: {}".format(yi))
            result = signum(yi)
            if result != selectedClass:
                loss = selectedClass - result
                arr = np.asarray(selectedRow)
                weightMatrix = (etaValue * loss) * arr + weightMatrix
    return weightMatrix


def signum(yi):
    return 1 if yi >= 0 else -1


def test(weightMatrix, testSet, feature1, feature2, bias):
    print(testSet)
    resultDF = pd.DataFrame(columns=['Actual Class', 'Predicted Class', 'Result'])
    totalTrue = 0
    totalFalse = 0
    for i in testSet.index:
        testRow = [[bias, testSet[feature1][i], testSet[feature2][i]]]
        actualResult = testSet['species'][i]
        yPredicted = np.dot(testRow, weightMatrix.T)
        predictedResult = signum(yPredicted)
        if actualResult == predictedResult:
            resultDF.loc[len(resultDF.index)] = [actualResult, predictedResult, True]
            totalTrue = totalTrue + 1
        else:
            resultDF.loc[len(resultDF.index)] = [actualResult, predictedResult, False]
            totalFalse = totalFalse + 1
    print(resultDF)
    accuracy = (totalTrue / (totalTrue + totalFalse)) * 100
    print('accuracy: {}'.format(accuracy))
    plotTestGraph(testSet, feature1, feature2)


def plotTestGraph(testSet, xAxis, yAxis):
    class1DataFrame = testSet.loc[testSet['species'].isin([-1])]
    class2DataFrame = testSet.loc[testSet['species'].isin([1])]

    plt.figure('Graph')
    plt.cla()
    plt.scatter(class1DataFrame[xAxis], class1DataFrame[yAxis], color='red')
    plt.scatter(class2DataFrame[xAxis], class2DataFrame[yAxis], color='blue')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.show()


if __name__ == '__main__':
    originalDataframe = pd.read_csv(r'penguins.csv')
    originalDataframe = dataCleaning(originalDataframe)

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
