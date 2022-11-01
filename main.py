import random
from tkinter import *
import csv
from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def plotGraph(xAxis, yAxis):
    plt.figure('Graph')
    plt.cla()
    plt.scatter(class1[xAxis], class1[yAxis], color='red')
    plt.scatter(class2[xAxis], class2[yAxis], color='blue')
    plt.scatter(class3[xAxis], class3[yAxis], color='green')
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.show()


def firstNeuron():
    # label.config(text=feature1Value.get())
    # print(feature1Value.get())
    # print(feature2Value.get())
    # print(class1Value.get())
    # print(class2Value.get())
    # if biasCheckBox.get() == 0:
    #     print(-1)
    # else:
    #     print(1)
    # print('Learning Rate {}'.format(learningRateTextField.get()))
    # print('Number of Epochs {}'.format(numberOfEpochsTextField.get()))
    return 1


def dataCleaning(dataFrame):
    numberOfMales = dataframe.gender.value_counts().male
    numberOfFemales = dataframe.gender.value_counts().female
    na = dataframe.gender.value_counts()
    print(numberOfFemales)
    print(numberOfMales)
    print(na)
    if numberOfMales > numberOfFemales:
        dataframe.gender.replace({np.NAN: 'male'}, inplace=True)
    else:
        dataframe.gender.replace({np.NAN: 'female'}, inplace=True)

    dataframe.gender.replace({'male': 1, 'female': 0}, inplace=True)
    return dataFrame


def buildGUI():
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
    learningRateTextField = Entry(main_window, width=20).pack()

    # Add Epochs
    numberOfEpochsHeader = Label(main_window, text="Add Number Of Epochs").pack()
    numberOfEpochsTextField = Entry(main_window, width=20).pack()

    # Select Bias
    biasCheckBox = IntVar()
    checkbox = Checkbutton(main_window, text='Bias', variable=biasCheckBox).pack()

    # Start Classification
    button = Button(main_window, text="Start", command=firstNeuron).pack()

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
                           command=lambda: plotGraph(xAxisValue.get(), yAxisValue.get())).pack()

    main_window.mainloop()


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

if __name__ == '__main__':
    dataframe = pd.read_csv(r'penguins.csv')
    dataframe = dataCleaning(dataframe)
    class1 = dataframe.loc[dataframe['species'].isin(['Adelie'])]
    class2 = dataframe.loc[dataframe['species'].isin(['Gentoo'])]
    class3 = dataframe.loc[dataframe['species'].isin(['Chinstrap'])]
    print(class1)
    class1 = shuffle(class1)
    class2 = shuffle(class2)
    class3 = shuffle(class3)
    print(class1)
    buildGUI()
