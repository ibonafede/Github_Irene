import common
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt

def draw_patterns_decision_boundaries(data, labels, numClass, numFigure, numRows, numColumns, numSubFigure, xRange, yRange, Z, decisionBoundaryColors, patternColors, accuracy, title, legend_colors, legend_labels):
    common.subplotting_decisionboundaries_drawing(numFigure,numRows,numColumns,numSubFigure,xRange,yRange,Z,decisionBoundaryColors)
    common.subplot_legend_drawing(numFigure,numRows,numColumns,numSubFigure, legend_colors, legend_labels,3)
    separate_class_data = common.separate_pattern_classes(data, labels, numClass)
    for i in range(numClass):
        common.subplotting_patterns(numFigure,numRows,numColumns,numSubFigure,separate_class_data[i],patternColors[i])
    
    strTitle="%s (acc: %.3f)"%(title,accuracy)
    common.subplotting_title(numFigure,numRows,numColumns,numSubFigure,strTitle)

def compute_accuracy(classifier,data,labels):
    results=classifier.predict(data)
    accuracy=sum(results==labels)

    return accuracy/labels.shape[0]

##--- MAIN ---

trFilePath=...
vaFilePath=...
teFilePath=...
featureCount=...

trData, trLabels = common.load_labeled_dataset(trFilePath,featureCount)
vaData, vaLabels = common.load_labeled_dataset(vaFilePath,featureCount)
teData, teLabels = common.load_labeled_dataset(teFilePath,featureCount)

#clf=KNeighborsClassifier(...)
#clf=SVC(kernel="linear", C=...)                                  
#clf=SVC(kernel="rbf",gamma=..., C=...)

# Addestramento classificatore
clf.fit(trData, trLabels)

# Calcolo accuratezza
training_accuracy = compute_accuracy(clf,trData,trLabels)
validation_accuracy = compute_accuracy(clf,vaData,vaLabels)
test_accuracy = compute_accuracy(clf,teData,teLabels)

if (featureCount==2):
    # Disegno dei pattern e delle aree di probabilità
    all_patterns_coordinates = np.concatenate((trData, vaData))
    xMin, xMax, yMin, yMax = common.calculate_2D_min_max(all_patterns_coordinates,10)
    decisionBoundaryMap, xRange, yRange=common.calculate_2D_decision_boundary_map(clf,xMin,xMax,yMin,yMax,2)

    plt.figure(num=1,figsize=(18, 8),dpi=96)

    colors = ["red","coral","gold", "yellowgreen", "green", "mediumaquamarine", "mediumturquoise", "cornflowerblue", "blue", "purple"]
    fadedColors=common.color_fading(colors,0.7)
    colorMap = matplotlib.colors.ListedColormap(fadedColors)

    legend_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    draw_patterns_decision_boundaries(trData, trLabels, 10, 1, 1, 3, 1, xRange, yRange, decisionBoundaryMap, colorMap, colors, training_accuracy, "Training", fadedColors, legend_labels)
    draw_patterns_decision_boundaries(vaData, vaLabels,10, 1, 1, 3, 2, xRange, yRange, decisionBoundaryMap, colorMap,colors, validation_accuracy, "Validation", fadedColors, legend_labels)
    draw_patterns_decision_boundaries(teData, teLabels,10, 1, 1, 3, 3, xRange, yRange, decisionBoundaryMap, colorMap,colors, test_accuracy, "Test", fadedColors, legend_labels)
    
    plt.show()
else:
    print("Accuratezza training set: ",training_accuracy)
    print("Accuratezza validation set: ",validation_accuracy)
    print("Accuratezza test set: ",test_accuracy)
#---