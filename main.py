
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import numpy as np
import math
import pickle
# %matplotlib inline

class_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
no_of_classes = 3

def prepare_dataset(data_path):
    df = pd.read_csv(data_path,header=None)

    feature_dict = {i:label for i,label in zip(
                    range(4),
                      ('sepal length in cm',
                      'sepal width in cm',
                      'petal length in cm',
                      'petal width in cm', ))}


    df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
    df.dropna(how="all", inplace=True) # to drop the empty line at file-end

    print df.head()
    for i in range(3):
        rows = (df['class label'] == class_dict[i])
        for j in range(4):
            col_max = df.iloc[:,j][rows].max()
            col_min = df.iloc[:,j][rows].min()
            df.iloc[:,j][rows] = (df.iloc[:,j][rows] - col_min) / (col_max - col_min)

    df.to_pickle(r'C:\Users\weineja\Documents\Machine Learning\ML Data\LDA\dataset.pkl')
    return df

def plot_histograms(df):
    feature_dict = {i: label for i, label in zip(
        range(4),
        ('sepal length in cm',
         'sepal width in cm',
         'petal length in cm',
         'petal width in cm',))}
    #assigning x and y matrices
    X = df.iloc[:,0:4].values
    y = df['class label'].values

    #syntax for implementing an encoder
    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1

    # to reference to encoder values later
    encoder_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    #axes.ravel() describes the position of the axes of each figure
    #go through each field to determine the bin sizes
    for ax, cnt in zip(axes.ravel(), range(4)):
        # set bin sizes
        min_b = math.floor(np.min(X[:, cnt]))
        max_b = math.ceil(np.max(X[:, cnt]))
        #can use array_like object for bins
        bins = np.linspace(min_b, max_b, 25)

        for key,colour in zip(encoder_dict.keys(),['red','blue','green']):
            rows = (y == key)
            #accessing a certain column (cnt) from multiple lists(rows) within a list
            ax.hist(X[rows,cnt],bins = bins,color = colour,alpha = 0.8)

            ax.set_title('{}'.format(feature_dict[cnt]))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
    fig.tight_layout()
    plt.show()

def calculate_class_means(df):
    means = []
    # print dataset['class label'].unique()
    for i in range(3):
        rows = (df['class label'] == class_dict[i])
        means.append([float(df.iloc[:,0][rows].mean()),
                      float(df.iloc[:,1][rows].mean()),
                      float(df.iloc[:,2][rows].mean()),
                      float(df.iloc[:,3][rows].mean()),

                      class_dict[i]])

    return means

def in_between_scatter(df,means):
    X = df.iloc[:,0:4].values
    y = df['class label'].values
    print type(X)
    for i in range(no_of_classes):
        row_indices = (y == class_dict[i])
        scatter_plot = np.zeros((4,4))
        rows = X[row_indices,:]
        for row in rows:
            row_4_1 = row.reshape(4,1)
            mean_4_1 =  np.array(means[i][0:4]).reshape(4,1)
            scatter_plot += (row_4_1 - mean_4_1).dot(row - mean_4_1.T)
    return scatter_plot
def main():

    # dataset = prepare_dataset(r'C:/Users/weineja/Documents/KNN/iris.data.txt')
    try:
        dataset = pd.read_pickle(r'C:\Users\weineja\Documents\Machine Learning\ML Data\LDA\dataset.pkl')
    except IOError:
        print 'incorrect location or pickle object does not exist'
        exit()

    # plot_histograms(dataset)
    means = calculate_class_means(dataset)
    print means[0][0:4]
    # X[rows, cnt]
    # print means[:,1]
    # need to solve eigenvalues to find
    # calculate
    in_between_scatter(dataset,means)
    # print means

if __name__ == "__main__":
    main()