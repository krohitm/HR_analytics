import itertools
import numpy as np
import pandas as pd
import scipy.special as special
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

def plot_confusion_matrix(Feature_covariance, features, normalize=False, 
  title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the feature_covariance.
    Normalization can be applied by setting `normalize=True`.
    """
    #cm = confusion_matrix(Y_test, Y_pred);
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    
    tick_marks = np.arange(len(features))
    plt.xticks(tick_marks, features, rotation=45)
    plt.yticks(tick_marks, features)

    cm = Feature_covariance
    if normalize:
        cm = np.around(np.transpose((np.transpose(cm.astype('float')) 
          / cm.sum(axis=1))) * 100, decimals = 2)
        print("Percentage confusion matrix")
    else:
        print('Feature_covariance')
    print cm

    c = np.array([[0] * cm.shape[0]] * cm.shape[1])    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      if (i - j) == 0:
        c[i][j] = 1
      else:
        c[i][j] = -1

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text( j, i, cm[i, j], horizontalalignment="center",
          color="white" if c[i, j] == 1 else "black")

    plt.imshow(c, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

def analyse_features(data):
  hist, edges = np.histogram(data['number_project'], density = True)
  x = np.linspace(0, 1, 1000)
  p1 = figure(title = "Distribution of sales type", 
    background_fill_color = "#E8DDCB")
  p1.quad(top = hist, bottom = 0, left = edges[:-1], right = edges[1:],
    fill_color = "#036564", line_color = "#033649")
  
  number_project = np.array(data['number_project'])
  mu = np.mean(number_project)
  sigma = np.mean(number_project)
  pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
  cdf = (1 + special.erf((x-mu)/np.sqrt(2*sigma**2)))/2
  p1.line(x, pdf, line_color="#D95B43", line_width=8, alpha=0.7, legend="PDF")
  p1.line(x, cdf, line_color="black", line_width=2, alpha=0.7, legend="CDF")

  p1.legend.location = "top_left"
  p1.xaxis.axis_label = 'Number of Projects'
  p1.yaxis.axis_label = 'Pr(Number of Projects)'

  #output_file('histogram.html', title = "histogram.py example")
  show(gridplot(p1, ncols = 2, plot_width = 400, plot_height = 400))


# Compute confusion matrix
def main(data, Feature_covariance):
  np.set_printoptions(precision=2)
  data = pd.DataFrame(data)
  features = data.columns
  #Feature_covariance = data.cov()
  #print Feature_covariance
  Feature_covariance = np.array(Feature_covariance)

  analyse_features(data)
  #plot_confusion_matrix(Feature_covariance, features)
