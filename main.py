import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visual as visualize
#from sklearn import preprocessing


def get_data(filename):
	data = pd.read_csv(filename)
	labels = data[['left']]
	data = data.drop(['left'],1)
	#data = preprocessing.scale(data)
	Feature_covariance = data.cov()
	#print max(data['satisfaction_level'])
	#print data['number_project'].unique()
	#print data['salary'].unique()
	visualize.main(data, Feature_covariance)


def main():
	data = get_data('/Users/GodSpeed/Documents/CodeWork/HR_analytics/HR_comma_sep.csv')

main()