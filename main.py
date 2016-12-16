import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as special
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.charts import Bar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
import plot_confusion_matrix as plc

def get_data():
	filename = '/Users/GodSpeed/Documents/CodeWork/HR_analytics/HR_comma_sep.csv'
	data = pd.read_csv(filename)
	data_orig = data
	labels = data[['left']]
	data = data.drop(['left'],1)
	data_orig.info()
	data_orig.describe()
	unique_sales = data_orig['sales'].unique()
	unique_salary = data_orig['salary'].unique()
	#sales_dict = {}
	for i in range (len(unique_sales)):
		data_orig['sales'].replace(to_replace = unique_sales[i],
			value = i, inplace = True)
	#	sales_dict[unique_sales[i]] = i

	#salary_dict = {}
	for i in range (len(unique_salary)):
		data_orig['salary'].replace(to_replace = unique_salary[i],
			value = i, inplace = True)
	#	salary_dict[unique_salary[i]] = i
	#print sales_dict
	#print salary_dict
	return data, data_orig, labels


def visualize_data(data, data_orig, labels):
	hist, edges = np.histogram(data['satisfaction_level'], density = False,
	 bins = 'fd')
	p1 = figure(title = "Distribution of Satisfaction Level", tools = "")
	p1.quad(top = hist, bottom = 0, left = edges[:-1], right = edges[1:],
		fill_color = "#4169E1", line_color = "#033649") 
	p1.legend.location = "top_left"
	p1.xaxis.axis_label = 'Satisfaction Level'
	p1.yaxis.axis_label = 'Number of Employees'


	hist, edges = np.histogram(data['last_evaluation'], density = False,
	 bins = 'fd')
	p2 = figure(title = "Distribution of Last Evaluation", tools = "")
	p2.quad(top = hist, bottom = 0, left = edges[:-1], right = edges[1:],
        fill_color = "#4169E1", line_color = "#033649")
	p2.legend.location = "top_left"
	p2.xaxis.axis_label = 'Last Evaluation'
	p2.yaxis.axis_label = 'Number of Employees'


	p3 = Bar(data['number_project'], 'number_project', 
        values = 'number_project', agg = 'count', 
        title = 'Distribution of Number of Projects', 
        legend = False, tools = '', color = '#4169E1')


	hist, edges = np.histogram(data['average_montly_hours'], density = False,
	 bins = 'fd')
	p4 = figure(title = "Distribution of Average Monthly Hours", tools = "")
	p4.quad(top = hist, bottom = 0, left = edges[:-1], right = edges[1:],
        fill_color = "#4169E1", line_color = "#033649")
	p4.xaxis.axis_label = 'Average Monthly Hours'
	p4.yaxis.axis_label = 'Number of Employees'


	p5 = Bar(data['time_spend_company'], 'time_spend_company', 
        values = 'time_spend_company', agg = 'count', 
        title = 'Distribution of Time Spent in Company', 
        legend = False, tools = '', color = '#4169E1')


	p6 = Bar(data['Work_accident'], 'Work_accident', values = 'Work_accident',
        agg = 'count', title = 'Distribution of Work Accidents', 
        legend = False, tools = '', color = '#4169E1')


	p7 = Bar(data['promotion_last_5years'], 'promotion_last_5years', 
        values = 'promotion_last_5years', agg = 'count', 
        title = 'Distribution of Promotion in Last 5 years', 
        legend = False, tools = '', color = '#4169E1')

	p8 = Bar(data['sales'], 'sales', values = 'sales', 
        agg = "count", title="Distribution of Job Domain", 
        legend = False, tools = "", color = '#4169E1')


	p9 = Bar(data['salary'], 'salary', values = 'salary',
        agg = "count", title = 'Distribution of Salary', 
        legend = False, tools = '', color = '#4169E1')


	p10 = Bar(labels, 'left', values = 'left',
        agg = "count", title = 'Distribution of People left', 
        legend = False, tools = '', color = '#4169E1')

	show(gridplot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
		ncols = 2, plot_width = 400, plot_height = 400))


	data_covariance = data_orig.cov()
	sns.heatmap(data_covariance, vmax=.8, square=True)
	plt.title("Covariance of Data")
	plt.show()
	data_corr = data_orig.corr()
	sns.heatmap(data_corr , vmax=.8, square=True)
	plt.title("Correlation of Data")
	plt.show()

	sns.set(color_codes=True)
	var = list(data_orig.columns)
	var.remove('left')
	sns.pairplot(data_orig, hue='left', vars = var, palette="husl")
	plt.show()


def main():
	data, data_orig, labels = get_data()
	#visualize_data(data, data_orig, labels)

	Y = np.array(data_orig[['left']])
	X = np.array(data_orig.drop(['left'],1))
	clf = KNeighborsClassifier(n_neighbors = 1)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
	scores = cross_val_score(clf, X_train, Y_train.ravel(), cv=5, 
		scoring = 'f1')
	print "mean accuracy of validation: ", scores.mean()
	clf = clf.fit(X_train, Y_train.ravel())
	Y_pred = clf.predict(X_test)
	f1 = f1_score(Y_test, Y_pred)
	print 'Test accuracy: ', f1
	plc.main(Y_test, Y_pred)

	"""with feature selection"""
	Y = np.array(data_orig[['left']])
	X = np.array(data_orig.drop([
		'left', 'number_project', 'sales', 'Work_accident'],1))
	clf = KNeighborsClassifier(n_neighbors = 1)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
	scores = cross_val_score(clf, X_train, Y_train.ravel(), cv=5, scoring = 'f1')
	print "mean accuracy of validation: ", scores.mean()
	clf = clf.fit(X_train, Y_train.ravel())
	Y_pred = clf.predict(X_test)
	f1 = f1_score(Y_test, Y_pred)
	print 'Test accuracy: ', f1
	plc.main(Y_test, Y_pred)

main()