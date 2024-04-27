# Import Needed Libraries
import pandas
import sklearn
import matplotlib
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle


df = pd.read_csv('KDD Test+_numerical.csv')

dataset = df.values
cols = [i for i in df.columns.to_list() if i != "class"]

XX = df[cols]
Y = df['class']

# define minmax scaler
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(XX)

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['LGB'] = LGBMClassifier()
	#models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['MPL'] = MLPClassifier()
	models['bayes'] = GaussianNB()
	models['RF'] = RandomForestClassifier()
	return models


# define dataset
X = X
Y = dataset[:,41]
# define the base models
level0 = list()
level0.append(('SVM', SVC()))
level0.append(('MLP', MLPClassifier()))
level0.append(('LR', LogisticRegression()))
level0.append(('DT', DecisionTreeClassifier()))
level0.append(('LGB', LGBMClassifier()))
level0.append(('NB', GaussianNB()))
level0.append(('RF', RandomForestClassifier()))
level0.append
# define meta learner model
level1 = LGBMClassifier()
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X, Y)
# make a prediction for one example
yhat = model.predict(X)


pickle_out = open("new_model.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

print("Saved Model")
print(f"Prediction {model.predict(X[0:7])}")