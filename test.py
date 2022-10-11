import numpy as np
import matplotlib.pylab as plt
import tests
import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score


bike_sharing = data.BikeSharing()
bike_sharing.y_train[np.where(bike_sharing.y_train < 2000)[0]] = 0
bike_sharing.y_train[np.where((bike_sharing.y_train >= 2000) & (bike_sharing.y_train < 4000))[0]] = 1
bike_sharing.y_train[np.where((bike_sharing.y_train >= 4000) & (bike_sharing.y_train < 6000))[0]] = 2
bike_sharing.y_train[np.where(bike_sharing.y_train >= 6000)[0]] = 3

bike_sharing.y_test[np.where(bike_sharing.y_test < 2000)[0]] = 0
bike_sharing.y_test[np.where((bike_sharing.y_test >= 2000) & (bike_sharing.y_test < 4000))[0]] = 1
bike_sharing.y_test[np.where((bike_sharing.y_test >= 4000) & (bike_sharing.y_test < 6000))[0]] = 2
bike_sharing.y_test[np.where(bike_sharing.y_test >= 6000)[0]] = 3

bg = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1), n_estimators=10, max_samples=0.9, random_state=42)
bg.fit(bike_sharing.X_train, bike_sharing.y_train)
bg_y_pred = bg.predict(bike_sharing.X_test)
print("bg accuracy: ", bg.score(bike_sharing.X_test, bike_sharing.y_test))
print("bg precision: ", precision_score(bike_sharing.y_test, bg_y_pred, average='weighted', zero_division=1))

rf = RandomForestClassifier(max_depth=3, min_samples_leaf=0.1, n_estimators=200, max_samples=0.8, max_features=0.8, random_state=42)
rf.fit(bike_sharing.X_train, bike_sharing.y_train)
rf_y_pred = rf.predict(bike_sharing.X_test)
print("rf accuracy: ", rf.score(bike_sharing.X_test, bike_sharing.y_test))
print("rf precision: ", precision_score(bike_sharing.y_test, rf_y_pred, average='weighted', zero_division=1))

ad = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1), n_estimators=40)
ad.fit(bike_sharing.X_train, bike_sharing.y_train)
ad_y_pred = ad.predict(bike_sharing.X_test)
print("ad accuracy: ", ad.score(bike_sharing.X_test, bike_sharing.y_test))
print("ad precision: ", precision_score(bike_sharing.y_test, ad_y_pred, average='weighted', zero_division=1))


# bg accuracy:  0.54421768707483
# bg precision:  0.6163631768132063
# rf accuracy:  0.54421768707483
# rf precision:  0.6072357311719014
# ad accuracy:  0.5510204081632653
# ad precision:  0.6044868296969137