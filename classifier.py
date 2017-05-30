import pandas as pd
from sklearn import svm
import numpy as np
import pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

#if __name__ == '__main__':        
    
RUN = 1
init_col = 0
NUM_COLORS = 20

if RUN:    
    df_kitchen_top_colors = pd.read_pickle('kitchen_top_colors.pkl')
    df_bedroom_top_colors = pd.read_pickle('bedroom_top_colors.pkl')
    df_bathroom_top_colors = pd.read_pickle('bathroom_top_colors.pkl')
    df_living_top_colors = pd.read_pickle('living_top_colors.pkl')
    Np = len(df_living_top_colors)
    df = pd.concat([df_kitchen_top_colors.loc[:,'color%d_rgb' % init_col:'color%d_rgb' % (NUM_COLORS-1)], 
                            df_bedroom_top_colors.loc[:,'color%d_rgb' % init_col:'color%d_rgb' % (NUM_COLORS-1)], 
                            df_bathroom_top_colors.loc[:,'color%d_rgb' % init_col:'color%d_rgb' % (NUM_COLORS-1)],
                            df_living_top_colors.loc[:,'color%d_rgb' % init_col:'color%d_rgb' % (NUM_COLORS-1)]])
    np_df = np.array(df)
    
    X = np.zeros((4*Np,3*(NUM_COLORS-init_col)))
    
    for c in range(0,NUM_COLORS-init_col,3):
        for num, rgb in enumerate(np_df[:,c]):
            X[num,c], X[num,c+1], X[num,c+2] = rgb
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    y = np.zeros(len(X))
    classes = [0, 1, 2, 3]
    for c in classes:
        y[c*Np:(c+1)*Np] = c
    
    #X = SelectKBest(chi2, k=3).fit_transform(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    #clf = svm.SVC(C=1., kernel = 'rbf')
    #clf = SGDClassifier(loss="hinge", penalty="l2")
    #clf = NearestCentroid()
    #clf = tree.DecisionTreeClassifier()
    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    #                                 max_depth=1, random_state=0)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(9, 3), random_state=1)
    #clf = LogisticRegression()
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    #X = np.array(df_kitchen_pixels)
    scores = cross_val_score(clf, X, y, cv=5, scoring='r2')
    print("R^2: %.3f (%.3f)") % (scores.mean(), scores.std())
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    matrix = metrics.confusion_matrix(y_test, pred)
    print(matrix)
    plt.matshow(matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    report = metrics.classification_report(y_test, pred)
    print(report)
else:
    print(clf.predict(X[544,:].reshape(1, -1)))
    w = clf.coef_[0]
    print(w)
    a = -w[0] / w[1]
    xx = np.linspace(-3,1.5)
    yy = a * xx - clf.intercept_[0] / w[1]
    
    plt.close('all')
    h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
    
    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.legend()
#    df_kitchen_top_colors = pd.read_pickle('kitchen_top_colors.pkl')
#    df_bedroom_top_colors = pd.read_pickle('bedroom_top_colors.pkl')
#    df_bathroom_top_colors = pd.read_pickle('bathroom_top_colors.pkl')
#    df_living_top_colors = pd.read_pickle('living_top_colors.pkl')