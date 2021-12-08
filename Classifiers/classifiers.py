import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model, Sequential

from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# AdaBoostClassifier
# AdaBoostRegressor
# ExtraTreesClassifier
# ExtraTreesRegressor
# GradientBoostingClassifer
# GradientBoostingRegressor
# RandomForestClassifier
# RandomForestRegressor
# RandomTreesEmbedding

# classifiers
class Classifiers:
    def __init__(self, X_train = None, y_train = None):
        self.X_train = X_train
        self.y_train = y_train

    def RandomForestClassifier(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        model = RandomForestClassifier(random_state=42).fit(self.X_train, self.y_train)
        return model

    def BinaryRelevance(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        model = BinaryRelevance(GaussianNB()).fit(self.X_train, self.y_train)
        return model
        
    def ClassifierChain(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # model = ClassifierChain(RandomForestClassifier(n_estimators=100)).fit(self.X_train, self.y_train)
        model = ClassifierChain(KNeighborsClassifier()).fit(self.X_train, self.y_train)
        return model

    def LabelPowerset(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        model = LabelPowerset(GaussianNB()).fit(self.X_train, self.y_train)
        return model
    
    def MultiOutputClassifier(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        model = MultiOutputClassifier(KNeighborsClassifier()).fit(self.X_train, self.y_train)
        return model
    
    def OneVsRestClassifier(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        model = OneVsRestClassifier(SVC()).fit(self.X_train, self.y_train)
        return model
    
        # define and fit the model
    def cnnModel_one_hidden(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # define model
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(7,1)))
        model.add(MaxPooling1D(pool_size=1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        # model.add(MaxPooling1D(pool_size=1))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(93, activation='sigmoid'))
        print(model.summary())
        model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs = 512, batch_size=128, verbose = 0 )
        return model

    def cnnModel_two_hidden(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # define model
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(7,1)))
        model.add(MaxPooling1D(pool_size=1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        # model.add(MaxPooling1D(pool_size=1))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(93, activation='sigmoid'))
        print(model.summary())
        model.compile(optimizer='rmsprop', loss = 'binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs = 512, batch_size=128, verbose = 0 )
        return model
    
    # http://scikit.ml/api/skmultilearn.ensemble.partition.html#skmultilearn.ensemble.LabelSpacePartitioningClassifier
    # http://scikit.ml/api/skmultilearn.html
    # https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff