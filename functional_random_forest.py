import sys
import joblib
import math
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                                                 
                                                                          
class FunctionalRandomForest:
    def __init__(self,estimator_type,n_estimators=100,class_weight='balanced',
                 balanced_bootstrap=True,theta=0.01,random_state=None,
                 n_jobs=-1,low_memory=True):
            if not(estimator_type=='regressor' or estimator_type=='classifier'):
                print('Illegal estimator_type, must be classifier or regressor')
                sys.exit() 
            self.estimator_type = estimator_type 
            self.n_estimators = n_estimators
            self.class_weight = class_weight
            self.balanced_bootstrap = balanced_bootstrap
            self.theta = theta
            self.random_state = random_state
            self.n_jobs = n_jobs
            self.low_memory = low_memory
            self.criterion = None
            self.min_samples_leaf = None
            self.tree_list = []
            self.ib_samples = None
            self.oob_samples = None
            self.parameters = {}

    
    def set_parameters(self,parameters):
        self.min_samples_leaf = parameters['min_samples_leaf']
        self.parameters['min_samples_leaf'] = self.min_samples_leaf
        self.criterion = parameters['criterion']
        self.parameters['criterion'] = self.criterion
        self.theta = parameters['theta']
        self.parameters['theta'] = self.theta


    def predict(self,X):
        prediction = None
        bagged_pred_Y = []
        for tree in self.tree_list:
            bagged_pred_Y.append(tree.predict(X))  
        bagged_pred_Y = np.array(bagged_pred_Y)
        if self.estimator_type=='classifier':
            major_vote_Y = [] 
            for j in range(X.shape[0]):         
                unique_labels = np.unique(bagged_pred_Y[:,j])
                counts_dict = {label:0 for label in unique_labels}
                for label in bagged_pred_Y[:,j]:
                        counts_dict[label] += 1
                max_count = 0
                max_count_label = None    
                for label in unique_labels:
                    if counts_dict[label] >= max_count:
                        max_count = counts_dict[label]
                        max_count_label = label
                major_vote_Y.append(max_count_label)           
            prediction = np.array(major_vote_Y)    
        else:
            mean_Y = [] 
            for j in range(X.shape[0]):
                mean_Y.append(np.mean(bagged_pred_Y[:,j]))
            prediction = np.array(mean_Y) 
        return prediction
        

    def score(self,X,Y):
        score = None
        pred_Y = self.predict(X)
        if self.estimator_type=='classifier':
            score = balanced_accuracy_score(Y,pred_Y)
        else:
            score = np.mean((pred_Y-Y)**2)
        return score        
                 
                 
    def fit(self,X,Y):
        if self.estimator_type=='classifier':
            if self.balanced_bootstrap:
                self._bootstrap_balanced(Y)
            else:
                self._bootstrap(Y.shape[0])   
        else:
            self._bootstrap(Y.shape[0])
        for i in range(self.n_estimators):
            if self.random_state:
                random_state = self.random_state+i
            else:
                random_state = i 
            estimator = FunctionalTree(self.estimator_type,
                                       random_state=random_state,
                                       criterion=self.criterion,
                                    min_samples_leaf=self.min_samples_leaf,
                                       theta=self.theta,
                                       class_weight=self.class_weight)
            self.tree_list.append(estimator)
        if self.low_memory:    
            for tree in self.tree_list:
                tree.fit(X[self.ib_samples[i]],Y[self.ib_samples[i]])    
        else:
            self.tree_list = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self.tree_list[i].fit)(X[self.ib_samples[i]],Y[self.ib_samples[i]]) for i in range(self.n_estimators))
        return self                                
        

    def _bootstrap(self,n_samples):
        np.random.seed(self.random_state)
        self.ib_samples = np.random.choice(n_samples,(self.n_estimators,n_samples))
        all_indexes = [i for i in range(n_samples)]
        self.oob_samples = []
        for j in range(self.ib_samples.shape[0]):
            self.oob_samples.append([all_indexes[i] for i in range(len(all_indexes)) if not(all_indexes[i] in self.ib_samples[j])])
        
        
    def _bootstrap_balanced(self,Y):
        n_samples = Y.shape[0]
        unique_classes = np.unique(Y)
        n_classes = len(unique_classes)
        n_samples_balanced_classes = int(math.floor(n_samples/n_classes))
        indexes_by_class = {label:[i for i in range(len(Y)) if Y[i]==label] for label in unique_classes}
        ib_samples_by_class = {label:None for label in unique_classes}
        for key,val in indexes_by_class.items():
            np.random.seed(self.random_state)
            ib_samples_by_class[key] = np.random.choice(val,(self.n_estimators,n_samples_balanced_classes))
        self.ib_samples = []
        for i in range(self.n_estimators):
            tree_ib_samples = []
            for key,val in ib_samples_by_class.items():
                tree_ib_samples.extend(val[i])
            self.ib_samples.append(tree_ib_samples)        
        all_indexes = [i for i in range(n_samples)]
        self.oob_samples = []
        for i in range(len(self.ib_samples)):
            ib = self.ib_samples[i]
            self.oob_samples.append([all_indexes[i] for i in range(len(all_indexes)) if not(all_indexes[i] in ib)])
        
                
class FunctionalTree:
    def __init__(self,estimator_type,random_state,criterion=None,
                               min_samples_leaf=1,overlapping=False,theta=0.01,
                                                      class_weight='balanced'):
            if estimator_type=='classifier':
                self.criterion = 'gini'
                if criterion:
                    self.criterion = criterion
                self.tree = DecisionTreeClassifier(criterion=self.criterion,
                                          min_samples_leaf=min_samples_leaf,
                                                     class_weight=class_weight) 
            else:
                self.criterion = 'mse'
                if criterion:
                    self.criterion = criterion
                self.tree = DecisionTreeRegressor(criterion=self.criterion,
                                             min_samples_leaf=min_samples_leaf)
            self.random_state = random_state
            self.criterion = criterion
            self.min_samples_leaf = min_samples_leaf
            self.class_weight = class_weight
            self.theta = theta
            self.transform_matrix = []
            

    def predict(self,X):
        transformed_X = self._transform(X)
        pred = self.tree.predict(transformed_X)
        return pred
                        
                 
    def fit(self,X,Y):
        np.random.seed(self.random_state)
        p = X.shape[1]
        scale = 1/self.theta
        begin = 0
        end = 0
        while begin<p:
            new_row = np.zeros(p)
            offset = int(math.floor(np.random.exponential(scale)))
            if offset==0:
                offset=1
            end = begin + offset 
            if end>p:
                end=p
            new_row[begin:end] = 1/(end-begin) 
            self.transform_matrix.append(new_row)
            begin = end
        self.transform_matrix = np.array(self.transform_matrix).transpose()
        transformed_X = self._transform(X)
        self.tree.fit(transformed_X,Y) 
        return self
    
    
    def _transform(self,X):
        transformed_X = []
        for x in X:
            transformed_x = np.matmul(x,self.transform_matrix)
            transformed_X.append(transformed_x)
        return np.array(transformed_X)
        
        
        
        
                   
