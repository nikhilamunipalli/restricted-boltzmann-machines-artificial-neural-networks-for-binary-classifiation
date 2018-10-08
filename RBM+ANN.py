#importing libraries
import numpy as np
import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

#importing datasets
dataset = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test = test.iloc[:,[1,3,4,5,6,8,10]]
X = dataset.iloc[:,[2,4,5,6,7,9,11]]
y = dataset.iloc[:,1]

y = y[X['Embarked'].notnull()].values
X = X[X['Embarked'].notnull()]

train_length = len(X)
test_length = len(test)

X = X.append(test)
X = X.values

#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[2]])
X[:,[2]]= imputer.transform(X[:,[2]])

#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,[5]])
X[:,[5]] = imputer.transform(X[:,[5]])

#categorical data handling
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()

l=[0,1,3,4,6]

for j in l :
     X[:,j] = labelencoder.fit_transform(X[:,j].astype(str))

X = X.astype(float)

onc = OneHotEncoder(categorical_features = [0])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

onc = OneHotEncoder(categorical_features = [4])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

onc = OneHotEncoder(categorical_features = [10])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

onc = OneHotEncoder(categorical_features = [18])
onc.fit(X)
X = onc.fit_transform(X).toarray()
X= X[:,1:]

#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#converting to torch tensors
X = torch.FloatTensor(X)


#class RBM
class RBM():
    def __init__(self,nh,nv):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,v0,vk,ph0,phk):
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.a += torch.sum((ph0-phk),0)
        self.b += torch.sum((v0-vk),0)

#instantiation
nb_x = len(X)
nv = len(X[0])
nh = 50
batch_size = 10
rbm = RBM(nh,nv)
train = []
nb_epoch = 50

#training the rbm
for epoch in range(nb_epoch+1):
    train_loss = 0
    s = 0
    for id_x in range(0,nb_x,batch_size):
        vk = X[id_x : id_x + batch_size]
        v0 = X[id_x : id_x + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(50):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        p = phk.numpy()
        
        if epoch== nb_epoch and batch_size == 10:
           train.append(p)
           
        train_loss += torch.mean(torch.abs(v0 - vk))
        s+=1;
    print('epoch :'+str(epoch)+' loss : '+str(train_loss/s))

#training set conversion
train = np.vstack(train)

#seperation of test and train
X_test1 = train[train_length:,:]
X = train[:train_length,:]

#traintestsplit
'''from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)'''

#building the ANN model
def build_classifier():
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = nh , output_dim = 50, init = 'uniform', activation = 'relu'))
    
    #internal layers
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
    
    #output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = 'rmsprop' , metrics =['accuracy'],loss = 'binary_crossentropy' )
    return classifier

#kfoldcrossvalidaton
from sklearn.model_selection import cross_val_score as cs
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 5, nb_epoch = 100)
accuracies = cs(classifier,X = X, y = y, cv = 10, n_jobs = -1)
accuracy = accuracies.mean()

#fitting the model
classifier.fit(X,y)

#predicting results
y_pred = classifier.predict(X_test1)
 

















