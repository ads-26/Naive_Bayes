# -*- coding: utf-8 -*-

import sklearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve, auc

led = pd.read_csv("led_display.csv")


target = 'Class'
features = led.columns[led.columns != target]
classes = led[target].unique()
test = led.sample(frac=0.33)
train = led.drop(test.index)


probs = {}
probcl = {}
for x in classes:
    traincl = train[train[target]==x][features]
    clsp = {}
    tot = len(traincl)
    for col in traincl.columns:
        colp = {}
        for val,cnt in traincl[col].value_counts().iteritems():
            pr = cnt/tot
            colp[val] = pr
        clsp[col] = colp
    probs[x] = clsp
    probcl[x] = len(traincl)/len(train)
def probabs(x):
    
    probab = {}
    for cl in classes:
        pr = probcl[cl]
        for col,val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab
def classify(x):
    probab = probabs(x)
    mx = 0
    mxcl = ''
    for cl,pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl

#Train data
b = []
for i in train.index:
    b.append(classify(train.loc[i,features]) == train.loc[i,target])
print("Accuracy (train) :", (sum(b)/len(train))*100)

#Test data
y_pred=[]
y_test=[]
b = []

for i in test.index:
    y_pred.append(classify(test.loc[i,features]))
    y_test.append(test.loc[i,target])
    b.append(classify(test.loc[i,features]) == test.loc[i,target])

print("Accuracy (test) :", (sum(b)/len(test))*100)

s=[]
fn=[]
tn=[]
acc = accuracy_score(y_test,y_pred)
print(str(acc))
cf=(confusion_matrix(y_test,y_pred))
print(cf)


for i in range(len(cf)):
    f=1
    for j in range(len(cf[0])):
        f+=cf[i][j]
    fn.append(f)
for i in range(len(cf)):
    f=1
    for j in range(len(cf[0])):
        f+=cf[j][i]
    tn.append(f)
    

pre=0
rec=0
i=0
for i in range(len(cf)):
    pre+=cf[i][i]/fn[i]
    rec+=cf[i][i]/tn[i]
    
precision=pre*100/len(cf)
recall=rec*100/len(cf)


print('Average precision score: {0:0.2f}'.format(precision))
print('Average recall score: {0:0.2f}'.format(recall))
#avg_recall=recall(y_test,y_pred)


#y_pred_proba = clf.predict_proba(X_test)[:,1]
#print(y_pred_proba)
#fpr,tpr,threshold = roc_curve(y_test,y_pred)

#plt.plot(fpr,tpr,color='y')

f1 = (2*precision*recall)/(precision+recall)
print('Average f1 score: {0:0.2f}'.format(f1))

        
print("***********************************************")
#print()

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 10

y_test1 = np.zeros((500,10))
y_pred1 = np.zeros((500,10))

for i in range(len(y_test)) :
    y_test1[i][y_test[i]-1]=1
 
for i in range(len(y_pred)) :
    y_pred1[i][y_pred[i]-1]=1
    
    
for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test1[:,i], y_pred1[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_pred1.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
lw = 2
plt.plot(fpr[2], tpr[2], color="red",
         lw=lw, label='(area = %0.2f)' % roc_auc[2])


plt.legend(loc="lower right")
