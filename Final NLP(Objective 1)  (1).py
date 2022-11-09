#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[2]:


heart = pd.read_csv('heart-1.csv')


# In[3]:


#Describing the Data set 

heart.head()


# In[4]:


heart.describe()


# In[5]:


heart.info()


# In[6]:


heart.shape


# In[7]:


heart.columns.values


# In[8]:


heart.nunique()


# In[9]:


age_plot= sns.countplot(data=heart, x="age")
plt.title('The mean is around 54')
plt.suptitle('Age histogram', fontweight = 'normal')

for ind, label in enumerate(age_plot.get_xticklabels()):
    if ind % 5 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)


# In[10]:


target = heart['target'].map({'Presence':1, 'Absence':0})
target_plot = sns.countplot(data = heart, x ="target")

heart['target'].value_counts()


# In[11]:


Labels = ['Men', 'Women']
order = heart['sex'].value_counts().index

plt.figure(figsize=(12,6))
plt.suptitle("Gender")

plt.subplot(1,2,1)
plt.title('Pie chart')
plt.pie(heart['sex'].value_counts(), labels = Labels, textprops={'fontsize':12})


# In[12]:


men_count = heart[heart['sex'] == 1]
print(men_count.shape)

women_count = heart[heart['sex'] == 0]
print(women_count.shape)


# In[13]:


#There is an imbalance between male and female and People with and without heart disease
#here we balance both by upscaling women to men values 

#male 713, female 312

#from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = TfidfVectorizer()
#matrix = vectorizer.fit_transform(heart)

from sklearn.utils import resample
women_upsample = resample(women_count,
                         replace = True,
                         n_samples = len(men_count),
                         random_state = 42)
print(women_upsample.shape)


# In[14]:


heart_upsampled = pd.concat([men_count, women_upsample])

print(heart_upsampled['sex'].value_counts())

heart_upsampled.groupby('sex').size().plot(kind ='pie',
                                        y = '1',
                                        label = "Type",
                                        autopct='%1.1f%%')

#now we have equally sampled men and women datasets. 


# In[15]:


#Upscaling people without heart disease

with_count = heart[heart['target'] == 1]
print(with_count.shape)

without_count = heart[heart['target'] == 0]
print(without_count.shape)


# In[16]:


from sklearn.utils import resample
without_upsample = resample(without_count,
                         replace = True,
                         n_samples = len(with_count),
                         random_state = 42)
print(without_upsample.shape)


# In[17]:


heart_target_upsampled = pd.concat([with_count, without_upsample])

print(heart_target_upsampled['target'].value_counts())

heart_target_upsampled.groupby('target').size().plot(kind ='pie',
                                        y = '1',
                                        label = "Type",
                                        autopct='%1.1f%%')


# In[18]:


Chest_pain_labels = ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"]
order = heart['cp'].value_counts().index

plt.figure(figsize=(10,5))
plt.suptitle("cp")

plt.subplot(1,2,1)
plt.title('Pie chart')
plt.pie(heart['cp'].value_counts(), textprops={'fontsize':12})
plt.subplots_adjust(left=0.125)

plt.subplot(1,2,2)
plt.title('Count plot')
sns.countplot(x='cp', data=heart, order=order)
plt.xticks([0,1,2,3], Chest_pain_labels, rotation=45)

plt.show()

heart['cp'].value_counts()


# In[19]:


target = heart['target'].map({'Presence':1, 'Absence':0})
inputs = heart.drop(['target'], axis=1)


# In[20]:


target_one = int(np.sum(target))
z_c = 0
remove_indices = []

for j in range(target.shape[0]):
    if target[j] == 0:
        z_c += 1
        if z_c > target_one:
            remove_indices.append(i)                                            

print("Indices before balancing data:", target.shape[0])
print("Indices to delete:", len(remove_indices))


# In[21]:


in_balanced = inputs.drop(remove_indices, axis=0)
tar_balanced = target.drop(remove_indices, axis=0)

#Index resetting
in_reset = in_balanced.reset_index(drop=True)
ta_reset = tar_balanced.reset_index(drop=True)

print("Inputs after balancing data:", in_reset.shape[0])
print("Targets after balancing data:", ta_reset.shape[0])  

in_balanced.head()


# In[22]:


labels = ["False", 'True']
order = heart['exang'].value_counts().index

plt.figure(figsize=(10,5))
plt.suptitle("exang")

plt.subplot(1,2,1)
plt.title('Pie chart')
plt.pie(heart['exang'].value_counts(), textprops={'fontsize':12})
plt.subplots_adjust(left=0.125)

plt.subplot(1,2,2)
plt.title('Count plot')
sns.countplot(x='exang', data=heart, order=order)
plt.xticks([0,1], labels=labels)

plt.show()

heart['exang'].value_counts()


# In[23]:


heart.info()


# In[24]:


ax = sns.countplot(x='target', hue='sex', data=heart)
legend_labels, _= ax.get_legend_handles_labels()
ax.legend(legend_labels, ['Female','Male'], bbox_to_anchor=(1,1))
plt.show()


# In[25]:


heart.rename(columns ={"target":"HeartDisease"})


# In[26]:


heart.head


# In[27]:


#Gender
heart['target_label'] = heart['target'].map({0: 'Present', 1: 'Absent'})
ax = sns.countplot(x='target_label', hue='sex', data=heart)
plt.show() 


# In[28]:


#Blood sugar level
ax = sns.countplot(x='target_label', hue='fbs', data=heart)
plt.show()


# In[29]:


plt.suptitle('Types of Chest pain against')
sns.countplot(data = heart,x ="target_label", hue = "cp")
plt.show()


# In[30]:


plt.suptitle('EKG results vs Heart Disease')
sns.countplot(data=heart, x='target_label', hue='restecg')
plt.show()


# In[31]:


heart.info()


# In[32]:


#Heart Disease based on BP 
plot_=sns.stripplot(data=heart, x="trestbps", y="age", hue="target_label")

for ind, label in enumerate(plot_.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)


# In[33]:


plt.suptitle('Excercise angina vs Heart Disease')
sns.countplot(data=heart, x='target_label', hue='exang')
plt.show()


# In[34]:


chol_plot= sns.countplot(data=heart, x="chol", hue="target_label")

for ind, label in enumerate(chol_plot.get_xticklabels()):
    if ind % 50 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)


# In[37]:


sns.countplot(data=heart, x='target_label', hue='thal')


# In[38]:


sns.countplot(data=heart, x='target_label', hue='slope')


# In[39]:


inputs = heart.drop(['target_label', 'target'], axis=1)


# In[40]:


cp = pd.get_dummies(heart['cp'], prefix='cp', drop_first=True)
EKG_results = pd.get_dummies(heart['restecg'], prefix='restecg', drop_first=True)
Number_of_vessels_fluro = pd.get_dummies(heart['ca'], prefix='ca', drop_first=True)
Thallium = pd.get_dummies(heart['thal'], prefix='thal', drop_first=True)

frames = [heart, cp, EKG_results, Number_of_vessels_fluro, Thallium]
heart = pd.concat(frames, axis=1)

heart.drop(columns = ['cp', 'restecg', 'ca', 'thal', 'slope'])

heart['target_label'] = heart['target'].map({0: 'Present', 1: 'Absent'})
inputs = heart.drop(['target', 'target_label'], axis=1)

heart.describe().T


# In[41]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[42]:


heart_scaled = heart.drop(['target_label', 'target'], axis=1)
heart_norm = (heart_scaled-heart_scaled.min())/(heart_scaled.max()-heart_scaled.min())
heart_norm = pd.concat((heart_norm, heart.target),1)

print("scaled heart dataset")
heart_norm.head()


# In[43]:


from sklearn.model_selection import train_test_split

#extracting the best features by removing irrelavant features to make our model run faster
#here we select features
features = heart_norm.drop(columns =['target'], axis = 1)

#Selecting the target feature 
target = heart_norm['target']

#Training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, shuffle = True, test_size = .2, random_state = 44)
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


# In[44]:


heart_norm.info()


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)

predicted = lr.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, predicted))
r2 = r2_score(y_test, predicted)

print('Root mean squared error: ', RMSE)
print("r2: ", r2)


# In[46]:


#Logistic Regression 

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.metrics import roc_curve

log = LogisticRegression()
log.fit(X_train, y_train)

log_pred = log.predict(X_test)

Log_score = accuracy_score(log_pred, y_test)

plt.figure()
metrics.plot_roc_curve(log, X_test, y_test)
plt.title("ROC Curve")
print("Logistic Regression Score:", Log_score)


# In[47]:


#Gaussion Naive Bayes Score, We import GuassianNB library from sklearn
 
from sklearn.naive_bayes import GaussianNB
    
G = GaussianNB()
G.fit(X_train, y_train)

pred_gauss = G.predict (X_test)
score_gauss = accuracy_score(pred_gauss, y_test)

plt.figure()
metrics.plot_roc_curve(G, X_test, y_test)
plt.title("ROC Curve")

plt.show()

print("Gaussian Naive Bayes Score: ", score_gauss)



# In[48]:


#K nearest Neigbours 

from sklearn.neighbors import KNeighborsClassifier 

K = KNeighborsClassifier(n_neighbors =3)
K.fit(X_train, y_train)

K_pred = K.predict(X_test)
K_accur = metrics.accuracy_score(y_test, K_pred)

plt.figure()
metrics.plot_roc_curve(K, X_test, y_test)
plt.title("ROC Curve")


print("K -Neighbors score ",K_accur )


# In[49]:


#Bagging Decision Tree 

from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 

bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, oob_score=True, max_samples =100, bootstrap = True, n_jobs=-1)

bag.fit(X_train, y_train)

bag_oob = bag.oob_score_

bag_pred = bag.predict(X_test)
bag_accur = metrics.accuracy_score(y_test, bag_pred)

plt.figure()
metrics.plot_roc_curve(bag, X_test, y_test)
plt.title("ROC Curve")


print("Bagging Score:" ,bag_accur, "Out of the bag:", bag_oob )


# In[50]:


#calculating the accuray of all the models

#Creating a dictionary of values for accuracy

Dict = {'Algorithms':[
               'Logistic Regression',
               'Gaussian Naive Bayes',
               'K-nearest Neigbors',
               'Bagging Decision Tree'],
               
        'Accuracy':[
                    Log_score,
                    score_gauss,
                    K_accur,
                    bag_oob]
               
               }
Dict = pd.DataFrame(Dict)

display(Dict)



# In[51]:


sns.barplot(y='Algorithms', x = 'Accuracy', data = Dict)


# In[52]:


from sklearn.metrics import plot_roc_curve

lr.fit(X_train, y_train);
log.fit(X_train, y_train);
K.fit(X_train, y_train);
G.fit(X_train, y_train);
bag.fit(X_train, y_train);


# In[107]:


disp = plot_roc_curve(log, X_test, y_test);
plot_roc_curve(K, X_test, y_test, ax = disp.ax_);
plot_roc_curve(G, X_test, y_test, ax = disp.ax_);
plot_roc_curve(bag, X_test, y_test, ax = disp.ax_);


# In[108]:


from sklearn import metrics

def model_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)


    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 'cm': cm}


# In[109]:


#Scores for Logistic Regression

log_eval = model_eval(log, X_test, y_test)

print('Accuracy:', log_eval['acc'])
print('Precision:', log_eval['prec'])
print('Recall:', log_eval['rec'])
print('F1 Score:', log_eval['f1'])
print('Cohens Kappa Score:', log_eval['kappa'])
print('Confusion Matrix:\n', log_eval['cm'])

Dict_log = {'Types_of_scores':['Accuracy',
               'Precision',
               'Recall',
               'F1 Score',
               'Cohens Kappa Score'],
               
        'Scores':[log_eval['acc'],
                    log_eval['prec'],
                    log_eval['rec'],
                    log_eval['f1'],
                    log_eval['kappa']]
               
               }
log_Dict = pd.DataFrame(Dict_log)

display(log_Dict)


# In[110]:


from sklearn.metrics import plot_confusion_matrix 
sns.barplot(y='Scores', x = 'Types_of_scores', data = log_Dict)
plot_confusion_matrix(log, X_test, y_test)


# In[111]:


knn_eval = model_eval(K, X_test, y_test)

print('Accuracy:', knn_eval['acc'])
print('Precision:', knn_eval['prec'])
print('Recall:', knn_eval['rec'])
print('F1 Score:', knn_eval['f1'])
print('Cohens Kappa Score:', knn_eval['kappa'])
print('Confusion Matrix:\n', knn_eval['cm'])

Dict_knn = {'Types_of_scores':['Accuracy',
               'Precision',
               'Recall',
               'F1 Score',
               'Cohens Kappa Score'],
               
        'Scores':[knn_eval['acc'],
                    knn_eval['prec'],
                    knn_eval['rec'],
                    knn_eval['f1'],
                    knn_eval['kappa']]
               
               }
knn_Dict = pd.DataFrame(Dict_knn)

display(knn_Dict)


# In[112]:


G_eval = model_eval(G, X_test, y_test)

print('Accuracy:', G_eval['acc'])
print('Precision:', G_eval['prec'])
print('Recall:', G_eval['rec'])
print('F1 Score:', G_eval['f1'])
print('Cohens Kappa Score:', G_eval['kappa'])
print('Confusion Matrix:\n', G_eval['cm'])

Dict_G = {'Types_of_scores':['Accuracy',
               'Precision',
               'Recall',
               'F1 Score',
               'Cohens Kappa Score'],
               
        'Scores':[G_eval['acc'],
                    G_eval['prec'],
                    G_eval['rec'],
                    G_eval['f1'],
                    G_eval['kappa']]
               
               }
G_Dict = pd.DataFrame(Dict_G)

display(G_Dict)


# In[113]:


bag_eval = model_eval(bag, X_test, y_test)

print('Accuracy:', bag_eval['acc'])
print('Precision:', bag_eval['prec'])
print('Recall:', bag_eval['rec'])
print('F1 Score:', bag_eval['f1'])
print('Cohens Kappa Score:', bag_eval['kappa'])
print('Confusion Matrix:\n', bag_eval['cm'])

Dict_bag = {'Types_of_scores':['Accuracy',
               'Precision',
               'Recall',
               'F1 Score',
               'Cohens Kappa Score'],
               
        'Scores':[bag_eval['acc'],
                    bag_eval['prec'],
                    bag_eval['rec'],
                    bag_eval['f1'],
                    bag_eval['kappa']]
               
               }
bag_Dict = pd.DataFrame(Dict_bag)

display(bag_Dict)


# In[114]:


sns.barplot(y='Scores', x = 'Types_of_scores', data = knn_Dict)
plot_confusion_matrix(K, X_test, y_test)


# In[115]:


sns.barplot(y='Scores', x = 'Types_of_scores', data = G_Dict)
plot_confusion_matrix(G, X_test, y_test)


# In[116]:


sns.barplot(y='Scores', x = 'Types_of_scores', data = bag_Dict)
plot_confusion_matrix(bag, X_test, y_test)


# In[121]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(heart_copy_1.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
plt.savefig('heatmap.png', dpi=100, bbox_inches='tight')


# In[120]:


heart_copy_1 = heart[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope','ca','thal','target']].copy()


# In[ ]:


plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(heart_copy_1.corr()[['target']].sort_values(by='target', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlated features with respect to target', fontdict={'fontsize':18}, pad=16);


# In[123]:


heart_1 = heart.copy()


# In[126]:


X = heart_1.iloc[:, 0:13]
y = heart_1.iloc[:,-1]

best = SelectKBest (score_func = chi2, k=10)
fit = best.fit(X,y)
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(X.columns)

f_scores = pd.concat([columns, scores], axis = 1)
f_scores.columns = ['Specs', 'Score'] 

print(f_scores.nlargest(12, 'Score'))


# In[53]:


sns.countplot(data= heart, x='cp',hue='target_label')
plt.title('Chest Pain Type v/s target\n')


# In[35]:


plt.figure(figsize=(16,6))
sns.distplot(heart['thalach'])


# In[ ]:




