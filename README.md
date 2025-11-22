<img width="1024" height="406" alt="image" src="https://github.com/user-attachments/assets/5bfe14e1-5fca-4fbd-83e9-9990c8a2f73e" /># EDA_EXP_6
# NAME:T.KAVINAJAI
# REGISTER NO:212223100020
**Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

**Algorithm**

1)Import pandas, numpy, seaborn, matplotlib, sklearn libraries.

**Program**
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
df=pd.read_csv(r'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';')
df.head()
df.isnull().sum()
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[~((df < lower) | (df > upper)).any(axis=1)]
df = remove_outliers_iqr(df)
print("Before:", df.shape)
print("After:", df.shape)
sns.histplot(df['alcohol'],color='green',bins=30,kde=True)
sns.histplot(df['volatile acidity'],bins=30,kde=True,color='blue')
sns.histplot(df['pH'],bins=30,kde=True,color='black')
sns.lineplot(x=df['alcohol'],y=df['quality'])
sns.lineplot(x=df['fixed acidity'],y=df['quality'],color='green')
df.head()
corr=df[['fixed acidity','volatile acidity','citric acid','pH','alcohol','quality']]
sns.heatmap(corr.corr(),cbar=True,annot=True,cmap='icefire')
plt.title("HEATMAP")
Y=df[['quality']]
X=df.drop(columns='quality')
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
model=LogisticRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
acc=accuracy_score(Y_test,y_pred)
print("Accuracy is : ",round(acc*100,3))
conf=confusion_matrix(Y_test,y_pred)
print(conf)
sns.heatmap(conf,cbar=True,annot=True,cmap='icefire')
plt.title("Confusion Matrix")
clas=classification_report(Y_test,y_pred,zero_division=True)
print(clas)
```
#**Your Name:T.KAVINAJAI**
#**Your Reg No.:212223100020**
**Output**
<img width="1024" height="406" alt="image" src="https://github.com/user-attachments/assets/428a1aaa-90b5-452c-b3f8-3d0334ddb017" />
<img width="849" height="304" alt="image" src="https://github.com/user-attachments/assets/a50d3d42-ef39-4faf-824c-a5ad304b8b1a" />
  <img width="1022" height="636" alt="image" src="https://github.com/user-attachments/assets/9396d632-28f6-4ca9-99fb-0efa81b08a3b" />
<img width="740" height="443" alt="image" src="https://github.com/user-attachments/assets/9b8791d5-a808-4fc0-b9d5-7887001f41ea" />
<img width="726" height="720" alt="image" src="https://github.com/user-attachments/assets/cb551a81-6a1c-4310-92a3-1c6d900c034f" />
<img width="1145" height="916" alt="image" src="https://github.com/user-attachments/assets/e3646de2-1e61-4e18-853a-e50425eed70f" />
<img width="1179" height="918" alt="image" src="https://github.com/user-attachments/assets/15226b98-51d5-4850-a267-5ba25c75021e" />
<img width="1218" height="286" alt="image" src="https://github.com/user-attachments/assets/a4807828-a91c-4516-9c47-76b58b2773a8" />

**Result**

THUS WINE ANALYSIS IS DONE SUCCESSFULLY
