# Kaggle_data_analysis_project

![alt text](https://github.com/biyichen/Kaggle_data_analysis_project/blob/master/Capture.PNG)

#### import the packages
```
import pandas as pd
import numpy as np
```
#### Load the train and test datasets into pandas dataframes, and drop the null value, and check the data type

```
na_values = ['[]','','NA']
test = pd.read_csv("/Volumes/Transcend/class/6211/test.csv",na_values = na_values)
train = pd.read_csv("/Volumes/Transcend/class/6211/train.csv",na_values = na_values)
train.dtypes
```
ID            int64
VAR_0001     object
VAR_0002      int64
VAR_0003      int64
VAR_0004      int64
VAR_0005     object
VAR_0006    float64
VAR_0007    float64
VAR_0008     object
VAR_0009     object
VAR_0010     object
VAR_0011     object
VAR_0012     object
VAR_0013    float64
VAR_0014    float64
VAR_0015    float64
VAR_0016    float64
VAR_0017    float64
VAR_0018    float64
VAR_0019    float64
VAR_0020    float64
VAR_0021    float64
VAR_0022    float64
VAR_0023    float64
VAR_0024    float64
VAR_0025    float64
VAR_0026    float64
VAR_0027    float64
VAR_0028    float64
VAR_0029    float64
             ...   
VAR_1906      int64
VAR_1907      int64
VAR_1908      int64
VAR_1909      int64
VAR_1910      int64
VAR_1911      int64
VAR_1912      int64
VAR_1913      int64
VAR_1914      int64
VAR_1915      int64
VAR_1916      int64
VAR_1917      int64
VAR_1918      int64
VAR_1919      int64
VAR_1920      int64
VAR_1921      int64
VAR_1922      int64
VAR_1923      int64
VAR_1924      int64
VAR_1925      int64
VAR_1926      int64
VAR_1927      int64
VAR_1928      int64
VAR_1929      int64
VAR_1930      int64
VAR_1931      int64
VAR_1932      int64
VAR_1933      int64
VAR_1934     object
target        int64
Length: 1934, dtype: object

#### change the data type
```
cat_columns = train.select_dtypes(['object']).columns
train[cat_columns] = train[cat_columns].apply(lambda x: x.astype('category'))
train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)
train.dtypes
```
ID            int64
VAR_0001       int8
VAR_0002      int64
VAR_0003      int64
VAR_0004      int64
VAR_0005       int8
VAR_0006    float64
VAR_0007    float64
VAR_0008       int8
VAR_0009       int8
VAR_0010       int8
VAR_0011       int8
VAR_0012       int8
VAR_0013    float64
VAR_0014    float64
VAR_0015    float64
VAR_0016    float64
VAR_0017    float64
VAR_0018    float64
VAR_0019    float64
VAR_0020    float64
VAR_0021    float64
VAR_0022    float64
VAR_0023    float64
VAR_0024    float64
VAR_0025    float64
VAR_0026    float64
VAR_0027    float64
VAR_0028    float64
VAR_0029    float64
             ...   
VAR_1906      int64
VAR_1907      int64
VAR_1908      int64
VAR_1909      int64
VAR_1910      int64
VAR_1911      int64
VAR_1912      int64
VAR_1913      int64
VAR_1914      int64
VAR_1915      int64
VAR_1916      int64
VAR_1917      int64
VAR_1918      int64
VAR_1919      int64
VAR_1920      int64
VAR_1921      int64
VAR_1922      int64
VAR_1923      int64
VAR_1924      int64
VAR_1925      int64
VAR_1926      int64
VAR_1927      int64
VAR_1928      int64
VAR_1929      int64
VAR_1930      int64
VAR_1931      int64
VAR_1932      int64
VAR_1933      int64
VAR_1934       int8
target        int64
Length: 1934, dtype: object

```
cat_columns2 = test.select_dtypes(['object']).columns
test[cat_columns2] = test[cat_columns2].apply(lambda x: x.astype('category'))
test[cat_columns2] = test[cat_columns2].apply(lambda x: x.cat.codes)
train=train.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
train.shape[1]
train.head()
type(train)
```
	ID	VAR_0001	VAR_0002	VAR_0003	VAR_0004	VAR_0005	VAR_0008	VAR_0009	VAR_0010	VAR_0011	...	VAR_1926	VAR_1927	VAR_1928	VAR_1929	VAR_1930	VAR_1931	VAR_1932	VAR_1933	VAR_1934	target
0	2	0	224	0	4300	1	0	0	0	0	...	98	98	998	999999998	998	998	9998	9998	2	0
1	4	0	7	53	4448	0	0	0	0	0	...	98	98	998	999999998	998	998	9998	9998	2	0
2	5	0	116	3	3464	1	0	0	0	0	...	98	98	998	999999998	998	998	9998	9998	2	0
3	7	0	240	300	3200	1	0	0	0	0	...	98	98	998	999999998	998	998	9998	9998	4	0
4	8	2	72	261	2000	2	0	0	0	0	...	98	98	998	999999998	998	998	9998	9998	0	1
5 rows × 1456 columns

pandas.core.frame.DataFrame

#### create model object with one hyperparamater set
```
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
X_train = train.iloc[:,1:1455]
y_train = train['target']
X_train.head()
```
#### Train the model
```
logreg.fit(X_train, y_train)
```
LogisticRegression(C=100000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
          
#### Evaluation metrics on test data
#### The coefficients
```
print('Coefficients: \n', logreg.coef_)
```
Coefficients: 
 [[  1.65847486e-15  -3.40141322e-13  -2.49449643e-13 ...,  -7.57836409e-13
   -3.36454929e-12   8.93754623e-15]]
   
#### The mean squared error
```
print("Mean squared error: %.2f"
      % np.mean((logreg.predict(X_train) - y_train) ** 2))
```
Mean squared error: 0.23

#### Explained variance score: 1 is perfect prediction
```
print('Variance score: %.2f' % logreg.score(X_train, y_train))

```
Variance score: 0.77


```
test=test.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
X_test = test.iloc[:,1:1455]
predictions = logreg.predict(X_test)
```
#### CRATE YOUR SUBMISSION FILE HERE
```

result = pd.DataFrame(predictions)
result.to_csv("A10b_G48027164.csv")
```


