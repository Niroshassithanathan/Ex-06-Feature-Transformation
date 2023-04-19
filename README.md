# Ex-06-Feature-Transformation
AIM:
 To read the given data and perform Feature Transformation process and save the data to a file.
EXPLANATION:
  Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
ALGORITHM:
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Print the transformed features

PROGRAM:
NAME: NIROSHA.S
REG NO: 212222230097
~~~.py
# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
~~~

# Reading CSV File
~~~.py
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 _Intro to DS/Exp_6/Data_to_Transform.csv")
df
~~~
# Basic Process
~~~.py
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()
~~~
# Before Transformation
~~~.py
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()
~~~
# Log Transformation
~~~.py
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()


df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
~~~
# Reciprocal Transformation
~~~.py
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# Square Root Transformation

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
~~~
# Power Transformation
~~~.py
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()


from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
~~~
# Quantile Transformation
~~~.py
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
~~~
# OUTPUT:
# Reading CSV File
# df

![DS F1](https://user-images.githubusercontent.com/121418437/232966656-3a56132f-151f-4e35-a057-7ed185d8fce9.png)

# Basic Process
# Head

![DS F2](https://user-images.githubusercontent.com/121418437/232966689-97a0cdea-4d60-44b6-9ddb-631d0c1f9e67.png)

# Info

![DS F3](https://user-images.githubusercontent.com/121418437/232966725-a0e4e1a2-3203-49a8-a8c4-b84d05aca71e.png)

# Describe

![DS F4](https://user-images.githubusercontent.com/121418437/232966751-cb6a7186-aa59-407a-8aa5-4b689453a09a.png)

# Tail

![DS F5](https://user-images.githubusercontent.com/121418437/232966778-fdd61c40-2553-4ea9-9aee-f384820bfd7b.png)

# Shape

![DS F6](https://user-images.githubusercontent.com/121418437/232966795-7f1637bd-c580-4335-acf9-327ee6c1c508.png)

# Columns

![DS F7](https://user-images.githubusercontent.com/121418437/232966813-06b47a07-151a-4a57-be2b-5eb25cbff5dd.png)

# Null Values

![DS F8](https://user-images.githubusercontent.com/121418437/232966902-be355710-2696-40df-bc29-f398dbb26bd2.png)

# Duplicate Values

![DS F9](https://user-images.githubusercontent.com/121418437/232966844-73d7eb74-c1f4-402f-80f3-c5a712928c43.png)

# Before Transformation
# Highly Positive Skew

![DS F10](https://user-images.githubusercontent.com/121418437/232966924-d99782b1-1e64-479b-9143-85521693d01d.png)

# Highly Negative Skew

![DS F11](https://user-images.githubusercontent.com/121418437/232966957-b5e6a3c9-494e-4e1b-8c9a-d4790c837494.png)

# Moderate Positive Skew

![DS F13](https://user-images.githubusercontent.com/121418437/232966981-8534c9da-2041-4a6d-9ad0-81f855e82579.png)

# Moderate Negative Skew

![DS F14](https://user-images.githubusercontent.com/121418437/232967024-e3a0be9b-6ab2-4f5d-932f-45eac4104d6a.png)

# Log Transformation
# Highly Positive Skew

![DS F15](https://user-images.githubusercontent.com/121418437/232967068-dc3b1643-bf03-42fb-a0b4-e3e59e9219d5.png)

# Moderate Positive Skew

![DS F16](https://user-images.githubusercontent.com/121418437/232967081-74150951-e337-40b5-9b80-99fdc5a81e2d.png)

# Reciprocal Transformation
# Highly Positive Skew

![DS F17](https://user-images.githubusercontent.com/121418437/232967108-01384326-60d7-4e4f-a3a7-d73598ff40a4.png)

# Square Root Transformation
# Highly Positive Skew

![DS F19](https://user-images.githubusercontent.com/121418437/232967127-11483a6d-5f3c-43ec-8806-306dc7db8d67.png)

# Power Transformation
# Moderate Positive Skew

![DS F20](https://user-images.githubusercontent.com/121418437/232967157-a6f81c66-2363-4bdc-9c64-4bc0008b055d.png)

# Moderate Negative Skew


# Quantile Transformation
# Moderate Negative Skew

![DS F21](https://user-images.githubusercontent.com/121418437/232967241-1b722ad8-8296-46a9-bb9a-baed43d589c3.png)


# RESULT:
    Thus feature transformation is done for the given dataset.
