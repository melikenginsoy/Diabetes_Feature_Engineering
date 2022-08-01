#############################################
# Feature Engineering of The Diabetes Dataset
#############################################

##################
# Business Problem
##################

# It is desired to develop a machine learning model that can predict whether people have
# diabetes when their characteristics are specified.
# In this project, necessary data analysis and feature engineering steps will be performed
# before model development.

####################
# About the Data Set
####################

# The dataset is part of the large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases
# in the USA. Data used for diabetes research on Pima Indian women aged 21 and over living in Phoenix,the 5th
# largest city of the State of Arizona in the USA.
# The target variable is specified as "outcome";
# 1 indicates positive diabetes test result, 0 indicates negative.

# Pregnancies: Number of pregnancies
# Glucose: 2-hour plasma glucose concentration in the oral glucose tolerance test
# BloodPressure: Diastolic Blood Pressure (mm Hg)
# SkinThickness: Thickness of Skin
# Insulin: 2-hour serum insulin (mu U/ml)
# DiabetesPedigreeFunction: A function that calculates the probability of having diabetes
#                           according to the descendants
# BMI: Body Mass Index
# Age: Age (year)
# Outcome: Diabetic ( 1 or 0 )


############################
# Exploratory Data Analysis
############################

# 1- Necessary Libraries:

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from statsmodels.stats.proportion import proportions_ztest
from sklearn.ensemble import RandomForestClassifier

# 2- Customize DataFrame to display:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# 3- Loading the Dataset:

def load():
    data = pd.read_csv("Diabetes/diabetes.csv")
    return data


df = load()

# 4- Dataset Overview:

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

"""
##################### Shape #####################
(768, 9)
##################### Types #####################
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object
##################### Head #####################
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0 33.600                     0.627   50        1
1            1       85             66             29        0 26.600                     0.351   31        0
2            8      183             64              0        0 23.300                     0.672   32        1
3            1       89             66             23       94 28.100                     0.167   21        0
4            0      137             40             35      168 43.100                     2.288   33        1
##################### Tail #####################
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction  Age  Outcome
763           10      101             76             48      180 32.900                     0.171   63        0
764            2      122             70             27        0 36.800                     0.340   27        0
765            5      121             72             23      112 26.200                     0.245   30        0
766            1      126             60              0        0 30.100                     0.349   47        1
767            1       93             70             31        0 30.400                     0.315   23        0
##################### NA #####################
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
##################### Quantiles #####################
                           count    mean     std    min     0%     5%     50%     95%     99%    100%     max
Pregnancies              768.000   3.845   3.370  0.000  0.000  0.000   3.000  10.000  13.000  17.000  17.000
Glucose                  768.000 120.895  31.973  0.000  0.000 79.000 117.000 181.000 196.000 199.000 199.000
BloodPressure            768.000  69.105  19.356  0.000  0.000 38.700  72.000  90.000 106.000 122.000 122.000
SkinThickness            768.000  20.536  15.952  0.000  0.000  0.000  23.000  44.000  51.330  99.000  99.000
Insulin                  768.000  79.799 115.244  0.000  0.000  0.000  30.500 293.000 519.900 846.000 846.000
BMI                      768.000  31.993   7.884  0.000  0.000 21.800  32.000  44.395  50.759  67.100  67.100
DiabetesPedigreeFunction 768.000   0.472   0.331  0.078  0.078  0.140   0.372   1.133   1.698   2.420   2.420
Age                      768.000  33.241  11.760 21.000 21.000 21.000  29.000  58.000  67.000  81.000  81.000
Outcome                  768.000   0.349   0.477  0.000  0.000  0.000   0.000   1.000   1.000   1.000   1.000
"""

# The data set consists of 9 numerical variables and 768 observation units.
# There are no missing observations in the data set.
# But looking at the descriptive statistics of the variables, for example Glucose, BloodPressure, SkinThickness,
# it is observed that their minimum values are 0. Since these values cannot be zero,
# they can be considered as missing values.
# By entering NaN value instead of these values, it will be considered as a NaN observation.

change_variable = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

df[change_variable] = df[change_variable].replace(0,np.NaN)

df.isnull().sum()

"""
Pregnancies                   0
Glucose                       5
BloodPressure                35
SkinThickness               227
Insulin                     374
BMI                          11
DiabetesPedigreeFunction      0
Age                           0
"""

# 5- Numeric and Categorical variables:

# Although the data set consists of 9 numerical variables, some variables may actually be categorical.
# For example, the dependent variable Outcome is binary coded. In fact, it is a categorical variable.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables:
        cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 768
Variables: 9
cat_cols: 1
num_cols: 8
cat_but_car: 0
num_but_cat: 1
"""
cat_cols
# ['Outcome']


# Analysis of numerical and categorical variables:

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


for col in cat_cols:
    cat_summary(df, col)

"""
Outcome  Ratio
0      500 65.104
1      268 34.896
"""


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

# Average of numerical variables relative to the dependent variable:

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# 6- Outliers Analysis:

# Define an outlier thresholds for variables:

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Check for outliers for variables:

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in df.columns:
    print(col, check_outlier(df, col))

"""
Pregnancies False
Glucose False
BloodPressure False
SkinThickness False
Insulin False
BMI False
DiabetesPedigreeFunction False
Age False
Outcome False
"""

# Since we chose the quartile values q1 and q3 in a wide range, there are no outliers in the variables.

# 7- The Missing Values Analysis:

# Missing value and ratio analysis for variables:

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)

"""
               n_miss  ratio
Insulin           374 48.700
SkinThickness     227 29.560
BloodPressure      35  4.560
BMI                11  1.430
Glucose             5  0.650

['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
"""

# 8- Correlation Analysis:

def correlated_cols(dataframe, plot=False):
    corr_matrix = dataframe.corr()

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr_matrix, cmap="RdBu")
        plt.show()
    return print(corr_matrix)


correlated_cols(df, plot=True)

"""
                          Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome
Pregnancies                     1.000    0.128          0.214          0.100    0.082 0.022                    -0.034 0.544    0.222
Glucose                         0.128    1.000          0.223          0.228    0.581 0.233                     0.137 0.267    0.495
BloodPressure                   0.214    0.223          1.000          0.227    0.098 0.289                    -0.003 0.330    0.171
SkinThickness                   0.100    0.228          0.227          1.000    0.185 0.648                     0.115 0.167    0.259
Insulin                         0.082    0.581          0.098          0.185    1.000 0.228                     0.130 0.220    0.303
BMI                             0.022    0.233          0.289          0.648    0.228 1.000                     0.155 0.026    0.314
DiabetesPedigreeFunction       -0.034    0.137         -0.003          0.115    0.130 0.155                     1.000 0.034    0.174
Age                             0.544    0.267          0.330          0.167    0.220 0.026                     0.034 1.000    0.238
Outcome                         0.222    0.495          0.171          0.259    0.303 0.314                     0.174 0.238    1.000
"""

# There is a moderate positive relationship between Insulin and Glucose.(0.581)
# There is a moderate positive correlation between BMI and SkinThickness. (0.648)
# There is a moderate positive correlation between Pregnancies and Age. (0.544)

############################
# Future Engineering
############################

# 1- Necessary actions for missing and outliers values:

na_cols = missing_values_table(df, True)
# ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# * Comparing the dependent variable in terms of missing variables:

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")



missing_vs_target(df, "Outcome", na_cols)

"""
                 TARGET_MEAN  Count
Glucose_NA_FLAG                    
0                      0.349    763
1                      0.400      5
                       TARGET_MEAN  Count
BloodPressure_NA_FLAG                    
0                            0.344    733
1                            0.457     35
                       TARGET_MEAN  Count
SkinThickness_NA_FLAG                    
0                            0.333    541
1                            0.388    227
                 TARGET_MEAN  Count
Insulin_NA_FLAG                    
0                      0.330    394
1                      0.369    374
             TARGET_MEAN  Count
BMI_NA_FLAG                    
0                  0.351    757
1                  0.182     11
"""

# * Filling the missing values in each variable specific to the variable:

# With median:

df["Glucose"] = df["Glucose"].fillna(df["Glucose"].median())
df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].median())
df["BMI"] = df["BMI"].fillna(df["BMI"].median())

# With KNN Method:

dff = pd.get_dummies(df[["Insulin","SkinThickness"]], drop_first=True)

dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

"""
 Insulin  SkinThickness
0      NaN          0.304
1      NaN          0.239
2      NaN            NaN
3    0.096          0.174
4    0.185          0.304
"""

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

"""
Insulin  SkinThickness
0    0.240          0.304
1    0.243          0.239
2    0.170          0.241
3    0.096          0.174
4    0.185          0.304
"""

df["Insulin"] = dff["Insulin"]

df["SkinThickness"] = dff["SkinThickness"]

# * Multivariate Outlier Analysis Local Outlier Factor (LOF)

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[5]

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# 2- Creating New Features:

# for Glucose:

df.loc[(df['Glucose'] < 70), 'GLUCOSE_CAT'] ="hipoglisemi"
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100) , 'GLUCOSE_CAT'] ="normal"
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] < 126) , 'GLUCOSE_CAT'] ="imparied glucose"
df.loc[(df['Glucose'] >= 126), 'GLUCOSE_CAT'] ="hiperglisemi"

df.groupby("GLUCOSE_CAT").agg({"Outcome":["mean","count"]})

# for Age:

df.loc[(df['Age'] >= 18) & (df['Age'] < 30) , 'AGE_CAT'] ="young_age"
df.loc[(df['Age'] >= 30) & (df['Age'] < 45) , 'AGE_CAT'] ="mature_age"
df.loc[(df['Age'] >= 45) & (df['Age'] < 65) , 'AGE_CAT'] ="middle_age"
df.loc[(df['Age'] >= 65) & (df['Age'] < 75) , 'AGE_CAT'] ="old_age"
df.loc[(df['Age'] >= 75) , 'AGE_CAT'] ="elder_age"

df.groupby("AGE_CAT").agg({"Outcome":["mean","count"]})

# for BMI:

df.loc[(df['BMI'] < 16), 'BMI_CAT'] ="overweak"
df.loc[(df['BMI'] >= 16) & (df['BMI'] < 18.5) , 'BMI_CAT'] ="weak"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25) , 'BMI_CAT'] ="normal"
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30) , 'BMI_CAT'] ="overweight"
df.loc[(df['BMI'] >= 30) & (df['BMI'] <= 45) , 'BMI_CAT'] ="obese"

df.groupby("BMI_CAT").agg({"Outcome":["mean","count"]})

# for Diastolic Blood Pressure:

df.loc[(df['BloodPressure'] < 70)  , 'DBP_CAT'] ="low"
df.loc[(df['BloodPressure'] >= 70) & (df['BMI'] < 90) , 'DBP_CAT'] ="normal"
df.loc[(df['BloodPressure'] >= 90 ) , 'DBP_CAT'] ="high"

df.groupby("DBP_CAT").agg({"Outcome":["mean","count"]})

# 3- One-Hot Encoding:

# Create a variable for each observation unit by making the variables we have one hot encoder.

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

# 4- Standardization for numerical variables:

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()

############
# Modelling
############

# Dependent variable:
y = df["Outcome"]

# Independent variables:
X = df.drop(["Outcome"], axis=1)

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7792207792207793
