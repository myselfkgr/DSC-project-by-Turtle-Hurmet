import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
from tabulate import tabulate
import warnings
warnings.filterwarnings( ' ignore ' )

df = pd.readcsv( 'student—mat.csv', delimiter= ' ; ' )
print(df.shape)
# Display the first few rows of the DataFrame as a table
print(tabulate(df.head( ) , headers='keys' , tablefmt= 'pretty' ) )
# Display the first 8 columns of the DataFrame as a table
df_subset = df.iloc[:, :8]
print(tabulate(df_subset.head(8), headers='keys' ,tablefmt= 'pretty' ) )

target cnt=df["age"].value_counts() 
print (target_cnt)
sns.countplot(x='age', data=df).set_title("Distribution of target variables")
plt.show()
df.hist (figsize=(6, 14), layout=(4,4), sharex=False)
#Show the plot
plt.show()

#Boxplot
df.plot(kind='box', figsize=(15, 12), layout=(4, 4), sharex=False, subplots=True)
#display
plt.show()
#PLOTLY
#Mapping dictionary for Medu values
fedu_mapping = {
    0: 'None',
    1: 'Primary Education (4th grade)',
    2: '5th to 9th grade',
    3: 'Secondary Education',
    4: 'Higher Education'
}

#Map numeric values to their corresponding descriptions

df['Fedu'] = df['Fedu'].map(fedu_mapping)

#Create a pie chart using plotly.express 
fig = px.pie (df, names-'Fedu', title='Education Received by Father', color_discrete_sequence=px.colors.qualitative.T10)
#Show the plot
fig.show()
#PLOTLY
#Mapping dictionary for Medu values

medu_mapping = {
    0: 'None',
    1: 'Primary Education (4th grade)',
    2: '5th to 9th grade',
    3: 'Secondary Education', 
    4: 'Higher Education'
}
#Map numeric values to their corresponding descriptions 
df['Medu'] = df['Medu'].map(medu mapping)

#Create a pie chart using plotly.express 
fig = px.pie(df, names ='Medu', title='Education Received by Mother', color_discrete_sequence=px.colors.qualitative.T10)

#Show the plot
fig.show()

#PLOTLY
#Mapping dictionary for studytime values

studytime_mapping = {
    1: '<2 hours',
    2: '2 to 5 hours', 
    3: '5 to 10 hours',
    4: '>10 hours'
}

#Map numeric values to their corresponding descriptions 
df['studytime'] = df ['studytime'].map(studytime_mapping)

#Create a pie chart using plotly.express
fig = px.pie(df, names'studytime', title='Weekly Study Time', color_discrete sequence=px.colors.qualitative.T10)

#Show the plot
fig.show()

studytime_mapping = {
    1: '<2 hours',

    2: '2 to 5 hours',

    3: '5 to 10 hours',

    4: '>10 hours'
}
#Map numeric values to their corresponding descriptions 
df['studytime'] = df['studytime'].map(studytime_mapping)

#Create a countplot

plt.figure(figsize=(10, 5))
sns.countplot(x='studytime', hue='sex', data=df, palette ='seismic')

#Display the plot
plt.show()

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#Assuming the file is 'student-mat.cav 
df = pd.read_csv('student-mat.cav', delimiter=';')

#Mapping dictionary for studytime values

studytime mapping ={
    1: '2 hours',

    2: '12 to 5 hours',

    3: "15 to 10 hours",

    4: '10 hours'
}
#Mapping dictionary for failures values

failures mapping ={

    O: 'No failures', 
    1: '1 failure',
    2: '2 failures',
    3: '3 or more failures!'
}
#Map numeric values to their corresponding descriptions

df['studytime'] =df['studytime'].map(studytime_mapping) 
df['failures'] =df['failures'].map(failures_mapping)

#Create a countplot
plt.figure(figsize (12, 6))
sns.countplot (x='studytine', hue='failures', data= df, palette ='viridis')

#Display the plot
plt.title('Study Time vs Failures')
plt.show()

import pandan po
import matplotlib.pyplot as plt
import seaborn as sns

#Assuming the file is 'student-mat.csv'

df = pd.read_csv('student mat.csv', delimiter=';')

#Mapping dictionary for reason valuw
reason mapping = {
    'home': 'Close to home',
    'reputation': 'School Roputation',
    'course': 'Course Reference',
    'other': 'Other'
}
#Map nominal values to their corresponding descriptions 
df['reason']= df['reason'].map(reason napping)

#Create a countplot
plt.figure(figsize=(12, 6)) 
sns.countplot(x-'school', hue-'reason', data-df, palette-'Set2')

#Display the plot
plt.title('School vs Reason for Choosing School')
plt.show()

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#Assuming the file is 'student-mat.csv' 
df=pd.read_csv('student-mat.csv', delimiter=';')

#Selecting specific attributes for the pairplot 
selected_attributes = ['studytime', 'traveltime', 'failures', 'age', 'sex']

#Create a pairplot for the selected attributes
sns.pairplot(df[selected_attributes], hue='sex', palette='Set2') 
plt.suptitle("Pairwise Plot of Selected Student Attributes", y=1.02)
plt.show()
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# Assuming the file is 'student-mat.csv' 
df = pd.read_csv('student-mat.csv', delimiter=';')

# Selecting specific attributes for the pairplot 
selected_attributes = ['G1', 'G2','G3', 'sex']

# Create a pairplot for the selected attributes 
sns.pairplot(df[selected_attributes], hue='sex', palette='Set2') 
plt.suptitle("Pairwise Plot of Selected Student Attributes", y=1.02) 
plt.show()

# Exclude non-numeric columns 
numeric_df = df.select_dtypes(include='number')

#Calculate the correlation matrix 
correlation_matrix = numeric_df.corr()

#Plot the correlation matrix using a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()