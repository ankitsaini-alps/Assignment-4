#!/usr/bin/env python
# coding: utf-8

# # DATA FRAME-1 - AIRBNB CLEANING AND VISUALIZATION
# 
# HR DATA SET CLEANING DONE SEPERATELY IN SAME NOTE BOOK BELOW

# In[144]:


import pandas as pd
import numpy as np


# In[145]:


# Capturing the Airbnb data set

url = "https://raw.githubusercontent.com/sandyalps/Airbnbdataset/main/Airbnb%20Dataset%2019.csv"

#df = pd.read_csv("D:\Airbnb data set cleaning and visualisation using python\Airbnb Dataset 19.csv")

df = pd.read_csv(url)


# In[146]:


# Exploring the dataset

print (df.head())


# In[147]:


print (df.info())


# In[148]:


missing_rows = df[df.isnull().any(axis=1)]

print (missing_rows)


# In[149]:


# delete NaN rows

df= df.dropna(subset=["last_review", "reviews_per_month"])


# In[150]:


print (df.info())


# In[179]:


df['last_review'] = pd.to_datetime(df['last_review'])


# In[151]:


print(df.describe())


# In[152]:


print(df.isnull().sum())


# In[153]:


print (df.info())


# In[154]:


duplicates_in_columns = df.duplicated().any()

print(duplicates_in_columns)


# In[155]:


for column in df.columns:
    unique_values = df[column].unique()
    print(f"Column '{column}' has {len(unique_values)} unique values:")
    print(unique_values)
    print("\n")


# In[156]:


df.to_csv('Cleaned Airbnb dataset.csv', index=False)


# # Visualization of Airbnb data set
# 
# 1. Purpose: Understand the distribution of prices for accommodations.
# 

# In[157]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.hist(df['price'], bins=20, color='skyblue')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')
plt.show()


# 2. Purpose: Compare the count of each room type.

# In[158]:


import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='room_type')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.title('Bar Chart of Room Types')
plt.show()


# 3. Purpose: Visualize the geographic distribution of accommodations.

# In[159]:


plt.figure(figsize=(10, 8))
plt.scatter(df['longitude'], df['latitude'], alpha=0.6, c='green', edgecolors='grey')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot of Latitude vs. Longitude')
plt.show()


# 4. Purpose: Show the proportion of accommodations in each neighbourhood group.

# In[160]:


plt.figure(figsize=(8, 8))
df['neighbourhood_group'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Pie Chart of Neighbourhood Groups')
plt.show()


# 5. Purpose: Track the trend of reviews received over time.

# In[181]:


plt.figure(figsize=(10, 6))
df_sorted = df.sort_values(by='last_review')
plt.plot(df_sorted['last_review'], df_sorted['reviews_per_month'], marker='o', color='orange')
plt.xlabel('Last Review Date')
plt.ylabel('Reviews per Month')
plt.title('Line Chart of Reviews per Month')
plt.xticks(rotation=45)
plt.show()


# 6. Purpose: Compare the distribution of prices for each room type.

# In[162]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='room_type', y='price')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.title('Box Plot of Price by Room Type')
plt.show()


# 7. Purpose: Understand the combination of neighbourhood groups and room types

# In[163]:


crosstab_df = pd.crosstab(df['neighbourhood_group'], df['room_type'])
crosstab_df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Neighbourhood Group')
plt.ylabel('Count')
plt.title('Stacked Bar Chart of Neighbourhood Groups and Room Types')
plt.show()


# # DATA FRAME -2 HR DATA SET

# In[167]:


url = "https://raw.githubusercontent.com/sandyalps/Airbnbdataset/main/Uncleaned%20HRDataset_v14.csv"


df2 = pd.read_csv(url)


# In[142]:


# Exploring the dataset

print (df2.head())


# In[143]:


print (df2.info())


# In[168]:


date_columns = ['DOB', 'DateofHire', 'LastPerformanceReview_Date']
df2[date_columns] = df2[date_columns].apply(pd.to_datetime)


# In[169]:


categorical_columns = ['MaritalDesc', 'CitizenDesc', 'HispanicLatino', 'RaceDesc', 'EmploymentStatus', 'Department', 'RecruitmentSource', 'PerformanceScore']
df2_encoded = pd.get_dummies(df2, columns=categorical_columns)


# In[170]:


missing_values = df2.isnull().sum()
print("Missing values in each column:")
print(missing_values)


# In[173]:


# Fill the manager id
df2['ManagerID'].fillna(-1, inplace=True)


# In[175]:


print (df2.info())


# In[183]:


print(df2.describe())


# In[185]:


print(df2.isnull().sum())


# Leaving DateofTermination coloumn as it is, this data is relevant fo analysis. and we can not fill in this data. 
# Because termination is happens when employees leaves the company.

# In[186]:


for column in df2.columns:
    unique_values = df2[column].unique()
    print(f"Column '{column}' has {len(unique_values)} unique values:")
    print(unique_values)
    print("\n")


# # Visualization of HR data set

# 1. To visualize the distribution of salaries among employees.

# In[187]:


plt.figure(figsize=(8, 6))
plt.hist(df2['Salary'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Histogram of Employee Salaries')
plt.show()


# 2. Purpose: To explore the relationship between employee salary and engagement survey scores.

# In[188]:


plt.figure(figsize=(8, 6))
plt.scatter(df2['Salary'], df2['EngagementSurvey'], alpha=0.6, c='orange')
plt.xlabel('Salary')
plt.ylabel('Engagement Survey Score')
plt.title('Salary vs. Engagement Survey Score')
plt.show()


# 3. Purpose: To compare the employee satisfaction levels across different races.

# In[193]:


plt.figure(figsize=(10, 6))
plt.boxplot([df2[df2['RaceDesc'] == race]['EmpSatisfaction'] for race in df2['RaceDesc'].unique()], 
            labels=df2['RaceDesc'].unique())
plt.xlabel('Race')
plt.ylabel('Employee Satisfaction Level')
plt.title('Employee Satisfaction Levels by Race')
plt.xticks(rotation=45, ha='right')
plt.show()


# 4. Purpose: To visualize the distribution of employees across different positions.

# In[195]:


import pandas as pd
import matplotlib.pyplot as plt


position_counts_by_dept = df2.groupby(['Department', 'Position']).size().reset_index(name='Count')


pivot_table = position_counts_by_dept.pivot(index='Department', columns='Position', values='Count')
pivot_table.fillna(0, inplace=True) 

plt.figure(figsize=(12, 8))
pivot_table.plot(kind='bar', stacked=True, colormap='tab20')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.title('Distribution of Employees Across Positions in each Department')
plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()


# 5. Purpose: To visualize the count of employees in each performance score category.

# In[196]:


import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=df2, x='PerformanceScore', palette='coolwarm')
plt.xlabel('Performance Score')
plt.ylabel('Count')
plt.title('Distribution of Performance Scores')
plt.show()


# In[ ]:




