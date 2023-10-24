# Import necessary libraries
import matplotlib.pyplot as plt
from adjustText import adjust_text
import re
import pandas as pd
import seaborn as sns
from pymatgen.core import Composition
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('dataframe.csv')

# Data Preprocessing
df['elements'] = df['formula'].apply(lambda x: str(x).split('-'))
elements_to_include = ['Si', 'C', 'Li', 'O']
for element in elements_to_include:
    df[element] = df['elements'].apply(lambda x: str(x).count(element) / len(str(x)))
    
# Visualize the distribution of bulk modulus
plt.figure(figsize=(8, 7))
sns.histplot(df['K_VRH'], bins=10, kde=True)
plt.xlabel('Bulk Modulus (GPa)', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.tight_layout()
plt.savefig('BulkModDist.png', dpi=600)

# Explore the relationship between elements and bulk modulus
plt.figure(figsize=(8, 8))
for element in elements_to_include:
    sns.scatterplot(x=element, y='K_VRH', data=df, label=element)
    plt.xlabel('Element Fractions', fontsize=18)
    plt.ylabel('Bulk Modulus', fontsize=18)
    plt.tight_layout()
    plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=14)
plt.legend(loc='upper center', ncol=4, fontsize=14)
plt.tight_layout()
plt.savefig('ConcSpaceGroup.png', dpi=600)

# Statistical Analysis
correlation_matrix = df.corr()

# Machine Learning Modeling (Optional)
X = df[elements_to_include]
y = df['K_VRH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualize Feature Importance
feature_importance = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Element': elements_to_include, 'Importance': feature_importance})
feature_importance_df.sort_values('Importance', ascending=False, inplace=True)

plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Element', data=feature_importance_df)
plt.xlabel('Feature Importance')
plt.ylabel('Element')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('ML.png')
plt.show()

# Calculate correlation coefficients
correlation_matrix = df[elements_to_include + ['K_VRH']].corr()

# Explore the relationship between space group and bulk modulus
plt.figure(figsize=(8, 7))
sns.boxplot(x='space_group_y', y='K_VRH', data=df)
plt.xlabel('Space Group', fontsize=20)
plt.ylabel('Bulk Modulus', fontsize=20)
plt.xticks(rotation='vertical')
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('Correlations.png', dpi=900)
plt.show()
