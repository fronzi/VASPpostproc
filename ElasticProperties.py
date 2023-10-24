#Marco Fronzi
#This code loads the elastic tensor dataset, preprocesses it, creates features, trains a Random Forest Regressor model, and evaluates its performance. It also includes a residual plot for visualization.

# Import necessary libraries
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty, CompositionToOxidComposition, OxidationStates
from matminer.featurizers.structure import DensityFeatures
from matminer.datasets.convenience_loaders import load_elastic_tensor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the elastic tensor dataset
df = load_elastic_tensor()

# Drop unnecessary columns
unwanted_columns = ["volume", "nsites", "elastic_anisotropy", "G_VRH", "poisson_ratio",
                    "compliance_tensor", "elastic_tensor", "elastic_tensor_original",
                    "K_Voigt", "G_Voigt", "K_Reuss", "G_Reuss"]
df = df.drop(unwanted_columns, axis=1)

# Create Composition and ElementProperty features
df = StrToComposition().featurize_dataframe(df, "formula")
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition")

# Add oxidation state features
df = CompositionToOxidComposition().featurize_dataframe(df, "composition_oxid", ignore_errors=True)
os_feat = OxidationStates()
df = os_feat.featurize_dataframe(df, "composition_oxid", ignore_errors=True)

# Add Density features
df_feat = DensityFeatures()
df = df_feat.featurize_dataframe(df, "structure")

# Define target variable (bulk modulus) and input features
y = df['K_VRH'].values
excluded_columns = ["K_VRH", "formula", "material_id", "structure", "composition", "composition_oxid"]
X = df.drop(excluded_columns, axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train a Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=50, random_state=1)
rf_reg.fit(X_train, y_train)

# Evaluate the model on training and test data
training_R2 = rf_reg.score(X_train, y_train)
training_RMSE = np.sqrt(mean_squared_error(y_true=y_train, y_pred=rf_reg.predict(X_train)))
test_R2 = rf_reg.score(X_test, y_test)
test_RMSE = np.sqrt(mean_squared_error(y_true=y_test, y_pred=rf_reg.predict(X_test)))

# Display model evaluation metrics
print(f'Training R2: {training_R2:.3f}')
print(f'Training RMSE: {training_RMSE:.3f}')
print(f'Test R2: {test_R2:.3f}')
print(f'Test RMSE: {test_RMSE:.3f}')

# Visualize the residuals
from figrecipes import PlotlyFig
pf_rf = PlotlyFig(x_title='Bulk modulus prediction residual (GPa)',
                  y_title='Probability',
                  title='Random forest regression residuals',
                  mode="notebook",
                  filename="rf_regression_residuals.html")

hist_plot = pf_rf.histogram(data=[y_train - rf_reg.predict(X_train),
                                  y_test - rf_reg.predict(X_test)],
                            histnorm='probability', colors=['blue', 'red'],
                            return_plot=True)
hist_plot["data"][0]['name'] = 'train'
hist_plot["data"][1]['name'] = 'test'
pf_rf.create_plot(hist_plot)
