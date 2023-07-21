import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# Load data
file_path = r'C:\Users\ACER\Desktop\kc_house_data.csv'
df = pd.read_csv(file_path)

# Select features and target variable
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
Y = df['price']

# Create a linear regression model
lm = LinearRegression()
lm.fit(X, Y)
score_lm = lm.score(X, Y)

# Create a pipeline with polynomial features and linear regression
input_pipeline = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
                  ('model', LinearRegression())]
pipe = Pipeline(input_pipeline)
pipe.fit(X, Y)
score_pipe = pipe.score(X, Y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# Create a Ridge regression model and calculate its score
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
score_ridge = ridge_model.score(x_test, y_test)

# Create polynomial features and use Ridge regression
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[features])
x_test_pr = pr.fit_transform(x_test[features])

ridge_model_poly = Ridge(alpha=0.1)
ridge_model_poly.fit(x_train_pr, y_train)
score_ridge_poly = ridge_model_poly.score(x_test_pr, y_test)

# Print the scores
print("Linear Regression Score:", score_lm)
print("Pipeline Score:", score_pipe)
print("Ridge Regression Score:", score_ridge)
print("Ridge Regression with Polynomial Features Score:", score_ridge_poly)
