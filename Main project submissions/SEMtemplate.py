import pandas as pd
from semopy import Model

# Define your model as a string (LISREL syntax)
model_desc = """
# Measurement model
# Latent variables: F1 and F2
F1 =~ y1 + y2 + y3
F2 =~ y4 + y5

# Structural model
F2 ~ F1
"""

# Create a DataFrame with your observed data
data = {
    'y1': [1, 2, 3, 4, 5],
    'y2': [2, 3, 4, 5, 6],
    'y3': [3, 4, 5, 6, 7],
    'y4': [4, 5, 6, 7, 8],
    'y5': [5, 6, 7, 8, 9]
}
df = pd.DataFrame(data)

# Create the model
model = Model(model_desc)

# Fit the model to the data
model.fit(df)

# Get the results
results = model.inspect()
print(results)

# Get parameter estimates
params = model.param_vals
print("Parameter Estimates:")
print(params)
