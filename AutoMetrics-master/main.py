import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sns.set()

cardata = pd.read_csv('cars.csv')


cardata.describe(include='all')

data = cardata.drop(['Model'],axis=1)
data.describe(include='all')

data_no_rv = data.dropna(axis=0)
data_no_rv.describe(include='all')

#datasns = sns.load_dataset("data_no_rv")

sns.displot(data_no_rv['Price'])

plt.show()

#Deal with outliers

q = data_no_rv['Price'].quantile(0.99)
data_price_in = data_no_rv[data_no_rv['Price']<q]
data_price_in.describe(include='all')

sns.displot(data_no_rv['Price'])

plt.show()
#copy from line 27 though 33 and do mileage, enginev, and year

#seaborn.countplot(x="Model", data=data_no_rv)
q = data_price_in['Mileage'].quantile(0.99)
data_mileage_in = data_price_in[data_price_in['Mileage']<q]
data_mileage_in.describe(include='all')

sns.displot(data_mileage_in['Mileage'])

plt.show()
#copy from line 37 though 43 and enginev, and year
q = data_mileage_in['EngineV'].quantile(0.99)
data_enginev_in = data_mileage_in[data_mileage_in['EngineV'] < q]
data_enginev_in.describe(include='all')

sns.displot(data_enginev_in['EngineV'])
plt.show()

q = data_enginev_in['Year'].quantile(0.01)  # For Year, consider low-end outliers too
data_year_in = data_enginev_in[data_enginev_in['Year'] > q]
sns.displot(data_enginev_in['Year'])
plt.show()


# Reset index for cleaned data
data_cleaned = data_year_in.reset_index(drop=True)

# Summary of cleaned data
data_cleaned.describe(include='all')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')

ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')

ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
# Define targets and inputs
targets = data_cleaned['log_price']
inputs = data_cleaned.drop(['log_price'], axis=1)

scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)
