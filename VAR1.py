#import pandas
#pandas.__path__

###pip install --target d:\somewhere\other\than\the\default package_name
###C:\Users\santo\AppData\Local\Programs\Python\Python313\Lib\site-packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# 1. Generate Sample Data
np.random.seed(42) # for reproducibility
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'Series1': np.cumsum(np.random.randn(100) * 0.5 + 0.1),
    'Series2': np.cumsum(np.random.randn(100) * 0.3 + 0.2)
}, index=dates)

# Introduce some interdependency
data['Series2'] = data['Series2'] + 0.5 * data['Series1'].shift(1)
data = data.dropna()

print("Sample Data Head:")
print(data.tail())
print("\n")

# 2. Check for Stationarity (using Augmented Dickey-Fuller test)
def check_stationarity(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]}')
    print(f'p-value for {name}: {result[1]}')
    if result[1] <= 0.05:
        print(f'{name} is likely stationary.\n')
    else:
        print(f'{name} is likely non-stationary. Consider differencing.\n')

print("Stationarity Checks:")
check_stationarity(data['Series1'], 'Series1')
check_stationarity(data['Series2'], 'Series2')

# If non-stationary, apply differencing (example for first difference)
data_diff = data.diff().dropna()
print("After Differencing (if needed):")
check_stationarity(data_diff['Series1'], 'Series1_diff')
check_stationarity(data_diff['Series2'], 'Series2_diff')

# 3. Fit the VAR Model
# Determine the optimal lag order (e.g., using AIC or BIC)
model = VAR(data_diff)
results_aic = model.select_order(maxlags=10)
print("Optimal Lag Order based on AIC/BIC:")
print(results_aic.summary())

results = model.fit(maxlags=15, ic='aic')
fitted_diff = results.fittedvalues


# Fit the model with a chosen lag order (e.g., lag 2)
var_model_fit = model.fit(2) # '2' is an example lag order
print("VAR Model Summary:")
print(var_model_fit.summary())
print("\n")

# 4. Forecasting
# Forecast future values
lag_order = var_model_fit.k_ar
forecast_input = data_diff.values[-lag_order:]
forecast_horizon = 20 # number of steps to forecast

forecast = var_model_fit.forecast(y=forecast_input, steps=forecast_horizon)
forecast_df = pd.DataFrame(forecast, columns=data.columns, 
                           index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), 
                                                periods=forecast_horizon, freq='D'))

print("Forecasted Values:")
print(forecast_df)

print("forecast df cumsum")
print(forecast_df.cumsum())



# 5. Plotting Forecast


# Align index and prepend the last known original value
reconstructed = pd.DataFrame(index=fitted_diff.index, columns=fitted_diff.columns)
reconstructed_fit = pd.DataFrame(index=forecast_df.index, columns=fitted_diff.columns)


for col in data.columns:
    initial_value = data[col].loc[fitted_diff.index[0] - pd.DateOffset(days=1)]
    reconstructed[col] = fitted_diff[col].cumsum() + initial_value

for col in data.columns:
    initial_value_proj = data[col].loc[data.index[len(data)-2]]
    reconstructed_fit[col] = forecast_df[col].cumsum() + initial_value_proj


plt.figure(figsize=(12, 5))
for col in data.columns:
    plt.plot(data.index, data[col], label=f'Original {col}')
    plt.plot(reconstructed.index, reconstructed[col], linestyle='--', label=f'Fitted {col}')
    plt.plot(reconstructed_fit.index, reconstructed_fit[col],linestyle='dotted', label=f'Fitted forecast {col}')
plt.title('Original Series vs VAR Fitted Values (Reconstructed) and VAR Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






