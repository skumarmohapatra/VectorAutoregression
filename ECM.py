import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

# 1. Generate or Load Cointegrated Data (Example)
np.random.seed(42)
n = 200
x = np.cumsum(np.random.normal(0, 1, n))
y = x + np.random.normal(0, 0.5, n) # y is cointegrated with x
data = pd.DataFrame({'x': x, 'y': y})

# 2. Perform Johansen Cointegration Test to determine cointegrating rank
# This step helps determine the 'coint_rank' for the VECM
johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)
print("Johansen Cointegration Test Results:")
print(johansen_test.lr1) # Trace statistic
print(johansen_test.cvt) # Critical values for trace statistic

# Assume from the Johansen test, we determine coint_rank = 1
coint_rank = 1

# 3. Fit the VECM
# k_ar_diff is the number of lags for the differenced terms
vecm_model = VECM(data, k_ar_diff=1, coint_rank=coint_rank)
vecm_fit = vecm_model.fit()

# Print the model summary to see coefficients, including the error correction terms
print("\nVECM Model Summary:")
print(vecm_fit.summary())

# You can also access specific components like the error correction coefficients
# The 'alpha' matrix in the VECM summary represents the adjustment coefficients
# towards the long-run equilibrium.
print("\nError Correction Coefficients (alpha):")
print(vecm_fit.alpha)
