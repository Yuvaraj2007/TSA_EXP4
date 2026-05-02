# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 02-05-2026

### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/AirPassengers.csv')

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

X = data['#Passengers']

plt.plot(X)
plt.title('Original Data')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/a0f362db-0b00-4869-be24-d420dbefb567" />

Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/edf09723-6b5f-4656-9282-7d4dffb03571" />

Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/41c2a4fe-018f-40d9-825f-e687a987bfc3" />

SIMULATED ARMA(2,2) PROCESS:
<img width="993" height="528" alt="image" src="https://github.com/user-attachments/assets/e00303e3-eb14-4c47-8b68-752f9e76ea24" />

Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/d8278596-b3cb-4878-8b09-d7bd782eb3b5" />

Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/1fb54d7b-542d-404f-b374-5c6d239841d6" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
