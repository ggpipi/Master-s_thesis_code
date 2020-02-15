from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import stats
import statsmodels.graphics.api as smg

def xylabel(a,b):
    plt.xlabel(a)
    plt.ylabel(b)
    plt.grid(axis='y')

def angle(x1, x2):
    lx1 = np.sqrt(x1.dot(x1))
    lx2 = np.sqrt(x2.dot(x2))
    return x1.dot(x2)/(lx1*lx2)

url = "../新汇总表.csv"
csv_data = pd.read_csv(url, engine='python')

xaxis = "years"
model = ols('Engel ~ I(np.log(gdp_per_capita/food_cpi))+I(1/(gdp_per_capita/food_cpi)**2)', csv_data).fit()
print(model.summary())
result1 = model.predict(csv_data)

plt.figure(figsize=(8, 4))
xaxis = "years"
xylabel(xaxis, "Engel coef")

plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result1, "o-", label="predict Engel 6")
plt.legend()
plt.savefig("../改进一.png")
plt.clf()
