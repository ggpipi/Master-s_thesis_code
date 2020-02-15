#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

xaxis = ""
model = ols('Engel ~ gdp_per_capita', csv_data).fit()
print(model.summary())
result1 = model.predict(csv_data)
print(result1)

model = ols('Engel ~ np.log(gdp_per_capita)', csv_data).fit()
print(model.summary())
result2 = model.predict(csv_data)

model = ols('Engel ~ np.log(gdp_per_capita) + I(1/gdp_per_capita)', csv_data).fit()
print(model.summary())
result3 = model.predict(csv_data)

model = ols('Engel ~ np.log(gdp_per_capita) + food_cpi + Gini', csv_data).fit()
print(model.summary())
result4 = model.predict(csv_data)

model = ols('Engel ~ gdp_per_capita + cpi_coef_of_variation + Gini', csv_data).fit()
print(model.summary())
result5 = model.predict(csv_data)

plt.figure(figsize=(12, 9))
plt.subplot(3, 2, 1)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result1, "o-", label='predict Engel 1')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 2)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result2, "o-", label='predict Engel 2')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 3)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result3, "o-", label='predict Engel 3')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 4)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result4, "o-", label='predict Engel 4')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 5)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result5, "o-", label='predict Engel 5')
plt.ylim(0.23, 0.61)
plt.legend()
plt.savefig("../5模型拟合.png")
plt.clf()

################### Logistic

model = smf.logit('Engel ~ gdp_per_capita', csv_data).fit()
print(model.summary())
result11 = model.predict(csv_data)

model = smf.logit('Engel ~ np.log(gdp_per_capita)', csv_data).fit()
print(model.summary())
result22 = model.predict(csv_data)

model = smf.logit('Engel ~ np.log(gdp_per_capita) + I(1/gdp_per_capita)', csv_data).fit()
print(model.summary())
result33 = model.predict(csv_data)

# model = smf.logit('Engel ~ np.log(gdp_per_capita) + food_cpi', csv_data).fit()
# print(model.summary())
# result4 = model.predict(csv_data)

model = smf.logit('Engel ~ np.log(gdp_per_capita) + food_cpi + Gini', csv_data).fit()
print(model.summary())
result44 = model.predict(csv_data)

model = smf.logit('Engel ~ gdp_per_capita + cpi_coef_of_variation + Gini', csv_data).fit()
print(model.summary())
result55 = model.predict(csv_data)

plt.figure(figsize=(12, 9))
plt.subplot(3, 2, 1)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result11, "o-", label='new predict Engel 1')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 2)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result22, "o-", label='new predict Engel 2')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 3)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result33, "o-", label='new predict Engel 3')
plt.ylim(0.23, 0.61)
plt.legend()
# plt.subplot(3, 2, 4)
# xylabel(xaxis, "Engel coef")
# plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
# plt.plot(csv_data["years"],result4, "o-", label='predict Engel 4')
# plt.ylim(0.23, 0.61)
# plt.legend()
plt.subplot(3, 2, 4)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result44, "o-", label='new predict Engel 4')
plt.ylim(0.23, 0.61)
plt.legend()
plt.subplot(3, 2, 5)
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], "v-", label='Engel')
plt.plot(csv_data["years"],result55, "o-", label='new predict Engel 5')
plt.ylim(0.23, 0.61)
plt.legend()
plt.savefig("../log5模型拟合.png")
plt.clf()

print(np.sqrt((result1-result11).dot(result1-result11)/len(result1-result11)))
print(np.sqrt((result2-result22).dot(result2-result22)/len(result2-result22)))
print(np.sqrt((result3-result33).dot(result3-result33)/len(result3-result33)))
print(np.sqrt((result4-result44).dot(result4-result44)/len(result4-result44)))
print(np.sqrt((result5-result55).dot(result5-result55)/len(result5-result55)))
print(np.sqrt((result1-csv_data["Engel"]).dot(result1-csv_data["Engel"])/len(result1-result11)))
print(np.sqrt((result2-csv_data["Engel"]).dot(result2-csv_data["Engel"])/len(result2-result22)))
print(np.sqrt((result3-csv_data["Engel"]).dot(result3-csv_data["Engel"])/len(result3-result33)))
print(np.sqrt((result4-csv_data["Engel"]).dot(result4-csv_data["Engel"])/len(result4-result44)))
print(np.sqrt((result5-csv_data["Engel"]).dot(result5-csv_data["Engel"])/len(result5-result55)))
print(np.sqrt((csv_data["Engel"]-result11).dot(csv_data["Engel"]-result11)/len(result1-result11)))
print(np.sqrt((csv_data["Engel"]-result22).dot(csv_data["Engel"]-result22)/len(result2-result22)))
print(np.sqrt((csv_data["Engel"]-result33).dot(csv_data["Engel"]-result33)/len(result3-result33)))
print(np.sqrt((csv_data["Engel"]-result44).dot(csv_data["Engel"]-result44)/len(result4-result44)))
print(np.sqrt((csv_data["Engel"]-result55).dot(csv_data["Engel"]-result55)/len(result5-result55)))


