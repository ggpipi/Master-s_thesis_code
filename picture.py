#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

url = "../新汇总表.csv"
csv_data = pd.read_csv(url,engine='python')

xaxis = ""

def xylabel(a,b):
    plt.xlabel(a)
    plt.ylabel(b)
    plt.grid(axis='y')

def angle(x1, x2):
    lx1 = np.sqrt(x1.dot(x1))
    lx2 = np.sqrt(x2.dot(x2))
    return x1.dot(x2)/(lx1*lx2)

plt.figure(figsize=(8, 4))
xylabel(xaxis, "GDP per capita")
plt.plot(csv_data["years"],csv_data["gdp_per_capita"], ".-")
plt.savefig("../人均gdp.png")
plt.clf()

xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],csv_data["Engel"], ".-")
plt.savefig("../恩格尔系数.png")
plt.clf()

xylabel(xaxis, "Engel coef")
# plt.text(1988,43,"city Engel coefficient")
# plt.text(2005,48,"country Engel coefficient")
plt.plot(csv_data["years"],csv_data["city_Engel"], "v-", label='Engel coef of city')
plt.plot(csv_data["years"],csv_data["country_Engel"], "o-", label='Engel coef of country')
plt.legend()
plt.savefig("../城乡恩格尔.png")
plt.clf()

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(csv_data["years"], 1/csv_data["gdp_per_capita"], "g^-", label="1/GDP per capita")
ax2.plot(csv_data["years"], csv_data["Engel"], "b.-", label="Engel coef")
ax1.set_xlabel('years')
ax1.set_ylabel('1/GDP per capita, color=g')
ax2.set_ylabel('Engel coef, color=b')
# ax1.legend()
# ax2.legend()
plt.savefig("../对比图.png", dpi=100)
plt.clf()

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(csv_data["years"], csv_data["cpi_coef_of_variation"]*100, "g.-")
ax2.plot(csv_data["years"], csv_data["Engel"], "b.-")
ax1.axis([1983,2019,90,115])
ax2.axis([1983,2019,0.1,0.6])
ax1.set_xlabel('years')
ax1.set_ylabel('cpi coef of variation, color=g')
ax2.set_ylabel('Engel coef, color=b')
# ax1.legend()
# ax2.legend()
plt.savefig("../cpi恩格尔对比.png")
plt.clf()

x1 = csv_data["city_cpi"][-20:, ]
x2 = csv_data["country_cpi"][-20:, ]
y1 = csv_data["city_food_cpi"][-20:, ]
y2 = csv_data["country_food_cpi"][-20:, ]
print(np.sqrt((x1-x2).dot(x1-x2)/len(x1)))
print(np.sqrt((y1-y2).dot(y1-y2)/len(y1)))
# print(angle(y1, y2))

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
xylabel(xaxis, "CPI")
# plt.text(1988,43,"city Engel coefficient")
# plt.text(2005,48,"country Engel coefficient")
plt.plot(csv_data["years"][-20:, ], x1, "v-", label='city CPI')
plt.plot(csv_data["years"][-20:, ], x2, "o-", label='country CPI')
plt.axis([1997, 2018, 98, 107])
plt.legend()
plt.subplot(2, 1, 2)
xylabel(xaxis, "food CPI")
# plt.text(1988,43,"city Engel coefficient")
# plt.text(2005,48,"country Engel coefficient")
plt.plot(csv_data["years"][-20:, ], y1, "v-", label='city food CPI')
plt.plot(csv_data["years"][-20:, ], y2, "o-", label='country food CPI')
plt.axis([1997, 2018, 94, 116])
plt.legend()
plt.savefig("../城乡cpi和城乡食品cpi.png")
plt.clf()

plt.figure(figsize=(8, 4))
xylabel(xaxis, "Engel coef")
# plt.text(1988,43,"city Engel coefficient")
# plt.text(2005,48,"country Engel coefficient")
plt.plot(csv_data["years"],csv_data["city_Engel"], "v-r", label='city Engel coef')
plt.plot(csv_data["years"],csv_data["country_Engel"], "o-g", label='country Engel coef')
plt.plot(csv_data["years"],csv_data["Engel"], "^-b", label='Engel coef')
plt.legend()
plt.savefig("../三恩格尔.png")
plt.clf()

xylabel(xaxis, "Gini coef")
plt.plot(csv_data["years"],csv_data["Gini"], ".-")
plt.savefig("../基尼系数.png")
plt.clf()

plt.figure(figsize=(8, 8))
ax1 = plt.subplot(2,1,1)
ax2 = ax1.twinx()
ax1.plot(csv_data["years"], csv_data["Engel"], "cv-", label='Engel coef')
ax2.plot(csv_data["years"], csv_data["gdp_per_capita"], "o-", label='Gdp per capita')
ax1.set_xlabel('years')
ax1.set_ylabel('Engel coef')
ax2.set_ylabel('Gdp per capita')
ax1.legend()
ax2.legend()
ax1 = plt.subplot(2,1,2)
ax2 = ax1.twinx()
ax1.plot(csv_data["years"], csv_data["Engel"], "cv-", label='Engel coef')
ax2.plot(csv_data["years"], np.log(csv_data["gdp_per_capita"]), "o-", label='ln(Gdp per capita)')
ax1.set_xlabel('years')
ax1.set_ylabel('Engel coef')
ax2.set_ylabel('ln(Gdp per capita)')
ax1.legend()
ax2.legend()
plt.savefig("../gdp与对数gdp.png")
plt.clf()
