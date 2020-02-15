from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import stats
import statsmodels.graphics.api as smg
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import acf
import copy

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

# c = csv_data.iloc[::-1, :].diff(1).dropna(inplace=False)
# plt.plot(csv_data["Engel"][4:], csv_data["Engel"][:-4], 'r.')
# plt.savefig('../一阶滞后.png')
# plt.clf()

# adf检验
# for i in range(1,8):
#     print(adfuller(np.diff(csv_data["cpi_coef_of_variation"], n=i)))

# 协整性检验
i = 1
for i in range(0,5):
    print(coint(np.diff(csv_data["Engel"],n=i), np.diff(csv_data["ln_gdp_per_capita"],n=i)))
    print(coint(np.diff(csv_data["Engel"], n=i), np.diff(1 / csv_data["gdp_per_capita"], n=i)))

var_data = csv_data[["Engel", "ln_gdp_per_capita", "Reciprocal_gdp_per_capita"]].diff().dropna()
# print(var_data)

mod = VAR(endog=var_data, dates=pd.date_range('1986', '2018', freq='Y'))
# 估计最优滞后项系数
# lag_order = mod.select_order()
# print(lag_order.summary())
re = mod.fit()
print(mod.fit().summary())

# 残差
resid = re.resid
# print(resid)


# 自相关性检验

re.plot_acorr(nlags=10, resid=True, linewidth=6)
plt.savefig("../自相关性检验.png")

# q检验 是否拒绝残差为白噪声
(resid_acf, qstat, pvalue) = acf(re.resid['Engel'], nlags=10, qstat=True, fft=False)
# 输出Q检验结果
print(qstat)
print(pvalue)

# # 格兰杰因果
# print(re.test_causality('Engel', ['Engel','ln_gdp_per_capita','Reciprocal_gdp_per_capita'], kind='wald').summary())
# #

# 绘制脉冲响应分析图
res_irf = re.irf(5)
res_irf.plot()
plt.savefig("../脉冲响应.png")
plt.clf()

# 进行方差分解，滞后系数为10
res_fevd = re.fevd(10)
# 输出方差分解结果
print(res_fevd.summary())
# 绘制方差分解图
res_fevd.plot(figsize=(10, 6))
plt.savefig("../方差分解图.png")
plt.clf()


# 预测
true1 = copy.deepcopy(csv_data["Engel"])
re.plot_forecast(20)
plt.savefig("../预测.png")
plt.clf()
fore = re.forecast(np.asarray(var_data)[[-1,]],10)[:,0]
ntime = np.asarray([2018,2019,2020,2021,2022,2023,2024,2025,2026,2027])
list1 = []
for i in range(len(fore)):
    list1.append(sum(fore[:i+1]))
list1 = np.asarray(list1)
list2 = np.asarray([csv_data["Engel"][32]]*10)
list3 = list1+list2
xaxis = "years"
plt.figure(figsize=(8, 4))
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"], true1, ".-")
plt.plot(ntime, list3, ".-")
plt.plot([2017, 2018], [true1[32], list3[0]], "r.-")
plt.savefig("../恩格尔系数预测未来.png")
plt.clf()

#对比
list1 = []
list2 = np.asarray(re.fittedvalues["Engel"])
for i in range(len(list2)):
    list1.append(sum(list2[:i+1]))
list1.insert(0,0)
list1.insert(0,0)
list1 = np.asarray(list1)
# print(list1)

pred = csv_data["Engel"]
pred[1:] = csv_data["Engel"][1]
pred = pred+list1
print(pred)
xaxis = "years"
xylabel(xaxis, "Engel coef")
plt.plot(csv_data["years"],true1, "v-", label="Engel")
plt.plot(csv_data["years"],pred, "o-", label="predict Engel 7")
plt.legend()
plt.savefig("../恩格尔系数预测.png")
plt.clf()



