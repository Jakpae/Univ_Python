# 실습 13.1
# Author : LeeSeungJun
# Student ID : 22312050
# Dept : Info & Comm. Eng
# Date : June 8, 2023

# 선형회귀 : 변수 사이의 상관관계 , 주어진 데이터 (]독립변수와 종속변수간의 관계를 (다항) 선형 모델 / (다항) 선형 방정식으로 생성)
# 다항 선형 회귀는 독립 변수와 종속 변수 사이의 관계를 다항식으로 모델링하는 선형 회귀 방법
# 알고있는 데이터를 통해 학습한 이후에 나중에 들어온 데이터를 분석하기 위해 즉, 예측하기 위해


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Numpy Linspace를 사용해서 -7.5와 7.5 사이에 100개의 난수를 생성하여 x에 넣음
x = np.linspace(-7.5, 7.5, 100)

# a,b,c,d의 대한 값을 지정
a, b, c, d = 1, 0, -25, 0

# x에 대한 이차식 y 값을 생성
y = a*x**3 + b*x**2 + c*x + d

# mu, sigma 값을 지정
mu, sigma = 0.0, 50.0

# mu. sigma, x.size를 사용하여 noise를 생성
noise = np.random.normal(mu, sigma, x.size)

# y와 noise를 사용하여 y_noise 생성
y_noise = y + noise
X = np.vstack([x**3, x**2, x, np.ones(len(x))]).T

# PolynomialFeatures를 사용하여 x_poly를 생성
poly_features = PolynomialFeatures()
X_poly = poly_features.fit_transform(X)

# X_poly와 y_noise를 사용하여 LinearRegression 객체 생성
lin_reg = LinearRegression()
# X_poly, y_noise 를 사용해 학습하기
lin_reg.fit(X_poly, y_noise)

print("lin_reg model - intercept_ ({}),\n coef_({}),\nscore(X,y)= ({})".\
      format(lin_reg.intercept_, lin_reg.coef_, lin_reg.score(X_poly, y)))

# y_noise와 X_poly를 사용하여 y_pred 생성
x_test = X
x_poly = poly_features.transform(x_test)
# x_poly에 대한 예측하기
y_pred = lin_reg.predict(x_poly)

# 그래프 그리기
plt.plot(x, y_noise, "k.", label="y_noise") # 그래프에 y_noise를 검은색 점 형태로 표시
plt.xlabel("x") # 그래프 좌측(x축)의 label을 x로 설정
plt.ylabel("y_noise, y_pred") # 그래프의 아래(y축)의 label을 y_pred로 설정
plt.plot(x, y_pred, "r-", label="y_pred") # 그래프에 y_pred를 빨간선으로 구현
plt.grid(True)
# 그래프의 타이틀 즉 제목을 설정
plt.title("y_noise = {:+.2f}*x**3 {:+.2f}*x**2 {:+.2f}*x {:+.2f} + noise".format(a, b, c, d))
plt.legend(loc="best") # 범례 설정
plt.show() # 그래프 보여주기
