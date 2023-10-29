# Homework 2.2
# Author : LeeSeungJun
# Student ID : 22312050
# Dept : Info & Comm. Eng
# Date : March 14, 2023
# Major features :
# - Input 10 float data and save list
# - Print Average , Variance , Standard Deviation of 10 float data

# 표준편차를 구하기 위한 math 모듈 imprt
import math

# 10개의 실수 데이터를 문자열 형태로 입력
data = (input("10개의 실수 데이터를 입력해주세요 : "))

# split을 사용해서 data 문자열을 나누기
float_strings = data.split()

# split을 사용해서 나눈 data 문자열을 출력
print("float_string : {}".format(float_strings))

# 나눈 문자열을 실수형 데이터로 변환
l_data = list(map(float, float_strings))

# 실수형 리스트 데이터를 출력
print("리스트 데이터 = {} ".format(l_data))

# 리스트 데이터의 평균 계산
avg = sum(l_data) / len(l_data)

# 리스트 데이터의 분산 계산 각 리스트에서 평균을 뺸 값을 제곱하고,
# 나온 결과를 모두 더한후 리스트의 전체 개수(10개)로 나눠서 구함
val = 0
for i in l_data:
    val = val + (i - avg)**2
variance = val / len(l_data)

# 표준편차는 분산의 제곱근 계산으로 구함, sqrt = 제곱근
standard_deviation = math.sqrt(variance)

# 다 구한 평균과 분산 표준편차를 출력
print("평균 = {}, 분산 = {}, 표준편차 = {}".format(avg, variance, standard_deviation))
