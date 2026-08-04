import numpy as np
from scipy import stats 

sample1 = [172,174,168,169,171,173,175,170,169,172]
sample2 = [165,167,169,170,168,166,164,167,169,165]

mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
print("Mean of sample 1:", mean1)
print("Mean of sample 2:", mean2)

std1 = np.std(sample1,ddof = 1)
std2 = np.std(sample2,ddof = 1)
print("Standard deviation of sample 1:", std1)
print("Standard deviation of sample 2:", std2)

n1 = len(sample1)
n2 = len(sample2)
print("Sample size of sample 1:", n1)
print("Sample size of sample 2:", n2)

t_test = (mean1 - mean2) /(np.sqrt((std1**2/n1) + (std2**2/n2)))
print("T-test statistic:", t_test)

ddof1 = n1 - 1
ddof2 = n2 - 1
print("Degrees of freedom for sample 1:", ddof1)
print("Degrees of freedom for sample 2:", ddof2)

p_value = 2 * (1 - stats.t.cdf(abs(t_test), df=min(ddof1, ddof2)))
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("I will reject the null hypothesis.")
else:
    print("I will accept the null hypothesis.")
