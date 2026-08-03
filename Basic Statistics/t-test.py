import numpy as np
from scipy import stats

sample = [172,174,168,169,171,173,175,170,169,172]
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
pop_mean = 170
n = len(sample)

t_test = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))
print("T-test statistic:", t_test)

p_value = 2*(1-stats.t.cdf(abs(t_test), df=n-1))
print("P-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("I will reject the null hypothesis.")
else:
    print("I will accept the null hypothesis.")
    