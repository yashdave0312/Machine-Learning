import numpy as np
from scipy.stats import norm 

sample = [172,174,168,169,171,173,175,170,169,172]
pop_mean = 170
pop_std = 3
samp_mean = np.mean(sample)
n = len(sample)
z_score = (samp_mean - pop_mean) / (pop_std / np.sqrt(n)) 
print("Z-score:", z_score)

p_value = 2 * (1 - norm.cdf(abs(z_score)))
print("P-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("I will reject the null hypothesis.")   
else:
    print("I will accept the null hypothesis.")   
