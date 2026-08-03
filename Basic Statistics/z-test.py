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
    print("Reject the null hypothesis: The sample mean is significantly different from the population mean.")   
else:
    print("Fail to reject the null hypothesis: The sample mean is not significantly different from the population mean.")   
    