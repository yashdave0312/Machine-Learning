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