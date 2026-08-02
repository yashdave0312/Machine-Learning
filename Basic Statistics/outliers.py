import numpy as np

arr =np.array([2,3,4,6,7,8,9,12,13,16,17,23,25,27,34,37,201])
print("Original Array:", arr)

Q1=np.percentile(arr,25)
print("Q1:", Q1)

Q3=np.percentile(arr,75)
print("Q3:", Q3)

IQR=Q3-Q1
print("IQR:", IQR)

Upper_limit=Q3+1.5*IQR
print("Upper limit:", Upper_limit)

Lower_limit=Q1-1.5*IQR
print("Lower limit:", Lower_limit)

Outliers=[]
for i in arr:
    if(i>Upper_limit or i<Lower_limit):
        Outliers.append(i)

print("Outliers in the given data:", Outliers)        