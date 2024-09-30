#23B1032-SRIKAR NIHAL
#23B0924-B.SOMANATH VASANTH
#23B1055-ANIRUDH
#Question 2 Task C
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def sample(loc, scale):
    uniform_random_variable_between_zero_and_one = np.random.uniform(low=0, high=1)
    inverseofguassian = norm.ppf(uniform_random_variable_between_zero_and_one, loc, scale)
    return inverseofguassian

# Generating the samples
sample1 = [sample(0, np.sqrt(0.2)) for _ in range(100000)]
sample2 = [sample(0, np.sqrt(1.0)) for _ in range(100000)]
sample3 = [sample(0, np.sqrt(5.0)) for _ in range(100000)]
sample4 = [sample(-2, np.sqrt(0.5)) for _ in range(100000)]

# Plotting normalized histograms
plt.figure(figsize=(10, 8))

plt.hist(sample1, bins=100, density=True, alpha=0.6, color='blue', label='MEAN=0, VARIANCE=0.2')
plt.hist(sample2, bins=100, density=True, alpha=0.6, color='red', label='MEAN=0, VARIANCE=1.0')
plt.hist(sample3, bins=100, density=True, alpha=0.6, color='yellow', label='MEAN=0, VARIANCE=5.0')
plt.hist(sample4, bins=100, density=True, alpha=0.6, color='green', label='MEAN=-2, VARIANCE=0.5')

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Normalized Histograms of Gaussian Samples')
plt.legend()
plt.grid(True)

# Saving the plot
plt.savefig('2C.png')

# Displaying the plot

