import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


np.random.seed(20)
data = [np.random.normal(0, 1, 200) ,np.random.normal(0, 2, 200)]


plt.figure(figsize=(10,10))
sns.violinplot(data=data)


plt.xlabel('sample data')
plt.ylabel('Value')
plt.title('Violin Plot')


plt.show()