import matplotlib.pyplot as plt
import numpy as np


categories = ['c1', 'c2', 'c3', 'c4', 'c5']
values = [50, 15, 20, 5, 10]


sorted_indices = np.argsort(values)[::-1]
sorted_values = np.array(values)[sorted_indices]
sorted_categories = np.array(categories)[sorted_indices]


cumulative_percentage = []
total_sum = np.sum(sorted_values)
current_sum = 0
for value in sorted_values:
    current_sum += value
    cumulative_percentage.append(current_sum / total_sum * 100)
fig, ax1 = plt.subplots()
ax1.bar(sorted_categories, sorted_values, color='blue', alpha=0.5)
ax1.set_xlabel('Categories')
ax1.set_ylabel('Values', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(sorted_categories, cumulative_percentage, color='green', marker='o', linestyle='--')
ax2.set_ylabel('Cumulative Percentage (%)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax1.grid(True)
plt.title('Pareto Plot')
plt.show()
