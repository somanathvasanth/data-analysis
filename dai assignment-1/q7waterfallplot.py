import matplotlib.pyplot as plt
import numpy as np


categories = ['Start', 'Increase A', 'Decrease B', 'Increase C', 'Decrease D', 'Increase E', 'End']
changes = [100, 40, -30, 20, -10, 25, 0]  
sum=0
cumulative_values=[]
for i in {0,1,2,3,4,5,6}:
    sum=sum+changes[i]
    cumulative_values.append(sum)

initial_value = changes[0] 
cumulative_values = np.insert(cumulative_values, 0, initial_value)

fig, ax = plt.subplots(figsize=(10, 10))  


colors = ['green' if x >= 0 else 'red' for x in changes]


bars = ax.bar(categories, cumulative_values[1:] - cumulative_values[:-1], bottom=cumulative_values[:-1], color=colors)


for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:+}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom')


ax.grid(True, linestyle='--', alpha=0.7)
plt.title('Waterfall Plot')


plt.xticks(rotation=45, ha='right')


plt.tight_layout()
plt.show()

