import numpy as np
import matplotlib.pyplot as plt


categories = ['c1', 'c2', 'c3', 'c4', 'c5']
values = [4, 22, 15, 17, 25]
angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()


values.append(values[0])
angles.append(angles[0])


fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))


bars = ax.bar(angles, values, width=0.5, color='skyblue', edgecolor='black')


ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)


ax.set_title('Coxcomb Plot', va='bottom')

plt.show()
