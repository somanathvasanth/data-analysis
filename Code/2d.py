#23B1032-SRIKAR NIHAL
#23B0924-B.SOMANATH VASANTH
#23B1055-ANIRUDH
#Question 2 Task D
import numpy as np
import matplotlib.pyplot as plt

def normal_simulation_using_galton_table(h, no_of_balls):
    final_positions = []
    for _ in range(no_of_balls):
        random_variable_which_chooses_0_or_1 = np.random.randint(0, 2, size=h)
        choosing_direction = []
        for x in random_variable_which_chooses_0_or_1:
            if x == 1:
                choosing_direction.append(1)
            else:
                choosing_direction.append(-1)  
        position = np.sum(choosing_direction)
        final_positions.append(position)
    
    return final_positions
no_of_balls = 10**5  
final_positions = normal_simulation_using_galton_table(10, no_of_balls)
h=10
plt.hist(final_positions, bins=100, density=True, edgecolor='black')
plt.title(f'Galton Board Simulation (h={10})')
plt.xlabel('Final Position')
plt.ylabel('Normalized Count')
plt.grid(True)
plt.savefig("2d1.png")
plt.clf()
final_positions = normal_simulation_using_galton_table(50, no_of_balls)
h=50
plt.hist(final_positions, bins=100, density=True, edgecolor='black')
plt.title(f'Galton Board Simulation (h={50})')
plt.xlabel('Final Position')
plt.ylabel('Normalized Count')
plt.grid(True)
plt.savefig("2d2.png")
plt.clf()
final_positions = normal_simulation_using_galton_table(100, no_of_balls)
h=100
plt.hist(final_positions, bins=100, density=True, edgecolor='black')
plt.title(f'Galton Board Simulation (h={100})')
plt.xlabel('Final Position')
plt.ylabel('Normalized Count')
plt.grid(True)
plt.savefig("2d3.png")
