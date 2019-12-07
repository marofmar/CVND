'''
Uniform Distribution
'''

# importing resources
import matplotlib.pyplot as plt
import numpy as np


# uniform distribution for 5 grid cells
# we use "p" to represent probability
p = [0.2, 0.2, 0.2, 0.2, 0.2]
print(p)

# probability distribution bar chart draw funtion
def display_map(grid, bar_width=1):
    if(len(grid) > 0):
        x_labels = range(len(grid))
        plt.bar(x_labels, height=grid, width=bar_width, color='b')
        plt.xlabel('Grid Cell')
        plt.ylabel('Probability')
        plt.ylim(0, 1) # range of 0-1 for probability values 
        plt.title('Probability of the robot being at each cell in the grid')
        plt.xticks(np.arange(min(x_labels), max(x_labels)+1, 1))
        plt.show()
    else:
        print('Grid is empty')

# uniform distribution generating function given the x_axis grid length.
def initialize_robot(grid_length):
    ''' Takes in a grid length and returns 
       a uniform distribution of location probabilities'''
    p = []
    p.append(float(1/grid_length))
    return p
