import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.

##############################################################################################################################

# setup(GUI = False, render_delay_sec = 0.1, gs = 11)
game = ShapePlacementGrid(GUI=True, render_delay_sec=0.001, gs=6, num_colored_boxes=5)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")


##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

# shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = execute('export')
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)
# input()   # <-- workaround to prevent PyGame window from closing after execute() is called, for when GUI set to True. Uncomment to enable.
# print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.



##########################################
# Write all your code in the area below. 
##########################################


import random

def solve_shapeshifting(grid_size):
    global grid 

    # Re-wrote masking part of this code using Claude
    def checkGrid():
        for color in range(4):
            mask = (grid == color)
            
            if np.any(mask[:-1, :] & mask[1:, :]):  
                return False
            if np.any(mask[1:, :] & mask[:-1, :]): 
                return False
            
            if np.any(mask[:, :-1] & mask[:, 1:]): 
                return False
            if np.any(mask[:, 1:] & mask[:, :-1]): 
                return False

        return True


    def iterative_search():
        global grid

        for _ in range(3):
            game.execute("switchshape")

        while True:
            empty_cells = get_empty_cells()

            if not empty_cells:
                return checkGrid()

            for target_x, target_y in empty_cells:

                if grid[target_y, target_x] != -1: continue
                
                for shape in range(9):
                    for color in range(4):
                        current_x, current_y = shapePos  

                        move_to_target(current_x, current_y, target_x, target_y)
                        game.execute('place')

                        grid = game.execute("export")[3]

                        if checkGrid():
                            break

                        game.execute("undo")
                        game.execute('switchcolor') 

                    game.execute('switchshape')

            remaining_empty_cells = get_empty_cells()
            if not remaining_empty_cells:
                return checkGrid()


    def get_empty_cells():
        return [(j, i) for i in range(grid_size) for j in range(grid_size) if grid[i][j] == -1]

    def move_to_target(current_x, current_y, target_x, target_y):
        global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done

        while (current_x, current_y) != (target_x, target_y):
            if current_y < target_y:
                game.execute("down")
                current_y += 1
            elif current_y > target_y:
                game.execute("up")
                current_y -= 1

            if current_x < target_x:
                game.execute("right")
                current_x += 1
            elif current_x > target_x:
                game.execute("left")
                current_x -= 1

        shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

        return current_x, current_y

    shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
    return iterative_search()


solve_shapeshifting(len(grid))
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
