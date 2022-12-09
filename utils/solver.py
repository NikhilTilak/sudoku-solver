import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pathlib
import os
from itertools import product, combinations
# TEST_IMAGES = pathlib.Path.cwd().parent.joinpath("test_images")
TEST_IMAGES = pathlib.Path("D:\Data Science\sudoku-solver\\test_images")

# make a class for sudoku board
class sudoku:
    def __init__(self, state: np.array):
        """ the state of the sudoku is a 9x9 array containing numbers from 0-9.
        0 represents an empty cell. """
        sudoku.state = state.astype(int)
        
    def __str__(self):
        return str(self.state)
    
    def __repr__(self):
        return str(self.state)
        
    def is_valid(self):
        # TODO : add checks to make sure the Sudoku is valid
        return True
    
    def update_state(self, new_state):
        self.state = new_state.astype(int)

    
def box(s: sudoku, i: int, j: int)-> np.array:
    """ returns the 3x3 submatrix in which the (i,j) lie"""
    x = i//3
    y = j//3    
    return s.state[3*x: 3*(x+1), 3*y: 3*(y+1)]

def get_row(n: int)-> list:
    return [(n,j) for j in range(9)]

def get_col(n: int)-> list:
    return [(j,n) for j in range(9)]

def get_box(n: int)-> list:
    rows_in_box = range(int(n/3)*3,(1+int(n/3))*3)
    cols_in_box = range(((n%3))*3, ((n%3)+1)*3)
    return list(product(rows_in_box, cols_in_box))


def plot_sudoku(s: sudoku, s_init=sudoku(np.zeros((9,9)))):
        fig, ax = plt.subplots(9,9, figsize=(5,5))
        for i in range(9):
            for j in range(9):
                if s.state[i,j]==0: text='' 
                else: text= str(s.state[i,j])
                if s_init.state[i,j]==0: #initially empty cell
                    ax[i,j].text(0.5,0.5,text, ha='center', va='center', c='b', font='Arial', size=15)
                else:
                    ax[i,j].text(0.5,0.5,text, ha='center', va='center', c='k', font='Arial', size=15)
                ax[i,j].set_aspect('equal')
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                ax[i,j].axes.xaxis.set_ticklabels([])
                ax[i,j].axes.yaxis.set_ticklabels([])
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        temp_filepath = str(TEST_IMAGES.joinpath("solution_tempfile.jpg"))
        fig.savefig(temp_filepath, bbox_inches='tight', pad_inches = 0)
        plt.close()
        sol_img = cv.imread(temp_filepath)
        os.remove(temp_filepath)
        return sol_img

def valid_numbers(s: sudoku):
    """Looks at the present state of the sudoku and generates a 2D arrray 
    containing a list of potential numbers in every cell"""

    valid_numbers = {}
    
    # Naive pass
    for i in range(9):
        for j in range(9):
            if s.state[i,j]==0: # empty cell
                
                valid = list(range(1,10)) # all possible values

                for element in s.state[i,:]: # checking all the elements in the row i
                    if element!=0:
                        try:
                            valid.remove(element)
                        except ValueError:
                            continue 
                        
                for element in s.state[:,j]: # checking all the elements in the column j
                    if element!=0:
                        try:
                            valid.remove(element)
                        except ValueError:
                            continue
                            
                for element in box(s,i,j).ravel(): # checking all the elements in the 3x3 sub-matrix containing (i,j)
                    if element!=0:
                        try:
                            valid.remove(element)
                        except ValueError:
                            continue                           
                valid_numbers[(i,j)]=valid # dict containing all the valid moves

            else: #already filled cell
                valid_numbers[(i,j)]=[s.state[i,j]]

    ################Find pairs##################

    val_old = valid_numbers.copy()

    # Checking for pair in rows
    for row in range(9):
        cells = get_row(row) # all the cells in row
        pairs = [comb for comb in combinations(cells,2)]
        for pair in pairs:
            combined_list = valid_numbers[pair[0]]+valid_numbers[pair[1]]
            combined_set = set(combined_list)
            if len(combined_set)==2 and len(combined_list)==4: # only 2 unique numbers in the pair of cells
                for cell in cells:
                    if cell not in pair:
                        for num in combined_set:
                            try:
                                valid_numbers[cell].remove(num)
                            except ValueError:
                                continue
    # Checking for pair in columns
    for col in range(9):
        cells = get_col(col) # all the cells in row
        pairs = [comb for comb in combinations(cells,2)]
        for pair in pairs:
            combined_list = valid_numbers[pair[0]]+valid_numbers[pair[1]]
            combined_set = set(combined_list)
            if len(combined_set)==2 and len(combined_list)==4: # only 2 unique numbers in the pair of cells
                for cell in cells:
                    if cell not in pair:
                        for num in combined_set:
                            try:
                                valid_numbers[cell].remove(num)
                            except ValueError:
                                continue
    # Checking for pair in box
    for bx in range(9):
        cells = get_box(bx) # all the cells in row
        pairs = [comb for comb in combinations(cells,2)]
        for pair in pairs:
            combined_list = valid_numbers[pair[0]]+valid_numbers[pair[1]]
            combined_set = set(combined_list)
            if len(combined_set)==2 and len(combined_list)==4:# only 2 unique numbers in the pair of cells
                for cell in cells:
                    if cell not in pair:
                        for num in combined_set:
                            try:
                                valid_numbers[cell].remove(num)
                            except ValueError:
                                continue
    
    ###############Find triples##################

    # Checking for triples in rows
    for row in range(9):
        cells = get_row(row) # all the cells in row
        triples = [comb for comb in combinations(cells,3)]
        for triple in triples:
            combined_list = valid_numbers[triple[0]]+valid_numbers[triple[1]]+valid_numbers[triple[2]]
            combined_set = set(combined_list)
            if len(combined_set)==3 and len(combined_list)>=6: # only 3 unique numbers in the triple of cells
                for cell in cells:
                    if cell not in triple:
                        for num in combined_set:
                            try:
                                valid_numbers[cell].remove(num)
                            except ValueError:
                                continue
    
    # Checking for triples in cols
    for col in range(9):
        cells = get_row(col) # all the cells in row
        triples = [comb for comb in combinations(cells,3)]
        for triple in triples:
            combined_list = valid_numbers[triple[0]]+valid_numbers[triple[1]]+valid_numbers[triple[2]]
            combined_set = set(combined_list)
            if len(combined_set)==3 and len(combined_list)>=6: # only 3 unique numbers in the triple of cells
                for cell in cells:
                    if cell not in triple:
                        for num in combined_set:
                            try:
                                valid_numbers[cell].remove(num)
                            except ValueError:
                                continue
    # Checking for triples in box
    for bx in range(9):
        cells = get_box(bx) # all the cells in row
        triples = [comb for comb in combinations(cells,3)]
        for triple in triples:
            combined_list = valid_numbers[triple[0]]+valid_numbers[triple[1]]+valid_numbers[triple[2]]
            combined_set = set(combined_list)
            if len(combined_set)==3 and len(combined_list)>=6: # only 3 unique numbers in the triple of cells
                for cell in cells:
                    if cell not in triple:
                        for num in combined_set:
                            try:
                                valid_numbers[cell].remove(num)
                            except ValueError:
                                continue
    
    return valid_numbers
                

def solve_sudoku(s: sudoku, verbose=False):
    s_input=sudoku(s.state)
    iteration=0
    solved = (s.state!=0).all()
    max_iter=100

    while not solved and (iteration<=max_iter):
        iteration+=1
        if verbose:print(20*"#"+f"Iter: {iteration} "+ 20*"#")

        state_initial = s.state.copy()
        update_matrix = s.state.copy()


        val_nums = valid_numbers(s)
        # if verbose:
        #     for key, val in val_nums.items():
        #         print(f"{key}:{val}\n")

    ################# UNIQUE check #############################
        for key, val in val_nums.items():
            if len(val)==1:
                i, j = int(key[0]), int(key[1])
                update_matrix[i,j] = int(val[0])

        if verbose:
            print("uniqueness check")
            print(update_matrix-s.state)

    ################# ROW check #############################
        for row in range(9):
            cells = get_row(row)
            for digit in range(1,10):
                digit_count=0
                for cell in cells:
                    if digit in val_nums[cell]:
                        digit_count+=1
                        location=cell
                if digit_count==1: #digit can be in excatly one place in this row
                        update_matrix[location]=digit

        if verbose:
            print("row check")
            print(update_matrix-s.state)

    ################# COL check #############################
        for col in range(9):
            cells = get_col(col)
            for digit in range(1,10):
                digit_count=0
                for cell in cells:
                    if digit in val_nums[cell]:
                        digit_count+=1
                        location=cell
                if digit_count==1: #digit can be in excatly one place in this column
                        update_matrix[location]=digit
        if verbose:
            print("col check")
            print(update_matrix-s.state)

    ################# 3x3 BOX check #############################
        for box in range(9):
            cells = get_box(box)
            for digit in range(1,10):
                digit_count=0
                for cell in cells:
                    if digit in val_nums[cell]:
                        digit_count+=1
                        location=cell
                if digit_count==1: #digit can be in excatly one place in this 3x3 box
                        update_matrix[location]=digit
        if verbose:
            print("3x3 box check")
            print(update_matrix-s.state)
##############################################################################################
        s.update_state(update_matrix)

        if verbose:
            print(f"s after iteration {iteration}")
            print(s.state)

        solved = (s.state!=0).all()
        if solved:
            if verbose:print(f"Solved after iteration {iteration} ? : {solved}")
            return plot_sudoku(s, s_input)

        changed = (update_matrix!=state_initial).any()
        if verbose:print(f"Changed after iteration {iteration} ? : {changed}")
        
        if not changed:
            if verbose:print("I tried my best but this one's hard.")
            return plot_sudoku(s, s_input)
        

#----------------------------------------------------------------------------
if __name__=="__main__":

    # input_array = np.array(
    # [[2, 0, 6, 5, 0, 0, 4, 1, 9],
    # [9, 0, 5, 0, 0, 2, 0, 0, 8],
    # [0, 0, 0, 3, 9, 4, 6, 0, 5],
    # [0, 5, 7, 4, 0, 0, 0, 6, 0],
    # [0, 1, 0, 9, 0, 8, 0, 4, 0],
    # [4, 0, 0, 0, 5, 0, 3, 8, 0],
    # [1, 7, 4, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 6, 7, 9, 0, 0, 0],
    # [0, 0, 9, 0, 4, 1, 8, 0, 3]]) #easy


    input_array = np.array(
    [[0, 0, 6, 0, 0, 0, 0, 1, 2],
     [3, 0, 8, 0, 5, 0, 0, 0, 0],
     [0, 0, 0, 7, 0, 4, 0, 0, 0],
     [5, 0, 0, 6, 2, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 7, 0, 0, 0],
     [4, 1, 0, 0, 8, 0, 0, 0, 0],
     [0, 9, 0, 5, 0, 0, 0, 8, 7],
     [8, 0, 0, 0, 0, 2, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 6, 0, 9]]) # medium

    
    # input_array = np.array(
    # [[0, 3, 9, 5, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 8, 0, 0, 0, 7, 0],
    #  [0, 0, 0, 0, 1, 0, 9, 0, 4],
    #  [1, 0, 0, 4, 0, 0, 0, 0, 3],
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 7, 0, 0, 0, 8, 6, 0],
    #  [0, 0, 6, 7, 0, 8, 2, 0, 0],
    #  [0, 1, 0, 0, 9, 0, 0, 0, 5],
    #  [0, 0, 0, 0, 0, 1, 0, 0, 8]]) # shortz301_hard

    # input_array = np.array(
    # [[4, 0, 0, 0, 0, 0, 9, 3, 8],
    #  [0, 3, 2, 0, 9, 4, 1, 0, 0],
    #  [0, 9, 5, 3, 0, 0, 2, 4, 0],
    #  [3, 7, 0, 6, 0, 9, 0, 0, 4],
    #  [5, 2, 9, 0, 0, 1, 6, 7, 3],
    #  [6, 0, 4, 7, 0, 3, 0, 9, 0],
    #  [9, 5, 7, 0, 0, 8, 3, 0, 0],
    #  [0, 0, 3, 9, 0, 0, 4, 0, 0],
    #  [2, 4, 0, 0, 3, 0, 7, 0, 9]]) # pair example

    s = sudoku(input_array)

    # print(valid_numbers(s))
    plt.imshow(solve_sudoku(s))
    plt.show()
 

    # val_nums = valid_numbers(s)
    # for key, val in val_nums.items():
    #     if len(val)==1: print(f"{key}:{val}\n")




    