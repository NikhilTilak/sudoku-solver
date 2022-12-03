import numpy as np
import matplotlib.pyplot as plt

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
        plt.show()

def box(s: sudoku, i: int, j: int)-> np.array:
    """ returns the 3x3 submatrix in which the (i,j) lie"""
    x = i//3
    y = j//3    
    return s.state[3*x: 3*(x+1), 3*y: 3*(y+1)]


def valid_numbers(s: sudoku):
    """Looks at the present state of the sudoku and generates a 2D arrray 
    containing a list of potential numbers in every cell"""

    valid_numbers = {}
    
    for i in range(9):
        for j in range(9):
            if s.state[i,j]==0: # empty cell
                
                valid = list(range(1,10)) # all possible values

                for element in s.state[i,:]: # checking all the elements in the row i
                    if element!=0:
                        valid.remove(element) # removing already filled in numbers 
                        
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
                            
                valid_numbers[f"{i}{j}"]=valid # dict containing all the valid moves
       
    return valid_numbers
                
    
def update(s: sudoku):
    """Helper function which updates the state of the sudoku after each interation of solving"""
    update_matrix = np.zeros((9,9))
    for key, val in valid_numbers(s).items():
        if len(val)==1:
            i, j = int(key[0]), int(key[1])
            update_matrix[i,j] = int(val.pop(0))
    different_from_before = (update_matrix!=np.zeros((9,9))).any()
    s.update_state(s.state + update_matrix)
    return different_from_before

def solve_sudoku(s: sudoku):
    s_initial = sudoku(s.state)
    """Solve Sudoku """
    if not isinstance(s, sudoku):
        raise TypeError("Your input is not a sudoku object.")
    for i in range(100):
        diff = update(s)
        if not diff:
            break
    if i>0 and not ((s.state==0).any()):
        print(f"solved in {i} turns")
        plot_sudoku(s, s_initial)
    else: 
        print(f" beep boop, beep boop. Cannot compute. beep boop, beep boop.")
          