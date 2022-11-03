Thanks for checking out my Sudoku-Solver.

The challenge is to correctly solve a Sudoku given an image of the Sudoku.

The general algorithm of version 1.0 of the project is shown below. 

v1.0 only accepts a high quality digital image of the Sudoku from the top-down perspective.

![version 1]("D:\Data Science\sudoku-solver\solver_v1.png")


Currently I am working on version 2 where the image is taken from a any angle using a camera.

The user can then select the corners of the puzzle and a straightened image of the sudoku is generated using a perspective transformation algorithm.

At this stage the neural network model performs worse on data generated using this process. 

Improvements are ongoing. Keep an eye out for version 2.0.

![version 2]("D:\Data Science\sudoku-solver\solver_v2.png")

The project is built using numpy, skimage and tensorflow.Keras.

The neural network was trained on a [dataset](https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset) containing printed digits from Kaggle.