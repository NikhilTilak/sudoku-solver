
import os
import sys
import pathlib
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('.'))
print(module_path)

if module_path not in sys.path:
    sys.path.append(module_path)
from utils.processImage import get_subimage, process_image, get_all_subimages
from utils.solver import sudoku, solve_sudoku, plot_sudoku

import base64
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import transform
import cv2
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State


app = Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        className="app-header",
        children=html.H1([
            'Drag and Drop or ',
            html.A('Select Files')
        ], className='.app-header--title'),
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div([html.H2('Uploaded Image', className='column'),
    html.H2('Interpreted Image', className='column'),
    html.H2('Solution', className='column')]),

    html.Div([html.Div(id='output-image-upload', className='column'),
    html.Div(id='processed-image', className='column'),
    html.Div(id='solution', className='column')])


])

@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'))
def update_output(contents):
    if contents is not None:
        return html.Img(src=contents)

@app.callback(Output('processed-image', 'children'),
              Input('upload-image', 'contents'),
              prevent_initial_call=True
)
def update_processed(contents):
    if contents is not None:
        if contents[11]=='p':
            extension='png'
            bound = 22
        if contents[11]=='j':
            extension='jpeg'
            bound=23
        filename = "image"+f".{extension}"
        with open(filename, "wb") as fh:
            fh.write(base64.b64decode(contents[bound:]))
        image = imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.fastNlMeansDenoising(np.uint8(image))

        simges = get_all_subimages(image)

        _, buffer = cv2.imencode(f".{extension}", simges)
        modified_contents = base64.b64encode(buffer).decode('utf-8')
        modified_contents = contents[:bound] + modified_contents
        return html.Img(src=modified_contents)

@app.callback(Output('solution', 'children'),
              Input('upload-image', 'contents'),
              prevent_initial_call=True
)
def update_solution(contents):
    if contents is not None :
        if contents[11]=='p':
            extension='png'
            bound = 22
        if contents[11]=='j':
            extension='jpeg'
            bound=23
        filename = "image"+f".{extension}"
        with open(filename, "wb") as fh:
            fh.write(base64.b64decode(contents[bound:]))
        image = imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.fastNlMeansDenoising(np.uint8(image))

        s = process_image(image)
        print(s)
        solution_image = solve_sudoku(s)
        
        _, buffer = cv2.imencode(".jpg", solution_image)
        modified_contents = base64.b64encode(buffer).decode('utf-8')
        modified_contents = contents[:bound] + modified_contents
        return html.Img(src=modified_contents)
        # return str(s.state)

if __name__ == '__main__':
    app.run_server(debug=True)
