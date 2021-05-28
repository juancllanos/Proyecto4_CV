import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
#

import base64
from PIL import Image
import numpy as np
import io
#
from utils import prediction_function


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):
    imgdata = base64.b64decode(contents)
    image = np.fromstring(imgdata,np.uint8)
    pred = prediction_function(image)

    return html.Div([
        html.H5('Prueba'),
        html.Img(src=contents),
        html.Div('La imgen ha sido clasificada como digito ' + str(pred)),
        html.Hr(),

    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)

