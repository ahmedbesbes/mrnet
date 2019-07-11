# Introducing callbacks

# -*- coding: utf-8 -*-
import base64
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


import pandas as pd
import numpy as np

import cv2


# prepare the data -- begin

cases = pd.read_csv('../data/valid-acl.csv',
                    header=None,
                    names=['Case', 'Abnormal'],
                    dtype={'Case': str, 'Abnormal': np.int64}
                    )
case_list = cases['Case'].tolist()

# prepare the data -- end


app = dash.Dash()

# Boostrap CSS.

app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501


app.layout = html.Div(
    html.Div([
        html.Div(
            [
                html.H1(children='Knee MRI Analysis', className='nine columns')
            ], 
            className="row"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P('Select a medical case (i.e. a patient)'),
                        html.Div([
                            dcc.Checklist(
                                id='ground_truth'
                                
                            )
                        ]),
                        html.Div([
                            dcc.Dropdown(
                                id='cases',
                                options=[
                                    {'label': case, 'value': case} for case in case_list
                                ],
                                value='1130',
                                placeholder="Pick a case",
                                clearable=False

                            )
                        ],
                            style={'margin-bottom': 20}
                        ),
                    ],
                    className='six columns',
                    style={'margin-top': '10'}
                ),
            ], className="row"
        ),

        html.Hr(),

        html.Div(
            [
                html.Div([
                    html.Div([
                        dcc.Slider(id='slider_axial')
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Div([
                                html.Img(
                                    id="slice_axial",
                                    width='250',
                                    height='250',
                                ),
                            ], 
                            style={'margin-right': '15', 'float': 'left'}
                        ),
                        html.Div([
                                html.Img(
                                    id="cam_axial",
                                    width='250',
                                    height='250',
                                ),
                            ], 
                            # style={'margin-right': '5'}
                        )
                    ])
                    ],
                    className="four columns"
                ),
                html.Div([
                    html.Div([
                        html.Img(id="slice_coronal"),
                        html.Img(id="cam_coronal")
                    ]),
                    html.Div([
                        dcc.Slider(id='slider_coronal')
                    ])],
                    className="four columns"
                ),
                html.Div([
                    html.Div([
                        html.Img(id="slice_sagittal"),
                        html.Img(id="cam_sagittal")
                    ]),
                    html.Div([
                        dcc.Slider(id='slider_sagittal')
                    ])],
                    className="four columns"
                ),

            ],
            className='row'
        )
    ], className='ten columns offset-by-one')
)


# update axial slider --- begin
@app.callback(
    dash.dependencies.Output('slider_axial', 'value'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_value(selected_case):
    mri = np.load(f'../data/valid/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2

@app.callback(
    dash.dependencies.Output('slider_axial', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max(selected_case):
    mri = np.load(f'../data/valid/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider_axial', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks(selected_case):
    mri = np.load(f'../data/valid/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    marks = {str(i): '{}'.format(i) for i in range(number_slices)[::2]}
    return marks

# update axial slider --- end

# update coronal slider --- begin
@app.callback(
    dash.dependencies.Output('slider_coronal', 'value'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_value(selected_case):
    mri = np.load(f'../data/valid/coronal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2

@app.callback(
    dash.dependencies.Output('slider_coronal', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max(selected_case):
    mri = np.load(f'../data/valid/coronal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider_coronal', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks(selected_case):
    mri = np.load(f'../data/valid/coronal/{selected_case}.npy')
    number_slices = mri.shape[0]
    marks = {str(i): '{}'.format(i) for i in range(number_slices)[::2]}
    return marks

# update coronal slider --- end

# update sagittal slider --- begin
@app.callback(
    dash.dependencies.Output('slider_sagittal', 'value'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_value(selected_case):
    mri = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2

@app.callback(
    dash.dependencies.Output('slider_sagittal', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max(selected_case):
    mri = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider_sagittal', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks(selected_case):
    mri = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    number_slices = mri.shape[0]
    marks = {str(i): '{}'.format(i) for i in range(number_slices)[::2]}
    return marks

# update sagittal slider --- end

# update slice axial --- begin
@app.callback(
    dash.dependencies.Output('slice_axial', 'src'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_axial', 'value'),

    ])
def update_slice_axial(selected_case, selected_slice):
    s = np.load(f'../data/valid/axial/{selected_case}.npy')[selected_slice]
    cv2.imwrite(f'./slice_axial.png', s)
    encoded_image = base64.b64encode(open('./slice_axial.png', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())
# update slice axial --- end

# update cam axial --- begin
@app.callback(
    dash.dependencies.Output('cam_axial', 'src'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_axial', 'value'),

    ])
def update_cam_axial(selected_case, selected_slice):
    selected_case = int(selected_case) - 1130
    selected_case = '0' * (4 - len(str(selected_case))) + str(selected_case) 
    src = f'./CAMS/axial/{selected_case}/cams/{selected_slice}.png'
    encoded_image = base64.b64encode(open(src, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


# update slice axial --- end



if __name__ == '__main__':
    app.run_server(debug=True)
