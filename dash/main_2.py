# - * -coding: utf - 8 - * -
import base64
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


import pandas as pd
import numpy as np
import numpy as np

import cv2


# prepare the data--begin

cases = pd.read_csv('../data/valid-acl.csv',
                    header=None,
                    names=['Case', 'Abnormal'],
                    dtype={
                        'Case': str,
                        'Abnormal': np.int64
                    }
                    )
case_list = cases['Case'].tolist()


predictions = pd.read_csv('./val_data.csv')


# prepare the data--end


app = dash.Dash()

# Boostrap CSS.

app.css.append_css({
    'external_url': "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
})  # noqa: E501


app.layout = html.Div(
    [
        html.Div(
            html.H1(children='Interpretation of MRNet models through Class Activation Maps (CAM)',
                    style={
                        'text-align': 'center'
                    }
                    ),
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P('Select an MRI exam:'),
                        html.Div([
                            dcc.Dropdown(
                                id='cases',
                                options=[{
                                    'label': case,
                                    'value': case
                                }
                                    for case in case_list
                                ],
                                placeholder="Pick a case",
                                clearable=False
                            )
                        ],
                            style={
                            'margin-bottom': 20
                        }
                        ),
                    ],
                    className='col-3',
                    style={
                        'margin-top': '10'
                    }
                ),
                html.Div([
                    html.Div([
                        html.Div([
                            html.P('Select true labels :'),
                            dcc.RadioItems(
                                id="label_radioitems",
                                options=[
                                    {'label': 'Positive (ACL tear)',
                                     'value': 'acl'},
                                    {'label': 'Negative (Normal)',
                                     'value': 'normal'},
                                ],
                                value='acl',
                                labelStyle={'display': 'inline-block'}
                            ),
                        ],
                            style={'float': 'left', 'width': '45%'}
                        ),

                        html.Div([
                            html.P('Select predicted labels :'),
                            dcc.RadioItems(
                                id="pred_radioitems",
                                options=[
                                    {'label': 'Positive (ACL tear)',
                                     'value': 'acl'},
                                    {'label': 'Negative (Normal)',
                                     'value': 'normal'},
                                ],
                                value='acl',
                                labelStyle={'display': 'inline-block'}
                            ),
                        ]
                        ),
                    ]),
                ],
                    className='col-6'
                ),
                html.Div([
                    html.Div(id="number_of_cases"),
                    html.Span(
                        "txt",
                        id="label_badge",
                        className="badge badge-success badge-large",
                        style={'font-size': '15px'}
                    ),
                ],
                    className='col-3'

                )

            ],
            className="row alert alert-success",
        ),

        html.Div([
            html.Div([
                html.P(id='summary', style={'font-size': '20px'}),
                html.Div([
                    html.Div('This probability is a weighted average of the three probabilities of tears over each plane', style={
                        'float': 'left', 'font-size': '20px'}),
                    html.Div('Slide over the slices of each MRI to inspect highlighted regions of tear as depicted by CAMs', style={
                        'float': 'left', 'font-size': '20px'}),
                ],
                )
            ],
                className="col-12 alert alert-info"),
        ],
            className='row'
        ),

        html.Hr(),

        html.Div(
            [
                html.Div([
                    html.Div([
                        dcc.Slider(id='slider_axial', updatemode='drag')
                    ],
                        style={'margin-right': '5px'}

                    ),
                    html.Hr(),
                    html.P(id="score_axial", style={'text-align': 'center'}),
                    html.Div([
                        html.Div([
                            html.Img(
                                id="slice_axial",
                            ),
                        ],
                            style={'float': 'left', 'margin-right': '5px'}
                        ),
                        html.Div([
                            html.Img(
                                id="cam_axial",
                            ),
                        ],
                        )
                    ],

                    ),
                    html.P(id="title_axial", style={'text-align': 'center'})
                ],
                    className="col-4"
                ),
                html.Div([
                    html.Div([
                        dcc.Slider(id='slider_coronal', updatemode='drag')
                    ],
                        style={'margin-right': '5px'}

                    ),
                    html.Hr(),
                    html.P(id="score_coronal", style={'text-align': 'center'}),
                    html.Div([
                        html.Div([
                            html.Img(
                                id="slice_coronal",
                            ),
                        ],
                            style={'float': 'left', 'margin-right': '5px'}
                        ),
                        html.Div([
                            html.Img(
                                id="cam_coronal",
                            ),
                        ],
                        )
                    ],

                    ),
                    html.P(id="title_coronal", style={'text-align': 'center'})
                ],
                    className="col-4"
                ),
                html.Div([
                    html.Div([
                        dcc.Slider(id='slider_sagittal', updatemode='drag')
                    ]),
                    html.Hr(),
                    html.P(id="score_sagittal", style={
                           'text-align': 'center'}),
                    html.Div([
                        html.Div([
                            html.Img(
                                id="slice_sagittal",
                            ),
                        ],
                            style={'float': 'left', 'margin-right': '5px'}
                        ),
                        html.Div([
                            html.Img(
                                id="cam_sagittal",
                            ),
                        ],
                        )
                    ],

                    ),
                    html.P(id="title_sagittal", style={'text-align': 'center'})


                ],
                    className="col-4"
                ),

            ],
            className='row'
        )






    ],
    className='container-fluid',
)


if __name__ == '__main__':
    app.run_server(debug=True, port=8060)
