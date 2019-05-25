# Introducing callbacks

# -*- coding: utf-8 -*-
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


import pandas as pd
import numpy as np


# prepare the data -- begin

cases = pd.read_csv('../data/train-abnormal.csv',
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
                html.H1(children='MRI Analysis',
                        className='nine columns'),
                html.Div(children='''
                         A web application framework for knee MRI analysis. 
                        ''',
                         className='nine columns'
                         )
            ], className="row"
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.P('Select a medical case (i.e. a patient)'),
                        html.Div([
                            dcc.Dropdown(
                                id='cases',
                                options=[
                                    {'label': case, 'value': case} for case in case_list
                                ],
                                value='0000',
                                placeholder="Pick a case",
                            )
                        ],
                            style={'margin-bottom': 20}
                        ),
                        html.P('Select an mri slice'),
                        html.Div([
                            dcc.Slider(
                                id='slider',
                            )],
                        )
                    ],
                    className='six columns',
                    style={'margin-top': '10'}
                ),

                html.Div(
                    [
                        dcc.RadioItems(
                        options=[
                            {'label': 'Abnormal', 'value': 'NYC'},
                            {'label': 'ACL', 'value': 'MTL'},
                            {'label': 'Meniscus', 'value': 'SF'}
                        ],
                        value='MTL',
                        # labelStyle={'display': 'inline-block'}
                    )],
                    className='six columns'
                )

            ], className="row"
        ),

        html.Hr(),

        html.Div(
            [
                html.Div([
                    dcc.Graph(
                        id='heatmap_1',
                    )
                ], className='six columns'
                ),

                html.Div([
                    dcc.Graph(
                        id='heatmap_2',
                    )
                ], className='six columns'
                )
            ], className="row"
        )
    ], className='ten columns offset-by-one')
)


# slider update --- begin

@app.callback(
    dash.dependencies.Output('slider', 'value'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_value(selected_case):
    mri = np.load(f'../data/train/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2


@app.callback(
    dash.dependencies.Output('slider', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max(selected_case):
    mri = np.load(f'../data/train/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks(selected_case):
    mri = np.load(f'../data/train/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    marks = {str(i): '{}'.format(i) for i in range(number_slices)[::2]}
    return marks

# slider update --- end

# mri update slice --- begin


@app.callback(
    dash.dependencies.Output('heatmap_1', 'figure'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider', 'value')
    ]
)
def update_mri(selected_case, s):
    if selected_case is None:
        selected_case = '0000'
    if s is None:
        s = 0

    mri = np.load(f'../data/train/axial/{selected_case}.npy')[s]
    data = {
        'z': mri,
        'type': 'heatmap'
    }
    figure = {
        'data': [data],
    }
    return figure

# mri update slice --- end

# mri update heatmap --- begin


@app.callback(
    dash.dependencies.Output('heatmap_2', 'figure'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider', 'value')
    ]
)
def update_heatmap(selected_case, s):
    if selected_case is None:
        selected_case = '0000'
    if s is None:
        s = 0

    mri = np.load(f'../data/train/axial/{selected_case}.npy')[s]
    data = {
        'z': mri,
        'type': 'heatmap'
    }
    figure = {
        'data': [data],
    }
    return figure

# mri update slice --- end


if __name__ == '__main__':
    app.run_server(debug=True)
