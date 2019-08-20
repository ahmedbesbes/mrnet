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
import numpy as np

import cv2


# prepare the data -- begin

cases = pd.read_csv('../data/valid-acl.csv',
                    header=None,
                    names=['Case', 'Abnormal'],
                    dtype={'Case': str, 'Abnormal': np.int64}
                    )
case_list = cases['Case'].tolist()


predictions = pd.read_csv('./val_data.csv')


# prepare the data -- end


app = dash.Dash(show_undo_redo=False)

# Boostrap CSS.

app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})  # noqa: E501
#app.css.append_css({'external_url': "https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css"})  # noqa: E501



app.layout = html.Div(
    html.Div([
        html.Div(
            
                html.H1(children='Interpretation of MRNet models through Class Activation Maps (CAM)', 
                        className='twelve columns',
                        style={'text-align': 'center'}                      
                    )
            ,
            className="row",
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
                                placeholder="Pick a case",
                                clearable=False
                            )
                        ],
                            style={'margin-bottom': 20}
                        ),
                    ],
                    className='three columns',
                    style={'margin-top': '10'}
                ),
                html.Div([
                    html.Div([
                        html.Div([
                            html.P('Select true labels :'),
                            dcc.RadioItems(
                                id="label_radioitems",
                                options=[
                                    {'label': 'Positive (ACL tear)', 'value': 'acl'},
                                    {'label': 'Negative (Normal)', 'value': 'normal'},
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
                                    {'label': 'Positive (ACL tear)', 'value': 'acl'},
                                    {'label': 'Negative (Normal)', 'value': 'normal'},
                                ],
                                value='acl',
                                labelStyle={'display': 'inline-block'}
                            ),
                        ]
                        ),
                    ]),
                ],
                    className='six columns'
                ),
                html.Div([
                    html.Div(id="number_of_cases"),
                    html.Span(
                            id="label_badge", 
                            className="badge badge-success badge-large",
                            style={'font-size': '15px'}
                            ),
                ],
                    className='three columns'

                )

            ], className="row"
        ),

        html.Div([

            html.Div([
                html.P(id='summary', style={'font-size': '20px'}),
                html.Div([
                        html.Div('This probability is a weighted average of the three probabilities of tears over each plane', style={'float': 'left', 'font-size': '20px'}),
                        html.Div('Slide over the slices of each MRI to inspect highlighted regions of tear as depicted by CAMs', style={'float': 'left', 'font-size': '20px'}),

                    ],
                    # style={'text-align': 'center'}
                    )
                    ],
                    className="twelve columns"),

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
                    className="four columns"
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
                    className="four columns"
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
                    className="four columns"
                ),

            ],
            className='row'
        )
    ], className='twelve columns')
)


# select label --- begin
@app.callback(
    dash.dependencies.Output('cases', 'options'),
    [
        dash.dependencies.Input('label_radioitems', 'value'),
        dash.dependencies.Input('pred_radioitems', 'value'),
    ]
)
def set_label(selected_label, selected_pred):
    if (selected_label == 'acl') and (selected_pred == 'acl'):
        filtered_cases = predictions[(predictions['labels'] == 1) & 
                                     (predictions['preds'] >= 0.5)].index.tolist()

    elif (selected_label == 'acl') and (selected_pred == 'normal'):
        filtered_cases = predictions[(predictions['labels'] == 1) & 
                                     (predictions['preds'] < 0.5)].index.tolist()

    elif (selected_label == 'normal') and (selected_pred == 'acl'):
        filtered_cases = predictions[(predictions['labels'] == 0) & 
                                     (predictions['preds'] >= 0.5)].index.tolist()

    elif (selected_label == 'normal') and (selected_pred == 'normal'):
        filtered_cases = predictions[(predictions['labels'] == 0) & 
                                     (predictions['preds'] < 0.5)].index.tolist()                                 

    filtered_cases = [c + 1130 for c in filtered_cases]
    options = [{'label': fc, 'value': fc} for fc in filtered_cases]
    return options
# select label --- end

# set badge label --- begin

@app.callback(
    dash.dependencies.Output('label_badge', 'children'),
    [
        dash.dependencies.Input('label_radioitems', 'value'),
        dash.dependencies.Input('pred_radioitems', 'value'),
    ]
)
def set_badge_label(selected_label, selected_pred):
    if (selected_label == 'acl') and (selected_pred == 'acl'):
        text = 'true positive case'

    elif (selected_label == 'acl') and (selected_pred == 'normal'):
        text = 'false negative case'

    elif (selected_label == 'normal') and (selected_pred == 'acl'):
        text = 'false positive case'

    elif (selected_label == 'normal') and (selected_pred == 'normal'):
        text = 'true negative case'

    return text

# set badge label --- end

# set badge color --- begin

@app.callback(
    dash.dependencies.Output('label_badge', 'className'),
    [
        dash.dependencies.Input('label_radioitems', 'value'),
        dash.dependencies.Input('pred_radioitems', 'value'),
    ]
)
def set_badge_color(selected_label, selected_pred):
    if (selected_label == 'acl') and (selected_pred == 'acl'):
        className = 'badge badge-success'

    elif (selected_label == 'acl') and (selected_pred == 'normal'):
        className = 'badge badge-error'

    elif (selected_label == 'normal') and (selected_pred == 'acl'):
        className = 'badge badge-error'

    elif (selected_label == 'normal') and (selected_pred == 'normal'):
        className = 'badge badge-success'

    return className

# set badge color --- end

# set a case value --- begin

@app.callback(
    dash.dependencies.Output('cases', 'value'),
    [
        dash.dependencies.Input('label_radioitems', 'value'),
        dash.dependencies.Input('pred_radioitems', 'value'),
    ]
)
def set_badge_color(selected_label, selected_pred):
    if (selected_label == 'acl') and (selected_pred == 'acl'):
        filtered_cases = predictions[(predictions['labels'] == 1) & 
                                     (predictions['preds'] >= 0.5)].index.tolist()

    elif (selected_label == 'acl') and (selected_pred == 'normal'):
        filtered_cases = predictions[(predictions['labels'] == 1) & 
                                     (predictions['preds'] < 0.5)].index.tolist()

    elif (selected_label == 'normal') and (selected_pred == 'acl'):
        filtered_cases = predictions[(predictions['labels'] == 0) & 
                                     (predictions['preds'] >= 0.5)].index.tolist()

    elif (selected_label == 'normal') and (selected_pred == 'normal'):
        filtered_cases = predictions[(predictions['labels'] == 0) & 
                                     (predictions['preds'] < 0.5)].index.tolist()                                 

    filtered_cases = [c + 1130 for c in filtered_cases]
    case_value = np.random.choice(filtered_cases)
    return case_value

# set a case value --- end


# set summary --- begin

@app.callback(
    dash.dependencies.Output('summary', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('label_radioitems', 'value'),
        dash.dependencies.Input('pred_radioitems', 'value')
    ]
)
def set_summary(selected_case, selected_label, selected_pred):
   
    proba = predictions['preds'].tolist()[int(selected_case) - 1130]
    proba = np.round(proba, 4)

    if (selected_label == 'acl') and (selected_pred == 'acl'):
        status = 'correctly'

    elif (selected_label == 'acl') and (selected_pred == 'normal'):
        status = 'incorrectly'

    elif (selected_label == 'normal') and (selected_pred == 'acl'):
        status = 'incorrectly'

    elif (selected_label == 'normal') and (selected_pred == 'normal'):
        status = 'correctly'

    if selected_pred == 'acl':
        summary = f'This patient, denoted by the MRI exam n°{selected_case}, is {status} diagnosed with an ACL tear with an ACL tear probability of {proba}'
    elif selected_pred == 'normal':
        summary = f'This patient, denoted by the MRI exam n°{selected_case}, is {status} diagnosed to be normal with an ACL tear probability of {proba}'
 

    return summary

# set summary --- end






# set number of cases --- begin

@app.callback(
    dash.dependencies.Output('number_of_cases', 'children'),
    [
        dash.dependencies.Input('label_radioitems', 'value'),
        dash.dependencies.Input('pred_radioitems', 'value')
    ]
)
def set_number_cases(selected_label, selected_pred):
    if (selected_label == 'acl') and (selected_pred == 'acl'):
        n = predictions[(predictions['labels'] == 1) & 
                                     (predictions['preds'] >= 0.5)].shape[0]

    elif (selected_label == 'acl') and (selected_pred == 'normal'):
        n = predictions[(predictions['labels'] == 1) & 
                                     (predictions['preds'] < 0.5)].shape[0]

    elif (selected_label == 'normal') and (selected_pred == 'acl'):
        n = predictions[(predictions['labels'] == 0) & 
                                     (predictions['preds'] >= 0.5)].shape[0]

    elif (selected_label == 'normal') and (selected_pred == 'normal'):
        n = predictions[(predictions['labels'] == 0) & 
                                     (predictions['preds'] < 0.5)].shape[0]                                 

    msg = f"{n} MRI exams"
    return msg

# set number of cases --- end


# update axial slider --- begin
@app.callback(
    dash.dependencies.Output('slider_axial', 'value'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_value_axial(selected_case):
    mri = np.load(f'../data/valid/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2


@app.callback(
    dash.dependencies.Output('slider_axial', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max_axial(selected_case):
    mri = np.load(f'../data/valid/axial/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider_axial', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks_axial(selected_case):
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
def set_slider_value_coronal(selected_case):
    mri = np.load(f'../data/valid/coronal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2


@app.callback(
    dash.dependencies.Output('slider_coronal', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max_coronal(selected_case):
    mri = np.load(f'../data/valid/coronal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider_coronal', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks_coronal(selected_case):
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
def set_slider_value_sagittal(selected_case):
    mri = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices // 2


@app.callback(
    dash.dependencies.Output('slider_sagittal', 'max'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_max_sagittal(selected_case):
    mri = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    number_slices = mri.shape[0]
    return number_slices - 1


@app.callback(
    dash.dependencies.Output('slider_sagittal', 'marks'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def set_slider_marks_sagittal(selected_case):
    mri = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    number_slices = mri.shape[0]
    marks = {str(i): '{}'.format(i) for i in range(number_slices)[::2]}
    return marks

# update sagittal slider --- end

# update slider --- END

# Axial
##########################################################################

# write number of slice axial - begin


@app.callback(
    dash.dependencies.Output('title_axial', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_axial', 'value')
    ]
)
def write_num_slice_axial(selected_case, selected_slice):
    case = np.load(f'../data/valid/axial/{selected_case}.npy')
    num_slices = case.shape[0]
    title = f'Visualization of slice n°{selected_slice}/{num_slices} and its corresponding CAM'
    return title
# write number of slice axial - end


# write score axial - begin
@app.callback(
    dash.dependencies.Output('score_axial', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def write_score_axial(selected_case):
    score = predictions.iloc[int(selected_case) - 1130]['axial']
    score = np.round(score, 4)
    msg = f"ACL tear proba on axial plane : {score}"
    return msg
# write score axial


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

# Coronal
##########################################################################

# write number of slice coronal - begin


@app.callback(
    dash.dependencies.Output('title_coronal', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_coronal', 'value')
    ]
)
def write_num_slice_coronal(selected_case, selected_slice):
    case = np.load(f'../data/valid/coronal/{selected_case}.npy')
    num_slices = case.shape[0]
    title = f'Visualization of slice n°{selected_slice}/{num_slices} and its corresponding CAM'
    return title
# write number of slice coronal - end

# write score coronal - begin


@app.callback(
    dash.dependencies.Output('score_coronal', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def write_score_coronal(selected_case):
    score = predictions.iloc[int(selected_case) - 1130]['coronal']
    score = np.round(score, 4)
    msg = f"ACL tear proba on coronal plane : {score}"
    return msg
# write score coronal


# update slice coronal --- begin
@app.callback(
    dash.dependencies.Output('slice_coronal', 'src'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_coronal', 'value'),

    ])
def update_slice_coronal(selected_case, selected_slice):
    s = np.load(f'../data/valid/coronal/{selected_case}.npy')[selected_slice]
    cv2.imwrite(f'./slice_coronal.png', s)
    encoded_image = base64.b64encode(open('./slice_coronal.png', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())
# update slice coronal --- end
# update cam coronal --- begin


@app.callback(
    dash.dependencies.Output('cam_coronal', 'src'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_coronal', 'value'),

    ])
def update_cam_coronal(selected_case, selected_slice):
    selected_case = int(selected_case) - 1130
    selected_case = '0' * (4 - len(str(selected_case))) + str(selected_case)
    src = f'./CAMS/coronal/{selected_case}/cams/{selected_slice}.png'
    encoded_image = base64.b64encode(open(src, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())

# update slice coronal --- end

# Sagittal
##########################################################################

# write number of slice sagittal - begin


@app.callback(
    dash.dependencies.Output('title_sagittal', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_sagittal', 'value')
    ]
)
def write_num_slice_sagittal(selected_case, selected_slice):
    case = np.load(f'../data/valid/sagittal/{selected_case}.npy')
    num_slices = case.shape[0]
    title = f'Visualization of slice n°{selected_slice}/{num_slices} and its corresponding CAM'
    return title
# write number of slice sagittal - end

# write score sagittal - begin


@app.callback(
    dash.dependencies.Output('score_sagittal', 'children'),
    [
        dash.dependencies.Input('cases', 'value'),
    ]
)
def write_score_sagittal(selected_case):
    score = predictions.iloc[int(selected_case) - 1130]['sagittal']
    score = np.round(score, 4)
    msg = f"ACL tear proba on sagittal plane : {score}"
    return msg
# write score sagittal


# update slice sagittal --- begin
@app.callback(
    dash.dependencies.Output('slice_sagittal', 'src'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_sagittal', 'value'),

    ])
def update_slice_sagittal(selected_case, selected_slice):
    s = np.load(f'../data/valid/sagittal/{selected_case}.npy')[selected_slice]
    cv2.imwrite(f'./slice_sagittal.png', s)
    encoded_image = base64.b64encode(open('./slice_sagittal.png', 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())
# update slice saigttal --- end
# update cam sagittal --- begin


@app.callback(
    dash.dependencies.Output('cam_sagittal', 'src'),
    [
        dash.dependencies.Input('cases', 'value'),
        dash.dependencies.Input('slider_sagittal', 'value'),

    ])
def update_cam_sagittal(selected_case, selected_slice):
    selected_case = int(selected_case) - 1130
    selected_case = '0' * (4 - len(str(selected_case))) + str(selected_case)
    src = f'./CAMS/sagittal/{selected_case}/cams/{selected_slice}.png'
    encoded_image = base64.b64encode(open(src, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())
# update slice coronal --- end


if __name__ == '__main__':
    app.run_server(debug=True)
