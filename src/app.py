import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pickle
from dash.dependencies import Input, Output, State

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash( __name__, external_stylesheets=external_stylesheets)
server = app.server

# df = px.data.iris()
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv')
scores = pd.read_csv('results/scores.csv').round(4)
scores.rename(columns={'Unnamed: 0': ''}, inplace=True)
feature_importances = pd.read_csv('results/feature_importances.csv')
confusion_matrix = pd.read_csv('results/confusion_matrix.csv', index_col=0)
index_vals = df['class'].astype('category').cat.codes
model = pickle.load(open('results/model.pkl','rb'))


feature_importances.columns=['Features','Importances']
fig_feature_imp = px.bar(feature_importances, x='Importances', y='Features', template='simple_white')
fig_feature_imp.update_layout(yaxis={'categoryorder':'total ascending'})

fig_confusion_matrix = px.imshow(
    confusion_matrix,
    labels=dict(x='Predicted Label', y='True Label'),
    color_continuous_scale='blues'
)

# fig_confusion_matrix

fig = go.Figure(data=go.Splom(
                dimensions=[dict(label='sepal length',
                                 values=df['sepal length']),
                            dict(label='sepal width',
                                 values=df['sepal width']),
                            dict(label='petal length',
                                 values=df['petal length']),
                            dict(label='petal width',
                                 values=df['petal width'])],
                diagonal_visible=False, # remove plots on diagonal
                text=df['class'],
                marker=dict(color=index_vals,
                            showscale=False, # colors encode categorical variables
                            line_color='white', line_width=0.5)
                ))

fig.update_layout(
    title='Iris Data set',
    width=600,
    height=600,
)
markdown_text = '''
# Iris Predicitve Model
Goal: To construct a template for the output for ML models via Plotly Dash.
'''

app.layout = html.Div(
    # style={'textAlign':'center'},
    children=[
        # Header and goals
        dcc.Markdown(
            children=markdown_text
        ),

        html.Div(
            children=[   # The Inputs for the using the predicitve model.
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    'Petal length (cm)',
                                    className='three columns'
                                ),
                                html.Div(
                                    dcc.Input(id='petal-length', value='', type='number'),
                                    className='four columns'
                                )
                            ],
                            className='row'
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    'Petal width (cm)',
                                    className='three columns'
                                ),
                                html.Div(
                                    dcc.Input(id='petal-width', value='', type='number'),
                                    className='four columns'
                                )
                            ],
                            className='row'
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    'Sepal length (cm)',
                                    className='three columns'
                                ),
                                html.Div(
                                    dcc.Input(id='sepal-length', value='', type='number'),
                                    className='four columns'
                                )
                            ],
                            className='row'
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    'Sepal width (cm)',
                                    className='three columns'
                                ),
                                html.Div(
                                    dcc.Input(id='sepal-width', value='', type='number'),
                                    className='four columns'
                                )
                            ],
                            className='row'
                        ),
                    ],
                    className='seven columns'
                ),
                # End of inputs for predicitve model.
                html.Div(
                    id='prediction',
                    className='five columns'
                )
            ],
            className='twelve columns'
        ),
        html.Br(),
        html.H4('5-Fold Cross Validated Performance Metrics'),
        html.Div(
            children=[
                generate_table(scores)
            ],
            style={'margin':'auto', 'width':'33%'}
        ),
        html.Div(
            [   # Model performance and feature importances.
                html.Div(
                    # style={'width': '49%', 'display': 'inline-block'},
                    children=[
                        # generate_table(confusion_matrix),
                        html.H4('Normalized Confusion Matrix'),
                        dcc.Graph(
                            figure=fig_confusion_matrix
                        )
                    ],
                    className='six columns'
                ),
                html.Div(
                    # style={'width': '49%', 'display': 'inline-block'},
                    children=[
                        html.H4('Feature Importances'),
                        dcc.Graph(
                            id='bar-chart-feature-imp',
                            figure=fig_feature_imp
                        )
                    ],
                    className='six columns'
                ),
            ],
            className='row'
        ),

        # html.Div(
        #     style={'width': '49%', 'display': 'inline-block'},
        #     children=[
        #         dcc.Graph(
        #             id='scatter-plot-matrix',
        #             figure=fig
        #             )
        #         ]
        # )
    ],
    className='grid',
    style={'textAlign':'center'}
)

@app.callback(
    Output('prediction','children'),
    [Input('petal-length', 'value'),
        Input('petal-width', 'value'),
        Input('sepal-length', 'value'),
        Input('sepal-width', 'value')]
)
def ph(pl, pw, sl, sw):

    try:
        prediction = model.predict(pd.DataFrame([pl, pw, sl, sw]).transpose())
        proba = model.predict_proba(pd.DataFrame([pl, pw, sl, sw]).transpose())
        output_text = u'''
            This flower is a {} with probability of {}
        '''.format(confusion_matrix.index.to_numpy().flatten()[prediction], proba.flatten()[prediction])
    except Exception as e:
        return u'''Fill in all the flower measurements.'''
    return output_text

if __name__ == '__main__':
    app.run_server(debug=True)
