import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from io import StringIO
import base64

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# file uploader
app.layout = html.Div([
    html.H1("Used Car Price Predictor", style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(html.A('Upload File')),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'textAlign': 'center',
            'margin': '10px auto',
            'backgroundColor': '#D3D3D3',
            'color': 'black'
        },
        multiple=False
    ),
    html.Div(id='file-upload-feedback', style={'margin-bottom': '20px', 'text-align': 'center'}),
    html.Div(id='dynamic-content', style={'margin': '20px'}),
    dcc.Store(id='stored-data')
])

def parse_file_input(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(StringIO(decoded.decode('utf-8')))

@app.callback(
    [Output('file-upload-feedback', 'children'),
     Output('dynamic-content', 'children'),
     Output('stored-data', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_based_on_file(file_contents, file_name):
    if file_contents is not None:
        try:
            df = parse_file_input(file_contents)

            # column type preprocessing
            if 'Mileage' in df.columns:
                df['Mileage'] = df['Mileage'].str.split().str[0].astype(float)
            if 'Engine' in df.columns:
                df['Engine'] = df['Engine'].str.split().str[0].astype(float)
            if 'Power' in df.columns:
                df['Power'] = df['Power'].str.split().str[0].astype(float)
            if 'New_Price' in df.columns:
                df['New_Price'] = df['New_Price'].str.split().str[0].astype(float)

            numerical_cols = df.select_dtypes(include=['number']).columns
            numerical_cols = [col for col in numerical_cols if not col.startswith("Unnamed")]
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns

            target_dropdown = html.Div([
                html.Label("Select Target:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='target-selector',
                    options=[{'label': col, 'value': col} for col in numerical_cols],
                    value=numerical_cols[0] if not pd.Index(numerical_cols).empty else None,
                    style={'width': '150px'}
                )
            ], style={
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'margin-bottom': '20px',
                'background-color': '#f5f5f5',
                'padding': '10px',
                'width': '100%',
                'margin-left': 'auto', 
                'margin-right': 'auto'
            })

            analysis_section = html.Div([
                html.Div(
                    dcc.RadioItems(
                        id='categorical-selector',
                        options=[{'label': col, 'value': col} for col in categorical_cols],
                        value=categorical_cols[0] if not pd.Index(categorical_cols).empty else None,
                        inline=True,
                        style={'margin-bottom': '10px'}
                    ),
                    style={
                        'display': 'flex',
                        'justify-content': 'center',
                        'margin-bottom': '20px'
                    }
                ),
                html.H4(id='graph-title', style={'text-align': 'center', 'margin-top': '10px'}),  # set dynamically via update_averages_title()
                dcc.Graph(id='bar-chart', style={'margin-top': '20px'})
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})

            correlation_section = html.Div([
                html.H4(id='correlation-title', style={'text-align': 'center', 'margin-top': '10px'}),  # set dynamically via update_correlations_title()
                dcc.Graph(id='correlation-chart', style={'margin-top': '20px'})
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})

            return (
                "",
                html.Div([target_dropdown, html.Div([analysis_section, correlation_section])]),
                df.to_dict('records')
            )
        except Exception as e:
            return f"Error processing file: {str(e)}", "", None
    return "", "", None

@app.callback(
    Output('graph-title', 'children'),
    [Input('categorical-selector', 'value'),
     Input('target-selector', 'value')]
)
def update_averages_title(selected_category, selected_target):
    if selected_category and selected_target:
        return f"Average {selected_target} by {selected_category}"
    elif selected_target:
        return f"Average {selected_target} by Category"
    return "Average by Category"

@app.callback(
    Output('correlation-title', 'children'),
    Input('target-selector', 'value')
)
def update_correlations_title(selected_target):
    if selected_target:
        return f"Correlation Strength of Numerical Variables with {selected_target}"
    return "Correlation Strength of Numerical Variables"

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('categorical-selector', 'value'),
     Input('target-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_averages_bar_chart(selected_categorical, selected_target, data):
    if data:
        df = pd.DataFrame(data)
        if selected_categorical and selected_target:
            avg_target = df.groupby(selected_categorical)[selected_target].mean().sort_values()
            fig = px.bar(
                avg_target,
                x=avg_target.index,
                y=avg_target.values,
                text=avg_target.values,
                labels={'x': selected_categorical, 'y': f'Average {selected_target}'}
            )
            fig.update_traces(
                textposition='inside', 
                texttemplate='%{text:.6f}' 
            )
            fig.update_layout(
                title=f"Average {selected_target} by {selected_categorical}",
                xaxis_title=selected_categorical,
                yaxis_title=f"Average {selected_target}",
                title_x=0.5, 
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )
            return fig
    return px.bar()

@app.callback(
    Output('correlation-chart', 'figure'),
    [Input('target-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_correlations_bar_chart(selected_target, data):
    if data:
        df = pd.DataFrame(data)
        numerical_cols = df.select_dtypes(include=['number']).columns
        numerical_cols = [col for col in numerical_cols if not col.startswith("Unnamed")]
        if selected_target and numerical_cols:
            correlations = df[numerical_cols].corr()[selected_target].abs().sort_values(ascending=False)
            fig = px.bar(
                correlations,
                x=correlations.index,
                y=correlations.values,
                text=correlations.values,
                labels={'x': 'Feature', 'y': 'Correlation Strength (absolute value)'}
            )
            fig.update_traces(
                textposition='inside',
                texttemplate='%{text:.2f}'
            )
            fig.update_layout(
                title=f"Correlation Strength of Numerical Variables with {selected_target}",
                xaxis_title='Feature',
                yaxis_title='Correlation Strength (absolute value)',
                title_x=0.5,
                uniformtext_minsize=10,
                uniformtext_mode='hide'
            )
            return fig
    return px.bar()

if __name__ == '__main__':
    app.run_server(debug=True)
