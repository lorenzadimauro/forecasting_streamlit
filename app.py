import streamlit as st
import numpy as np
import datetime as dt
import plotly.express as px
import pandas as pd


def convert_order_date(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by='date')
    df.set_index('date', inplace=True)
    return df

def import_dataset(df_path):
    df = pd.read_csv(df_path, sep=",")
    df = convert_order_date(df)
    return df

def colum_timeseries(df, column):
    fig_title = df.columns.name + ' ' + column + ' time series'
    fig = px.line(df, x=df.index , y=column, title=fig_title, color_discrete_sequence= ["darkblue"], template='plotly_white')
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)
    return

def column_barchart(df, column):
    fig_title = df.columns.name + ' ' + column + ' barchart'
    fig = px.histogram(df, x=column, title=fig_title, color_discrete_sequence= ["navy"], template='plotly_white')
    st.plotly_chart(fig)
    return

def plot_graphs(df):
    num_columns = ['pressure', 'temperature_diff']
    cat_columns = ['cloud_cover', 'cloud_type']

    for num_col in num_columns:
        colum_timeseries(df, num_col)
    for cat_col in cat_columns:
        column_barchart(df, cat_col)
    return



def main():
    st.title("FOG NOWCASTING APP")

    # Datasets are imported
    grazzanise = import_dataset("datasets/grazzanise_gru.csv")
    trapani = import_dataset("datasets/trapani_gru.csv")
    treviso = import_dataset("datasets/treviso_gru.csv")

    grazzanise.columns.name = 'grazzanise'
    trapani.columns.name = 'trapani'
    treviso.columns.name = 'treviso'
    
    datasets = [grazzanise, trapani, treviso]
    selectable_elements = [df.columns.name for df in datasets]
    selected_df_name = st.selectbox("Select Dataset", selectable_elements)
    selected_df = datasets[selectable_elements.index(selected_df_name)]

    # Function for plotting graphs is called
    plot_graphs(selected_df)

    # Button to prepare dataset
    if st.button("Test on selected dataset"):

      eval_path_template = "evaluation/eval_df_{}.csv"
      eval_path = eval_path_template.format(selected_df.columns.name)
        
      # This function is called
      eval_df = import_dataset(eval_path)
      # Apply custom styling to highlight values > 70 in red for specific columns
      def style_specific_columns(col):
          if 'fog prob t+' in col:
              return ['color: red' if v > 70 else '' for v in col]
          return ['' for _ in col]
        
      eval_df_styled = eval_df.style.apply(style_specific_columns)
        
      # Display eval_df with custom styling
      st.write(eval_df_styled)
        
      # Display eval_df
      #st.dataframe(eval_df)

# Run the Streamlit app
if __name__ == "__main__":
    main()
