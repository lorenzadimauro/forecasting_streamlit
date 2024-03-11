import streamlit as st
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import plotly.express as px

from collections import UserDict
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

class TimeSeriesTensor(UserDict):
    """A dictionary of tensors for input into the RNN model.

    Use this class to:
      1. Shift the values of the time series to create a Pandas dataframe containing all the data
         for a single training example
      2. Discard any samples with missing values
      3. Transform this Pandas dataframe into a numpy array of shape
         (samples, time steps, features) for input into Keras

    The class takes the following parameters:
       - **dataset**: original time series
       - **target** name of the target column
       - **H**: the forecast horizon
       - **tensor_structures**: a dictionary discribing the tensor structure of the form
             { 'tensor_name' : (range(max_backward_shift, max_forward_shift), [feature, feature, ...] ) }
             if features are non-sequential and should not be shifted, use the form
             { 'tensor_name' : (None, [feature, feature, ...])}
       - **freq**: time series frequency (default 'H' - hourly)
       - **drop_incomplete**: (Boolean) whether to drop incomplete samples (default True)
    """

    def __init__(self, dataset, target, H, tensor_structure, freq='H', drop_incomplete=True):
        self.dataset = dataset
        self.target = target
        self.tensor_structure = tensor_structure
        self.tensor_names = list(tensor_structure.keys())

        self.dataframe = self._shift_data(H, freq, drop_incomplete)
        self.data = self._df2tensors(self.dataframe)

    def _shift_data(self, H, freq, drop_incomplete):

        # Use the tensor_structures definitions to shift the features in the original dataset.
        # The result is a Pandas dataframe with multi-index columns in the hierarchy
        #     tensor - the name of the input tensor
        #     feature - the input feature to be shifted
        #     time step - the time step for the RNN in which the data is input. These labels
        #         are centred on time t. the forecast creation time
        df = self.dataset.copy()

        idx_tuples = []

        # Shift the target column by H periods
        df['t+' + str(H)] = df[self.target].shift(-H, freq=freq)
        idx_tuples.append(('target', 'y', 't+' + str(H)))

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            dataset_cols = structure[1]

            for col in dataset_cols:

            # do not shift non-sequential 'static' features
                if rng is None:
                    df['context_'+col] = df[col]
                    idx_tuples.append((name, col, 'static'))

                else:
                    for t in rng:
                        sign = '+' if t > 0 else ''
                        shift = str(t) if t != 0 else ''
                        period = 't'+sign+shift
                        shifted_col = name+'_'+col+'_'+period
                        df[shifted_col] = df[col].shift(t*-1, freq=freq)
                        idx_tuples.append((name, col, period))

        df = df.drop(self.dataset.columns, axis=1)
        idx = pd.MultiIndex.from_tuples(idx_tuples, names=['tensor', 'feature', 'time step'])
        df.columns = idx

        if drop_incomplete:
            df = df.dropna(how='any')

        return df

    def _df2tensors(self, dataframe):

        # Transform the shifted Pandas dataframe into the multidimensional numpy arrays. These
        # arrays can be used to input into the keras model and can be accessed by tensor name.
        # For example, for a TimeSeriesTensor object named "model_inputs" and a tensor named
        # "target", the input tensor can be acccessed with model_inputs['target']

        inputs = {}
        y = dataframe['target']
        y = y.to_numpy()

        inputs['target'] = y

        for name, structure in self.tensor_structure.items():
            rng = structure[0]
            cols = structure[1]
            tensor = dataframe[name][cols].to_numpy()
            if rng is None:
                tensor = tensor.reshape(tensor.shape[0], len(cols))
            else:
                tensor = tensor.reshape(tensor.shape[0], len(cols), len(rng))
                tensor = np.transpose(tensor, axes=[0, 2, 1])
            inputs[name] = tensor

        return inputs

    def subset_data(self, new_dataframe):

        # Use this function to recreate the input tensors if the shifted dataframe
        # has been filtered.

        self.dataframe = new_dataframe
        self.data = self._df2tensors(self.dataframe)

all_features = ['pressure','three_hour_pressure_change', 'wind_direction', 
                'wind_speed', 'cloud_cover', 'FOG', 'temperature_diff',
                'present_weather_no precipitation at the station', 'present_weather_precipitation at the station',
                'cloud_type_cumulunimbus calvus', 'cloud_type_cumulus and stratocumulus not cumulogenitus',
                'cloud_type_cumulus humilis', 'cloud_type_cumulus mediocris', 
                'cloud_type_low clouds invisible due to darkness fog or sand', 'cloud_type_no low clouds', 
                'cloud_type_stratocumulus cumulogenitus', 'cloud_type_stratocumulus not cumulogenitus',
                'cloud_type_stratus fractus', 'cloud_type_stratus nebulosus']

T = 6
tensor_structure = {'X':(range(-T+1, 1), all_features)}

def convert_order_date(df):
  df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
  df = df.sort_values(by='date')
  df.set_index('date', inplace=True)
  return df

def import_dataset(df_path):
  df = pd.read_csv(df_path, sep=",")
  df = convert_order_date(df)
  if df_path == "datasets/grazzanise_gru.csv":
    df.columns.name = 'grazzanise'
  elif df_path == "datasets/trapani_gru.csv":
    df.columns.name = 'trapani'
  else:
    df.columns.name = 'treviso'
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

def dataset_scaling(df):
  train_start_dt = df.index.min()
  valid_start_dt = train_start_dt + pd.DateOffset(years=7)
  test_start_dt = valid_start_dt + pd.DateOffset(years=2)

  train = df.copy()[df.index < valid_start_dt][all_features]
  X_scaler = MinMaxScaler()
  num_columns = ['pressure', 'three_hour_pressure_change', 'wind_direction',
               'wind_speed', 'cloud_cover', 'temperature_diff']
  train[num_columns] = X_scaler.fit_transform(train[num_columns])

  look_back_dt = test_start_dt - dt.timedelta(hours=T-1)
  test = df.copy()[look_back_dt:][all_features]
  test[num_columns] = X_scaler.transform(test[num_columns])
  return test

def dataset_scaling_grazzanise(df):
  test_start_dt = dt.datetime.strptime("2021-11-30 07:00:00", "%Y-%m-%d %H:%M:%S")
  look_back_dt = test_start_dt - dt.timedelta(hours=T-1)
  test = df.copy()[test_start_dt:][all_features]
  num_columns = ['pressure', 'three_hour_pressure_change', 'wind_direction',
               'wind_speed', 'cloud_cover', 'temperature_diff']

  X_scaler = MinMaxScaler()
  test[num_columns] = X_scaler.fit_transform(test[num_columns])
  return test

def dataset_preparation(df):
  df = df.drop(columns= ['idstazione', 'visibility'])
  # codifico fog
  df['FOG'] = df['FOG'].map({'NO': 0, 'YES': 1})
  df_copy = df.copy()
  cat_columns = ['present_weather', 'cloud_type']
  df = pd.get_dummies(df, columns=cat_columns, drop_first=True)
  
  if df.columns.name == 'grazzanise':
    test = dataset_scaling_grazzanise(df)   
  
  else:
    test = dataset_scaling(df)
  
  return test, df_copy

def process_case(df, case_number, model_path):
    # Creare il TimeSeriesTensor per il caso corrente
    test_inputs = TimeSeriesTensor(df, 'FOG', case_number, tensor_structure)
    X_test = test_inputs['X']
    y_test = test_inputs['target']

    # Caricare il modello specifico per il caso corrente
    model = load_model(model_path.format(case_number))

    # Codice di adattamento per variabile binaria
    predictions = model.predict(X_test)

    # Creare un DataFrame con gli indici da test_inputs.dataframe
    result_df = pd.DataFrame(index=test_inputs.dataframe.index)

    result_df[f'fog prob t+{case_number}'] = np.round(predictions * 100, 2)

    return result_df

def dataset_evaluation(test, df_copy):
  
  # Creare una lista che contiene copy + i risultati dei sei casi
  result_dfs = [df_copy]
  # Utilizzare la funzione per ciascun caso
  for case_number in range(1, 7):
    model_path = "/content/model_{}.h5"
    result_df = process_case(test, case_number, model_path)
    result_dfs.append(result_df)

  return result_dfs

def create_eval_df(results, range_size):
    if range_size > len(results) or range_size < 1:
        raise ValueError("Il range specificato è fuori dal limite dei risultati.")

    eval_df = results[0]  # Inizia con il primo DataFrame

    for i in range(1, range_size+1):
        # Effettua l'inner join con il DataFrame successivo
        eval_df = pd.merge(eval_df, results[i], left_index=True, right_index=True, how='inner')

    return eval_df

def main():
    st.title("FOG NOWCASTING APP")

    # Datasets are imported
    grazzanise = import_dataset("datasets/grazzanise_gru.csv")
    trapani = import_dataset("datasets/trapani_gru.csv")
    treviso = import_dataset("datasets/treviso_gru.csv")
    
    datasets = [grazzanise, trapani, treviso]
    selectable_elements = [df.columns.name for df in datasets]
    selected_df_name = st.selectbox("Select Dataset", selectable_elements)
    selected_df = datasets[selectable_elements.index(selected_df_name)]

    # Function for plotting graphs is called
    plot_graphs(selected_df)

    # Button to prepare dataset
    if st.button("Prepare Dataset"):
      # This function is called
      test, df_copy = dataset_preparation(selected_df)

      # This function is called that returns a list of dfs
      results = dataset_evaluation(test, df_copy)

      # User selects horizon in a range from 1 to 6
      horizon = st.slider("Select Horizon", 1, 6, 1)
        
      # This function is called
      eval_df = create_eval_df(results, horizon)
        
      # Display eval_df
      st.dataframe(eval_df)

# Run the Streamlit app
if __name__ == "__main__":
    main()

