import streamlit as st
import os, io, webbrowser, base64, time, warnings, logging, random
import requests
from io import BytesIO
from PIL import Image

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU, Conv1D, Flatten, Bidirectional, LeakyReLU, ELU, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import chardet
import holidays
import torch
import tensorflow as tf
import lightgbm as lgb
import catboost as cb
import optuna
from darts.models import NHiTSModel
from darts import TimeSeries
import base64
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

seed = int(time.time())
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
print(f"Utilisation du seed: {seed}")

# =======================
# FONCTIONS UTILITAIRES
# =======================

def load_image_from_url(url):
    """
    Charge une image depuis une URL et la retourne sous forme d'objet PIL.Image.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image depuis l'URL : {url} ({e})")
        return None

def get_base64_image_from_url(url):
    """
    Télécharge une image depuis une URL et la convertit en base64 (string).
    Utile pour l'intégrer dans du HTML (balises <img>, etc.).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image depuis l'URL : {url} ({e})")
        return None

# =======================
# CLASSES & CALLBACKS
# =======================

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        loss = logs.get('loss')
        print(f"Epoch {epoch + 1}: Loss = {loss}, Learning rate = {lr}")

class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        n = inputs.shape[1]
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            A[i, i] = 1.0
            if i > 0:
                A[i, i-1] = 1.0
            if i < n-1:
                A[i, i+1] = 1.0
        A = tf.constant(A, dtype=tf.float32)
        batch_size = tf.shape(inputs)[0]
        A_batch = tf.tile(tf.expand_dims(A, 0), [batch_size, 1, 1])
        support = tf.matmul(inputs, self.kernel)
        output = tf.matmul(A_batch, support)
        return output

# =======================
# FONCTIONS HIMA
# =======================

def get_season(month):
    if month in [12, 1, 2]:
        return "hiver"
    elif month in [3, 4, 5]:
        return "printemps"
    elif month in [6, 7, 8]:
        return "été"
    else:
        return "automne"

class FTTHPredictor:
    def __init__(self, seq_length, prediction_type):
        self.seq_length = seq_length
        self.prediction_type = prediction_type
        self.encoder = OneHotEncoder(sparse_output=False)

    def detect_separator_and_encoding(self, filepath):
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read(10000))
        encoding = result['encoding']
        with open(filepath, 'r', encoding=encoding) as file:
            line = file.readline()
            separator = ',' if ',' in line else ';' if ';' in line else '\t'
        return separator, encoding

    def is_15_minute_data(self, data):
        time_deltas = data['date'].diff().dropna()
        return (time_deltas == timedelta(minutes=15)).all()

    def replace_outliers_with_median(self, data, outlier_dates):
        if 'volume' in data.columns:
            column_name = 'volume'
        elif 'call volumes' in data.columns:
            column_name = 'call volumes'
        else:
            return data
        median_value = data[column_name].median()
        data.loc[data['date'].isin(outlier_dates), column_name] = median_value
        return data

    def detect_recurring_holidays(self, data, threshold=0.2):
        data['month_day'] = data['date'].dt.strftime('%m-%d')
        from collections import defaultdict
        holiday_candidates = defaultdict(list)
        for date_, volume in data.groupby('month_day')['volume'].mean().items():
            if volume < data['volume'].mean() * threshold:
                holiday_candidates[date_].append(volume)
        recurring_holidays = [md for md, vols in holiday_candidates.items() if len(vols) > 1]
        return recurring_holidays

    def calculate_ema(self, data, span=7):
        data['EMA_volume'] = data['volume'].ewm(span=span, adjust=False).mean()
        return data

    def calculate_weekly_weights(self, data):
        data = self.calculate_ema(data, span=7)
        data['daily_weight'] = data['volume'] / data['EMA_volume']
        data['weekday'] = data['date'].dt.weekday
        weekday_weights = data.groupby('weekday')['daily_weight'].mean().reindex(range(7), fill_value=1)
        return data, weekday_weights

    def add_time_features(self, data):
        from scipy.signal import savgol_filter
        from statsmodels.tsa.seasonal import seasonal_decompose

        data['month'] = data['date'].dt.month
        data['day_of_month'] = data['date'].dt.day
        data['week_of_year'] = data['date'].dt.isocalendar().week
        data['day_of_week'] = data['date'].dt.weekday
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

        # Transformations cycliques
        data['sin_day'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['cos_day'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
        data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)

        # Encodage "saison"
        data['season'] = data['month'].apply(get_season)
        season_dummies = pd.get_dummies(data['season'], prefix='season')
        data = pd.concat([data, season_dummies], axis=1)
        data.drop('season', axis=1, inplace=True)

        data['smoothed_volume'] = data['volume'].rolling(window=5, min_periods=1).mean()
        data['ema_volume'] = data['volume'].ewm(span=5, adjust=False).mean()

        try:
            data['savgol_volume'] = savgol_filter(data['volume'], window_length=7, polyorder=2)
        except Exception as e:
            st.write("Savitzky-Golay a échoué :", e)
            data['savgol_volume'] = data['volume']

        data['rolling_mean_7'] = data['volume'].rolling(window=7, min_periods=1).mean()
        data['rolling_std_7'] = data['volume'].rolling(window=7, min_periods=1).std().fillna(0)
        data['adaptive_rolling_mean'] = self.adaptive_rolling_mean(data, base_window=7)
        data['diff'] = data['volume'].diff().fillna(0)

        try:
            decomposition = seasonal_decompose(data['volume'], model='additive', period=7, extrapolate_trend='freq')
            data['trend'] = decomposition.trend.fillna(method='bfill').fillna(method='ffill')
            data['seasonal_component'] = decomposition.seasonal.fillna(method='bfill').fillna(method='ffill')
        except Exception as e:
            st.write("La décomposition saisonnière a échoué :", e)
            data['trend'] = data['volume']
            data['seasonal_component'] = 0

        return data

    def adaptive_rolling_mean(self, data, base_window=7):
        volatility = data['volume'].rolling(window=base_window, min_periods=1).std().fillna(0)
        normalized_volatility = volatility / (volatility.max() if volatility.max() != 0 else 1)
        adaptive_window = (base_window * (1 - normalized_volatility)).round().astype(int)
        adaptive_window = adaptive_window.replace(0, 1)
        smoothed = []
        for idx, win in enumerate(adaptive_window):
            if idx < win:
                smoothed.append(data['volume'].iloc[:idx+1].mean())
            else:
                smoothed.append(data['volume'].iloc[idx-win+1:idx+1].mean())
        return smoothed

    def load_and_preprocess_data(self, filepath, holidays_list, holiday_weights, outlier_dates,
                                 predict_holidays_as_zero, predict_weekends_as_zero,
                                 predict_saturday_as_zero, predict_sunday_as_zero, working_hours):
        separator, encoding = self.detect_separator_and_encoding(filepath)
        try:
            data = pd.read_csv(filepath, sep=separator, encoding=encoding,
                               parse_dates=['date'], dayfirst=True, infer_datetime_format=True)
        except Exception as e:
            logging.error(f"Error loading CSV: {e}")
            st.error(f"Failed to load CSV: {e}")
            return None, None, None, None

        try:
            if 'tranches' in data.columns:
                data, data_scaled, scaler = self.handle_tranches_data(
                    data, holidays_list, holiday_weights, outlier_dates,
                    predict_holidays_as_zero, predict_weekends_as_zero,
                    predict_saturday_as_zero, predict_sunday_as_zero, working_hours
                )
            elif 'date' in data.columns and 'volume' in data.columns:
                data, data_scaled, scaler = self.handle_date_volume_data(
                    data, holidays_list, holiday_weights, outlier_dates,
                    predict_holidays_as_zero, predict_weekends_as_zero,
                    predict_saturday_as_zero, predict_sunday_as_zero, working_hours
                )
            else:
                raise ValueError("Le fichier CSV doit contenir les colonnes 'date' et 'volume', ou 'date', 'tranches' et 'call volumes'.")

            recurring_holidays = self.detect_recurring_holidays(data)
            recurring_holidays_dates = [pd.Timestamp(f"{year}-{md}")
                                        for md in recurring_holidays
                                        for year in range(data['date'].dt.year.min(), data['date'].dt.year.max() + 1)]
            for recurring_holiday in recurring_holidays_dates:
                if recurring_holiday not in holidays_list:
                    holidays_list.append(recurring_holiday)
                    holiday_weights.append(100)

            data, weekday_weights = self.calculate_weekly_weights(data)
            lag_limit = self.calculate_optimal_lags(data)

            for i in range(1, lag_limit + 1):
                data[f'volume_lag{i}'] = data['volume'].shift(i)
            data.dropna(inplace=True)

            st.subheader("📋 Aperçu des Données")
            data_preview = data[['date', 'volume', 'month', 'day_of_month']].head(10)
            data_preview.columns = ['Date', 'Volume', 'Mois', 'Jour du mois']
            st.write(data_preview)

            fig_time_series = go.Figure()
            fig_time_series.add_trace(go.Scatter(
                x=data["date"], y=data["volume"],
                mode="lines+markers",
                line=dict(color="lightgreen", width=2),
                marker=dict(size=5, color="aqua"),
                name="Volume"
            ))
            fig_time_series.update_layout(
                width=1200,
                height=500,
                plot_bgcolor="rgb(10,40,50)",
                paper_bgcolor="rgb(10,40,50)",
                font=dict(color="white"),
                xaxis=dict(showgrid=False, color="white"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)", color="white"),
                title="📉 Tendance des Volumes dans le Temps",
                title_font=dict(size=18, color="white")
            )
            st.plotly_chart(fig_time_series, width=600)
            st.subheader("Graphe Dynamique")
            plt.style.use('seaborn-v0_8-bright')
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            plot_acf(data['volume'], ax=ax[0], lags=50)

        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            st.error(f"Failed to preprocess data: {e}")
            return None, None, None, None

        return data, data_scaled, scaler, weekday_weights

    def handle_tranches_data(self, data, holidays_list, holiday_weights, outlier_dates,
                              predict_holidays_as_zero, predict_weekends_as_zero,
                              predict_saturday_as_zero, predict_sunday_as_zero, working_hours):
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])
        for holiday, weight in zip(holidays_list, holiday_weights):
            if predict_holidays_as_zero:
                data.loc[data['date'] == holiday, 'call volumes'] = 0
            else:
                data.loc[data['date'] == holiday, 'call volumes'] *= weight / 100
        for outlier_date in outlier_dates:
            data = self.replace_outliers_with_median(data, outlier_dates)
        data['call volumes'] = data['call volumes'].astype(float).ffill()
        data = self.add_time_features(data)
        volume_data = data['call volumes'].values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(volume_data)
        return data, data_scaled, scaler

    def handle_date_volume_data(self, data, holidays_list, holiday_weights, outlier_dates,
                                 predict_holidays_as_zero, predict_weekends_as_zero,
                                 predict_saturday_as_zero, predict_sunday_as_zero, working_hours):
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])

        for holiday, weight in zip(holidays_list, holiday_weights):
            if predict_holidays_as_zero:
                data.loc[data['date'] == holiday, 'volume'] = 0
            else:
                data.loc[data['date'] == holiday, 'volume'] *= weight / 100

        for outlier_date in outlier_dates:
            data = self.replace_outliers_with_median(data, outlier_dates)
        data['volume'] = data['volume'].astype(float).ffill()

        if not self.is_15_minute_data(data):
            if self.prediction_type == 'tranche':
                data = self.distribute_daily_volume_to_intervals(data, working_hours)

        data = self.add_time_features(data)
        volume_data = data['volume'].values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(volume_data)
        return data, data_scaled, scaler

    def distribute_daily_volume_to_intervals(self, data, working_hours):
        start_hour, end_hour = working_hours
        if end_hour <= start_hour:
            raise ValueError("L'heure de fin doit être supérieure à l'heure de début.")
        intervals = int((end_hour - start_hour) * 4)
        data['volume_per_interval'] = data['volume'] / intervals
        distributed_data = pd.DataFrame()
        for _, row in data.iterrows():
            base_time = datetime.combine(row['date'].date(), datetime.min.time()) + timedelta(hours=start_hour)
            for i in range(intervals):
                distributed_data = pd.concat([
                    distributed_data,
                    pd.DataFrame({
                        'date': [base_time + timedelta(minutes=15 * i)],
                        'volume': [row['volume_per_interval']]
                    })
                ], ignore_index=True)
        return distributed_data

    def calculate_optimal_lags(self, data):
        if len(data) < 2:
            return 1
        nlags = min(100, len(data) - 1)
        acf_values = sm.tsa.acf(data['volume'], nlags=nlags)
        try:
            optimal_lag = next((i for i, val in enumerate(acf_values) if abs(val) < 0.2),
                               min(30, len(data)-1))
        except StopIteration:
            optimal_lag = min(30, len(data)-1)
        return optimal_lag

    def plot_outlier_graph(self, data):
        fig = go.Figure()
        frames = [
            go.Frame(
                data=[
                    go.Scatter(
                        x=data["date"][:k],
                        y=data["volume"][:k],
                        mode="lines",
                        line=dict(color="lightgreen", width=2),
                        name="Volume"
                    )
                ],
                name=str(k)
            ) for k in range(1, len(data))
        ]
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["volume"],
            mode="lines",
            line=dict(color="rgba(100, 255, 100, 0.2)", width=6),
            name="Glow Effect",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=data["date"], y=data["volume"],
            mode="lines",
            line=dict(color="lightgreen", width=2),
            name="Volume"
        ))
        fig.update_layout(
            plot_bgcolor="rgb(10,40,50)",
            paper_bgcolor="rgb(10,40,50)",
            font=dict(color="white", family="Poppins, sans-serif"),
            xaxis=dict(showgrid=False, zeroline=False, color="white"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)", zeroline=False, color="white"),
            title=dict(text="📈 Évolution des Données dans le Temps",
                       font=dict(size=22, color="white", family="Arial Black"),
                       x=0.5, y=0.95, xanchor="center", yanchor="top")
        )
        fig.update_traces(line=dict(dash="dot"), selector=dict(name="Volume"))
        fig.update_layout(updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play", method="animate", args=[None])]
        )])
        st.plotly_chart(fig, width=600)

    def split_train_test(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        if train_size == 0:
            train_size = len(data) - 1
        return data[:train_size], data[train_size:]

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)

    def save_results(self, future_dates, future_predictions, original_data):
        future_predictions = [max(0, pred) for pred in future_predictions]
        results_df = pd.DataFrame({
            'date': future_dates,
            'predicted_volume': future_predictions
        })
        original_data_filtered = original_data.drop(
            columns=[col for col in original_data.columns if col.startswith("volume_lag")],
            errors='ignore'
        )
        results_df = pd.concat([results_df, original_data.reset_index(drop=True)], axis=1)
        timestamp = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
        filename = r"C:\Users\dell\Downloads\HIMA_prediction_" + timestamp + ".csv"
        results_df.to_csv(filename, index=False, sep=',', encoding='utf-8')
        st.success(f"Results saved successfully as {filename}.")
        return results_df.to_csv(index=False, sep=',', decimal=",", encoding='utf-8')

    def train_model(self, X_train, y_train, model_type, trial=None):
        model_type_lower = model_type.lower()
        ML_MODELS = ['adaboost', 'catboost', 'huber', 'svr', 'ensemble', 'elasticnet', 'lightgbm']

        if model_type_lower in ML_MODELS:
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

            def objective_optuna(trial):
                if model_type_lower == 'adaboost':
                    from sklearn.ensemble import AdaBoostRegressor
                    n_estimators = trial.suggest_categorical('n_estimators', [50, 200])
                    learning_rate = trial.suggest_categorical('learning_rate', [0.005, 0.001])
                    loss = trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
                    base_model = AdaBoostRegressor(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        loss=loss,
                        random_state=42
                    )

                elif model_type_lower == 'catboost':
                    iterations = trial.suggest_categorical('iterations', [100, 200])
                    depth = trial.suggest_categorical('depth', [4, 6, 8])
                    learning_rate = trial.suggest_categorical('learning_rate', [0.05, 0.1])
                    l2_leaf_reg = trial.suggest_categorical('l2_leaf_reg', [1, 3, 5, 7])
                    subsample = trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0])
                    base_model = cb.CatBoostRegressor(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        l2_leaf_reg=l2_leaf_reg,
                        subsample=subsample,
                        silent=True,
                        random_state=42
                    )

                elif model_type_lower == 'huber':
                    epsilon = trial.suggest_categorical('epsilon', [1.35, 1.5, 1.75])
                    alpha = trial.suggest_categorical('alpha', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
                    base_model = HuberRegressor(
                        epsilon=epsilon,
                        alpha=alpha
                    )

                elif model_type_lower == 'svr':
                    C = trial.suggest_categorical('C', [0.5, 1.0, 2.0])
                    epsilon = trial.suggest_categorical('epsilon', [0.05, 0.1, 0.2])
                    gamma = trial.suggest_categorical('gamma', [0.1, 0.01, 0.001])
                    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                    base_model = SVR(
                        C=C,
                        epsilon=epsilon,
                        gamma=gamma,
                        kernel=kernel
                    )

                elif model_type_lower == 'elasticnet':
                    alpha = trial.suggest_categorical('alpha', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
                    l1_ratio = trial.suggest_categorical('l1_ratio', [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
                    base_model = ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        random_state=42
                    )

                elif model_type_lower == 'ensemble':
                    weights = trial.suggest_categorical('weights', [[1,1,1,1], [2,1,1,2], [1,2,2,1]])
                    base_model = VotingRegressor(
                        estimators=[
                            ('adaboost', AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),
                            ('catboost', cb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=10, random_state=12, verbose=0)),
                            ('huber', HuberRegressor(epsilon=1.35, alpha=0.0001)),
                            ('mlp', MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=500, activation='relu', early_stopping=True, solver='adam', random_state=42))
                        ],
                        weights=weights,
                        n_jobs=-1
                    )

                elif model_type_lower == 'lightgbm':
                    num_leaves = trial.suggest_categorical('num_leaves', [31, 50])
                    n_estimators = trial.suggest_categorical('n_estimators', [100, 240, 300])
                    learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.005, 0.01])
                    base_model = lgb.LGBMRegressor(
                        random_state=42,
                        num_leaves=num_leaves,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate
                    )

                cv = TimeSeriesSplit(n_splits=3)
                mae_list = []
                for train_idx, val_idx in cv.split(X_train_reshaped):
                    X_tr, X_val = X_train_reshaped[train_idx], X_train_reshaped[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    base_model.fit(X_tr, y_tr)
                    y_pred = base_model.predict(X_val)
                    mae_list.append(mean_absolute_error(y_val, y_pred))

                return np.mean(mae_list)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective_optuna, n_trials=20)

            best_params = study.best_params
            st.write(f"Best parameters for {model_type_lower} via Optuna: {best_params}")

            if model_type_lower == 'adaboost':
                final_model = AdaBoostRegressor(
                    n_estimators=best_params['n_estimators'],
                    learning_rate=best_params['learning_rate'],
                    loss=best_params['loss'],
                    random_state=42
                )
            elif model_type_lower == 'catboost':
                final_model = cb.CatBoostRegressor(
                    iterations=best_params['iterations'],
                    depth=best_params['depth'],
                    learning_rate=best_params['learning_rate'],
                    l2_leaf_reg=best_params['l2_leaf_reg'],
                    subsample=best_params['subsample'],
                    silent=True,
                    random_state=42
                )
            elif model_type_lower == 'huber':
                final_model = HuberRegressor(
                    epsilon=best_params['epsilon'],
                    alpha=best_params['alpha']
                )
            elif model_type_lower == 'svr':
                final_model = SVR(
                    C=best_params['C'],
                    epsilon=best_params['epsilon'],
                    gamma=best_params['gamma'],
                    kernel=best_params['kernel']
                )
            elif model_type_lower == 'elasticnet':
                final_model = ElasticNet(
                    alpha=best_params['alpha'],
                    l1_ratio=best_params['l1_ratio'],
                    random_state=42
                )
            elif model_type_lower == 'ensemble':
                final_model = VotingRegressor(
                    estimators=[
                        ('adaboost', AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)),
                        ('catboost', cb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=10, random_state=12, verbose=0)),
                        ('huber', HuberRegressor(epsilon=1.35, alpha=0.0001)),
                        ('mlp', MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=500, activation='relu', early_stopping=True, solver='adam', random_state=42))
                    ],
                    weights=best_params['weights'],
                    n_jobs=-1
                )
            elif model_type_lower == 'lightgbm':
                final_model = lgb.LGBMRegressor(
                    random_state=42,
                    num_leaves=best_params['num_leaves'],
                    n_estimators=best_params['n_estimators'],
                    learning_rate=best_params['learning_rate']
                )

            final_model.fit(X_train_reshaped, y_train)
            return final_model

        elif model_type_lower == 'ttm':
            model = Sequential()
            model.add(Dense(1024, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3))
            model.add(Dense(1000, kernel_regularizer=l2(0.001)))
            model.add(LeakyReLU(alpha=0.3))
            model.add(Dropout(0.2))
            model.add(Dense(512, kernel_regularizer=l2(0.001)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.2))
            model.add(Dense(328, kernel_regularizer=l2(0.001)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.2))
            model.add(Dense(328, kernel_regularizer=l2(0.001)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.2))
            model.add(Dense(400, kernel_regularizer=l2(0.0001)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3))
            model.add(Dense(350, kernel_regularizer=l2(0.0001)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3))
            model.add(Dense(1, activation='linear'))
            optimizer = Adam(learning_rate=0.0005, decay=1e-5)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            st.write("📊 **Résumé du modèle TTM :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6, verbose=2)
            lr_logger = LearningRateLogger()
            model.fit(X_train, y_train, epochs=200, batch_size=8, callbacks=[early_stop, reduce_lr, lr_logger], verbose=2)
            st.write("Modèle TTM entraîné.")
            return model

        elif model_type_lower == 'llama':
            model = Sequential()
            model.add(Dense(512, input_shape=(X_train.shape[1],), activation='swish', kernel_regularizer=l2(0.0005)))
            model.add(Dropout(0.3))
            model.add(Dense(400, activation='swish', kernel_regularizer=l2(0.0005)))
            model.add(Dropout(0.2))
            model.add(Dense(400, activation='swish', kernel_regularizer=l2(0.0005)))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            optimizer = AdamW(learning_rate=0.0003, weight_decay=1e-5)
            model.compile(optimizer=optimizer, loss='huber', metrics=['mse'])
            st.write("📊 **Résumé du modèle LLAMA (Optimisé) :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
            lr_logger = LearningRateLogger()
            model.fit(X_train, y_train, epochs=200, batch_size=8, callbacks=[early_stop, reduce_lr, lr_logger], verbose=2)
            return model

        elif model_type_lower == 'timexer':
            model = Sequential()
            model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(800, activation='relu', kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.3))
            model.add(Dense(500, activation='relu', kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.3))
            model.add(Dense(250, activation='relu', kernel_regularizer=l2(0.01)))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            optimizer = Adam(learning_rate=0.0005)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            st.write("📊 **Résumé du modèle Timexer :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
            model.fit(X_train, y_train, epochs=350, batch_size=8, validation_split=0.2, callbacks=[early_stop], verbose=2)
            st.write("Modèle Timexer entraîné.")
            return model

        elif model_type_lower == 'nhits':
            model = Sequential()
            model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='swish'))
            model.add(Dropout(0.3))
            model.add(Dense(512, activation='swish'))
            model.add(Dropout(0.3))
            model.add(Dense(256, activation='swish'))
            model.add(Dropout(0.3))
            model.add(Dense(128, activation='swish'))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='swish'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            optimizer = Adam(learning_rate=0.001, amsgrad=True)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            def scheduler(epoch, lr):
                return lr * 0.95 if epoch > 10 else lr
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=2)
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
            model.fit(X_train, y_train, epochs=400, batch_size=128, validation_split=0.1,
                      callbacks=[early_stop, lr_scheduler], verbose=2)
            st.write("✅ Modèle NHITS (Optimisé) entraîné.")
            return model

        elif model_type_lower == 'moirai':
            model = Sequential()
            model.add(Dense(300, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.005)))
            model.add(Dropout(0.25))
            model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.005)))
            model.add(Dropout(0.25))
            model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.005)))
            model.add(Dropout(0.25))
            model.add(Dense(820, activation='relu', kernel_regularizer=l2(0.005)))
            model.add(Dropout(0.25))
            model.add(Dense(200, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(304, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(600, activation='relu'))
            model.add(Dropout(0.15))
            model.add(Dense(1, activation='linear'))
            optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            st.write("📊 **Résumé du modèle Moirai (Optimisé) :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6, verbose=2)
            st.write(f"Training Optimized MOIRAI: Learning rate: {optimizer.learning_rate.numpy()}")
            model.fit(X_train, y_train, epochs=400, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=2)
            return model

        elif model_type_lower == 'softs':
            model = Sequential()
            model.add(Dense(600, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.3))
            model.add(Dense(700, kernel_regularizer=l2(0.01)))
            model.add(LeakyReLU(alpha=0.15))
            model.add(Dropout(0.1))
            model.add(Dense(500, kernel_regularizer=l2(0.01)))
            model.add(LeakyReLU(alpha=0.1))
            model.add(Dropout(0.1))
            model.add(Dense(600))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.1))
            model.add(Dense(400))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(300))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.1))
            model.add(Dense(1, activation='linear'))
            optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
            model.compile(optimizer=optimizer, loss='huber', metrics=['mse'])
            st.write("📊 **Résumé du modèle softs :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-6, verbose=2)
            lr_logger = LearningRateLogger()
            st.write(f"Training Optimized SOFTS: Learning rate: {optimizer.learning_rate.numpy()}")
            model.fit(X_train, y_train, epochs=200, batch_size=6, callbacks=[early_stop, reduce_lr, lr_logger], verbose=2)
            return model

        elif model_type_lower == 'lstm':
            model = Sequential()
            model.add(Bidirectional(LSTM(512, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]))))
            model.add(Dropout(0.4))

            model.add(LSTM(256, return_sequences=True, activation='relu'))
            model.add(Dropout(0.3))
            model.add(LSTM(100, return_sequences=True, activation='relu'))
            model.add(Dropout(0.1))

            model.add(LSTM(8, return_sequences=False, activation='relu'))
            model.add(Dense(1, activation='linear'))
            model.add(tf.keras.layers.BatchNormalization())
            optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-5)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            st.write("📊 **Résumé du modèle LSTM Amélioré :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
            model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=160, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=2)
            st.write("✅ Modèle LSTM Amélioré entraîné.")
            return model

        elif model_type_lower == 'mlp':
            model = Sequential()
            model.add(Dense(612, input_shape=(X_train.shape[1],)))
            model.add(Activation(tf.nn.gelu))
            model.add(Dropout(0.2))
            model.add(Dense(100))
            model.add(Activation(tf.nn.gelu))
            model.add(Dropout(0.2))
            model.add(Dense(380))
            model.add(Activation(tf.nn.gelu))
            model.add(Dropout(0.2))
            model.add(Dense(320))
            model.add(Activation(tf.nn.gelu))
            model.add(Dropout(0.2))
            model.add(Dense(120))
            model.add(Activation(tf.nn.gelu))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            optimizer = AdamW(learning_rate=0.0005, weight_decay=1e-5)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
            model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=[early_stop, reduce_lr], verbose=2)
            return model

        elif model_type_lower == 'gru':
            model = Sequential()
            model.add(GRU(512, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(GRU(128, activation='tanh', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(64, activation='tanh', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(32, activation='tanh', return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(1, activation='tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            model.compile(optimizer=AdamW(learning_rate=0.001), loss='mean_absolute_error', metrics=['mse'])
            st.write("📊 **Résumé du modèle GRU :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=2)
            model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=200, batch_size=8, validation_split=0.2, callbacks=[early_stop], verbose=2)
            st.write("Modèle GRU entraîné.")
            return model

        elif model_type_lower == 'tcn':
            model = Sequential()
            model.add(Conv1D(filters=512, kernel_size=3, padding='causal', activation='swish',
                             input_shape=(X_train.shape[1], 1)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Conv1D(filters=256, kernel_size=3, padding='causal', activation='swish'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Conv1D(filters=128, kernel_size=3, padding='causal', activation='swish'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Flatten())
            model.add(Dense(100, activation='swish'))
            model.add(Dropout(0.1))
            model.add(Dense(64, activation='swish'))
            model.add(Dropout(0.1))
            model.add(Dense(1, activation='linear'))
            optimizer = AdamW(learning_rate=0.0005, weight_decay=1e-5)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            st.write("📊 **Résumé du modèle TCN :**")
            st.text(model.summary())
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=2)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
            model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=500, batch_size=64, validation_split=0.1, callbacks=[early_stop, reduce_lr], verbose=2)
            st.write("✅ Modèle TCN entraîné.")
            return model

        elif model_type_lower == 'nbeats':
            input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
            x = Dense(512, activation='relu')(input_layer)
            x = Dense(256, activation='relu')(x)
            block_output = Dense(256, activation='relu')(x)
            x = Dense(128, activation='relu')(block_output)
            output = Dense(1, activation='linear')(x)
            model = Model(inputs=input_layer, outputs=output)
            optimizer = AdamW(learning_rate=0.0001)
            model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse'])
            st.write("📊 **Résumé du modèle nbeats :**")
            early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=2)
            model.fit(X_train, y_train, epochs=200, batch_size=8, callbacks=[early_stop], verbose=2)
            st.write("Modèle N-BEATS entraîné.")
            return model

    def predict_future(self, model, data_scaled, scaler, start_date, end_date, holidays_list,
                       holiday_weights, outlier_dates, predict_holidays_as_zero,
                       predict_weekends_as_zero, predict_saturday_as_zero, predict_sunday_as_zero,
                       working_hours, model_type, weekday_weights, percentage_reduction=None):
        if self.prediction_type == 'tranche':
            future_dates = []
            current_date = start_date
            while current_date <= end_date:
                base_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=working_hours[0])
                intervals = (working_hours[1] - working_hours[0]) * 4
                for i in range(intervals):
                    future_dates.append(base_time + timedelta(minutes=15 * i))
                current_date += timedelta(days=1)
            future_dates = pd.to_datetime(future_dates)
        else:
            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        future_predictions = []
        reduction_factor = 1 - (percentage_reduction / 100) if percentage_reduction is not None else 1
        model_type_lower = model_type.lower()
        holiday_dates = [pd.to_datetime(h).date() for h in st.session_state.holidays]
        outlier_dates_set = [pd.to_datetime(o).date() for o in st.session_state.outliers]

        def force_zero(pred, d):
            if st.session_state.predict_saturday_as_zero and d.weekday() == 5:
                return 0
            if st.session_state.predict_sunday_as_zero and d.weekday() == 6:
                return 0
            if st.session_state.predict_weekends_as_zero and d.weekday() >= 5:
                return 0
            if st.session_state.predict_holidays_as_zero and d.date() in holiday_dates:
                return 0
            if d.date() in outlier_dates_set:
                return 0
            return pred

        if model_type_lower in ['arrima', 'sarima']:
            n_steps = len(future_dates)
            forecast = model.forecast(steps=n_steps)
            for i, f_date in enumerate(future_dates):
                weight = weekday_weights.get(f_date.weekday(), 1)
                holiday_weight = 1
                if f_date.date() in holiday_dates:
                    for h in st.session_state.holidays:
                        if pd.to_datetime(h).date() == f_date.date():
                            holiday_weight = st.session_state.holiday_weights[st.session_state.holidays.index(h)] / 100
                            break
                prediction = forecast[i] * reduction_factor
                prediction_rescaled = scaler.inverse_transform(np.array([[prediction]]))[0, 0]
                final_pred = force_zero(prediction_rescaled, f_date)
                final_pred = max(0, final_pred) * weight * holiday_weight
                future_predictions.append(final_pred)
            for idx, d in enumerate(future_dates):
                if st.session_state.predict_saturday_as_zero and d.weekday() == 5:
                    future_predictions[idx] = 0
                if st.session_state.predict_sunday_as_zero and d.weekday() == 6:
                    future_predictions[idx] = 0
                if st.session_state.predict_weekends_as_zero and d.weekday() >= 5:
                    future_predictions[idx] = 0
                if st.session_state.predict_holidays_as_zero and d.date() in holiday_dates:
                    future_predictions[idx] = 0
            return future_dates, future_predictions
        else:
            last_sequence = data_scaled[-self.seq_length:]
            for f_date in future_dates:
                weight = weekday_weights.get(f_date.weekday(), 1)
                holiday_weight = 1
                if f_date.date() in holiday_dates:
                    for h in st.session_state.holidays:
                        if pd.to_datetime(h).date() == f_date.date():
                            holiday_weight = st.session_state.holiday_weights[st.session_state.holidays.index(h)] / 100
                            break
                if force_zero(1, f_date) == 0:
                    prediction = np.array([0])
                else:
                    ml_models = ['catboost','ensemble', 'svr', 'adaboost', 'huber', 'ridge', 'elasticnet', 'lightgbm']
                    nn_models = ['lstm','tcn',  'mlp','nhits','gru','llama','ttm']
                    if model_type_lower in ml_models:
                        prediction = model.predict(last_sequence.reshape(1, -1))
                    elif model_type_lower in nn_models:
                        prediction = model.predict(last_sequence.reshape(1, self.seq_length, 1))
                    else:
                        prediction = model.predict(last_sequence.reshape(1, -1))
                    if prediction.ndim == 1:
                        prediction = prediction.reshape(-1, 1)
                    prediction *= reduction_factor
                    last_sequence = np.append(last_sequence[1:], prediction, axis=0).reshape(self.seq_length, 1)
                prediction_rescaled = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0]
                final_pred = force_zero(prediction_rescaled, f_date)
                final_pred = max(0, final_pred) * weight * holiday_weight
                future_predictions.append(final_pred)
            for idx, d in enumerate(future_dates):
                if st.session_state.predict_saturday_as_zero and d.weekday() == 5:
                    future_predictions[idx] = 0
                if st.session_state.predict_sunday_as_zero and d.weekday() == 6:
                    future_predictions[idx] = 0
                if st.session_state.predict_weekends_as_zero and d.weekday() >= 5:
                    future_predictions[idx] = 0
                if st.session_state.predict_holidays_as_zero and d.date() in holiday_dates:
                    future_predictions[idx] = 0
            return future_dates, future_predictions

    def blend_predictions(self, model_preds):
        X_blend = np.column_stack(model_preds)
        meta_learner = Ridge()
        meta_learner.fit(X_blend, np.mean(X_blend, axis=1))
        blended_preds = meta_learner.predict(X_blend)
        return blended_preds


# =======================
# APPLICATION STREAMLIT
# =======================

class Application:
    def __init__(self):
        self.filepath = ""

    def run(self):
        st.sidebar.title("🌟Navigation")
        st.markdown(
            """<style>
            [data-testid="stSidebar"]{background:linear-gradient(to bottom,#004080,#00274d);color:white;padding:20px;border-radius:10px;}
            .sidebar-title{color:white !important;font-size:22px !important;font-weight:bold;text-align:center;}
            [data-testid="stRadio"]{background:rgba(255,255,255,0.2);border-radius:10px;padding:10px;color:white;}
            [role="radiogroup"] label div{background:#ff1744 !important;color:white !important;border-radius:10px;padding:10px;transition:0.3s;}
            [role="radiogroup"] label div:hover{background:#ff4081 !important;}
            </style>""", unsafe_allow_html=True)
        menu = st.sidebar.radio(
            "",
            ["1-Accueil", "2-Chargez Historique", "3-Paramétrages & Prév"],
            index=0
        )
        st.markdown(
            """<style>
            body {
                background: linear-gradient(to bottom, #074e56, #000000);
            }
            .stButton {
                background-color: white !important;
                color: #000 !important;
                border-radius: 5px;
                padding: 10px;
            }
            .stSidebar * {
                color: white !important;
            }
            [role="radiogroup"] label div {
                background: none !important;
                color: white !important;
                border-radius: 5px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease-in-out;
                margin-bottom: 5px !important;
            }
            [role="radiogroup"] label div:hover {
                background: #074e56 !important;
            }
            input {
                width: 100%;
                padding: 10px;
                margin-bottom: 15px;
                border-radius: 5px;
            }
            </style>""", unsafe_allow_html=True)

        if menu == "1-Accueil":
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                # BG.png
                left_banner = load_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/BG.png")
                if left_banner is not None:
                    st.image(left_banner, width=150)
                else:
                    st.write("Bannière Gauche introuvable")
            with col2:
                st.markdown('<div class="banner-container"></div>', unsafe_allow_html=True)
            with col3:
                # banWFM.png
                right_banner = load_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/banWFM.png")
                if right_banner is not None:
                    st.image(right_banner, width=150)
                else:
                    st.write("Bannière Droite introuvable")

            st.markdown("<br><br>", unsafe_allow_html=True)

            # ===== Vidéo de fond (bg-bullet-animation-1920x800.mp4) =====
            try:
                video_url = "https://raw.githubusercontent.com/HIMA2702/HIMA/main/bg-bullet-animation-1920x800.mp4"
                video_bytes = requests.get(video_url).content
                video_b64 = base64.b64encode(video_bytes).decode()
                st.markdown(f"""
                    <style>
                        .video-container {{
                            position: relative;
                            width: 100%;
                            height: 55vh;
                            overflow: hidden;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }}
                        .video-container video {{
                            position: absolute;
                            top: 60%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            min-width: 100%;
                            min-height: 100%;
                            object-fit: cover;
                            filter: brightness(90%);
                        }}
                        .video-text {{
                            position: absolute;
                            color: white;
                            left: 0%;
                            font-size: 25px;
                            font-weight: bold;
                            z-index: 2;
                            max-width: 100%;
                            padding: 20px;
                            background: rgba(0, 0, 100, 0.5);
                            border-radius: 10px;
                        }}
                    </style>
                    <div class="video-container">
                        <video autoplay loop muted>
                            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <div class="video-text">
                            HIMA - High Integrated Moving Algorithms
                            <div style="margin-top: 7px; color: red;"> A smart Way to see the futur</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erreur lors du chargement de la vidéo de fond : {e}")

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <style>
                @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
                @keyframes floating { 0% { transform: translateY(0px); } 50% { transform: translateY(-5px); } 100% { transform: translateY(0px); } }
                .animated-card { padding: 30px; border-radius: 15px; width: 35%; text-align: center; color: white; margin-top: 50px; transition: transform 0.3s ease, background-color 0.3s ease; animation: fadeIn 1s ease-out, floating 3s infinite ease-in-out; }
                .animated-card:nth-child(1) { background: linear-gradient(45deg, #ff758c, #ff7eb3); }
                .animated-card:nth-child(2) { background: linear-gradient(45deg, #7a5fff, #a875ff); }
                .animated-card:nth-child(3) { background: linear-gradient(45deg, #17ead9, #6078ea); }
                .animated-card:hover { transform: scale(1.05); }
                .container { display: flex; justify-content: center; gap: 10px; margin: 10px 0; }
                .title { text-align: center; font-size: 32px; font-weight: bold;margin-top: 0px; margin-bottom: 30px; color: #333; }
                </style>
                <div class="title"></div>
                <div class="title"> </div>
                """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            st.markdown("""
                <div class="container">
                    <div class="animated-card">
                        <h3>Intelcia</h3>
                        <p>L’aventure Intelcia est née de l’audace d’une poignée de visionnaires de transformer un centre de contact d’un pays du Sud en un leader mondial de l’outsourcing. Le tout en ayant un impact réel et positif sur les clients, les collaborateurs et l’ensemble de l’écosystème.</p>
                    </div>
                    <div class="animated-card">
                        <h3>WFM</h3>
                        <p>WLF joue un rôle clé dans la gestion des opérations en assurant une optimisation des ressources et une amélioration continue des performances</p>
                    </div>
                    <div class="animated-card">
                        <h3>📖 HIMA</h3>
                        <p>HIMA est une solution avancée développée par le département WFM d’Intelcia, combinant intelligence artificielle, machine learning et analyses avancées pour optimiser les prévisions et la gestion des effectifs.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""<h2 style="color:#C23665; font-weight: bold;">POURQUOI <span style="color:#C23665;">INTELCIA ?</span></h2>""", unsafe_allow_html=True)
            st.markdown('<p style="font-size: 18px; line-height: 1.6;">Depuis plus de 20 ans, nous accompagnons les entreprises dans leur transformation. Nous les aidons à accélérer leur croissance en leur permettant de se concentrer sur leurs enjeux et leur cœur de métier. Grâce à une association unique d’expertises et de talents, nous développons les solutions les mieux adaptées pour répondre aux défis actuels et futurs de nos clients. Ainsi nous avons un impact durable sur leur compétitivité.</p>', unsafe_allow_html=True)
            st.subheader("")
            with st.container():
                # mapintelcia.png
                photo1 = load_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/mapintelcia.png")
                if photo1:
                    st.image(photo1, caption="", width=600)
                else:
                    st.write("Photo 1 non disponible.")

                # KB.png => lien clickable
                photo_middle_base64 = get_base64_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/KB.png")
                if photo_middle_base64:
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <a href="https://youtu.be/DWOpWu6htY4" target="_blank">
                                <img src="data:image/png;base64,{photo_middle_base64}" alt="Voir la suite" style="width:100%; max-width:600px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.2);">
                            </a>
                            <p style="font-size: 16px; font-weight: bold; color: #C23665;">Voir la suite</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("Image KB non disponible.")

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("""<h2 style="color:#C23665; font-weight: bold;">POURQUOI <span style="color:#C23665;">WFMR ?</span></h2>""", unsafe_allow_html=True)
                st.markdown('<p style="font-size: 18px; line-height: 1.6;">Optimisez les horaires en quelques minutes. Les outils de planification simplifient la planification des équipes et responsabilisent les responsables.</p>', unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">🚀 Prenez des décisions stratégiques basées sur des données précises et des analyses avancées.</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">🤖 "Découvrez **HIMA**, l’application intelligente qui révolutionne la prévision des volumes d’appels grâce aux algorithmes de Machine Learning et Deep Learning, optimisée pour s’adapter aux variations du calendrier et aux tendances historiques avec une précision inégalée."</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">✅ "**HIMA** révolutionne la prévision des volumes d’appels en analysant les tendances historiques et les effets calendaires pour offrir des estimations précises et fiables."</p>""", unsafe_allow_html=True)

                # MR.png => lien clickable
                photo_mr_base64 = get_base64_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/MR.png")
                if photo_mr_base64:
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <a href="https://urlr.me/Kbgv6m" target="_blank">
                                <img src="data:image/png;base64,{photo_mr_base64}" alt="Voir plus" style="width:14cm; height:14cm; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.2);">
                            </a>
                            <p style="font-size: 16px; font-weight: bold; color: #C23665;">Cliquez ici pour voir plus</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("Image MR non disponible.")

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)

                # photo2.png
                photo2 = load_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/photo2.png")
                if photo2:
                    st.image(photo2, caption="", width=1000)
                else:
                    st.write("Photo 2 non disponible.")

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("""<h2 style="color:#C23665; font-weight: bold;">POURQUOI <span style="color:#C23665;">HIMA ?</span></h2>""", unsafe_allow_html=True)
                st.markdown('<p style="font-size: 18px; line-height: 1.6;">Optimisez les horaires en quelques minutes. Les outils de planification simplifient la gestion des équipes et responsabilisent les managers.</p>', unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">🚀 Prenez des décisions stratégiques basées sur des données précises et des analyses avancées.</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">🤖 Grâce à l'IA, anticipez les besoins en effectifs et ajustez les plannings en temps réel.</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">✅ Optimisez les horaires en quelques minutes. Les outils de planification simplifient la gestion des équipes et responsabilisent les managers.</p>""", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="animated-card" style="background: linear-gradient(45deg, #ff9a8b, #ff6a88);">
                        <h3>📊 Analyse</h3>
                        <p>Exploitez l'IA pour mieux comprendre, Avec l'analyse, les connaissances s'étendent. Les données dévoilent ce que l'on défend, Pour des choix éclairés, un avenir grandissant.</p>
                    </div>
                    <div class="animated-card" style="background: linear-gradient(45deg, #5b86e5, #36d1dc);">
                        <h3>💡 Intelligence</h3>
                        <p>Découvrez le potentiel de l'IA pour enrichir vos connaissances, Avec une analyse pointue, faites briller votre expertise. Les données deviennent vos alliées, révélant des tendances, Pour des choix stratégiques, construisez un avenir prometteur.</p>
                    </div>
                    <div class="animated-card" style="background: linear-gradient(45deg, #f4c4f3, #fc67fa);">
                        <h3>⚙️ Forecasting</h3>
                        <p>Ces algorithmes de pointe, basés sur le machine learning et le deep learning, sont spécialement conçus pour optimiser le forecasting des appels, améliorant ainsi la précision des prévisions et la gestion des ressources.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("""<h2 style="color:#C23665; font-weight: bold;">REJOIGNEZ <span style="color:#C23665;">-NOUS ?</span></h2>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">Groupe multi-métier, multiculturel, multilingues, nous construisons avec les entreprises et les administrations des expériences utilisateurs uniques pour un lendemain meilleur.</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">Nous cultivons l’audace, la créativité et l’excellence, mais aussi la bienveillance et la solidarité. Nous rejoindre, c’est adhérer à ces valeurs qui sont au cœur de notre réussite.</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">🤖 Tout a été pensé pour leur permettre d’apprendre, de grandir et de s’épanouir au sein d’un leader mondial qui fait de l’agilité son fer de lance.</p>""", unsafe_allow_html=True)
                st.markdown("""<p style="font-size: 18px; line-height: 1.6;">Nous avons 20 ans, et pour longtemps encore.</p>""", unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)

                # Humain_Robot.mp4
                try:
                    video_url = "https://raw.githubusercontent.com/HIMA2702/HIMA/main/Humain_Robot.mp4"
                    video_bytes = requests.get(video_url).content
                    video_b64 = base64.b64encode(video_bytes).decode()
                    video_html = f""" 
                    <video width="700" controls autoplay loop muted style="display: block; margin: 0 auto;">
                        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
                        Votre navigateur ne supporte pas la balise vidéo.
                    </video> 
                    """
                    st.markdown(video_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur lors du chargement de la vidéo : {e}")

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<br><br>", unsafe_allow_html=True)

                # Footer
                linkedin_base64 = get_base64_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/linkedin.png")
                intelcia_base64 = get_base64_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/Instagram_icon.png")
                st.markdown(f"""
                <style>
                .footer {{
                    background: linear-gradient(to right, #dd6aa4, #87559e);
                    padding: 40px;
                    color: white;
                    text-align: left;
                    width: 100%;
                    position: relative;
                }}
                .footer-container {{
                    display: flex;
                    justify-content: space-between;
                    max-width: 1600px;
                    margin: auto;
                }}
                .footer-section {{
                    flex: 1;
                    margin: 0 30px;
                }}
                h3 {{
                    margin-bottom: 10px;
                    font-size: 16px;
                }}
                .footer a {{
                    color: white;
                    text-decoration: none;
                    font-size: 14px;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                .footer .footer-end {{
                    text-align: center;
                    margin-top: 20px;
                    font-size: 12px;
                }}
                .footer .social-links {{
                    margin-top: 10px;
                }}
                .footer .social-links a {{
                    margin-right: 15px;
                }}
                </style>
                <div class="footer">
                    <div class="footer-container">
                        <div class="footer-section">
                            <h3>NOUS CONNAÎTRE</h3>
                            <a href="https://www.intelcia.com/fr/decouvrez-intelcia" target="_blank">À propos d'Intelcia</a><br>
                            <a href="https://www.intelcia.com/fr/mentions-legales" target="_blank">Politique de confidentialité</a><br>
                            <a href="https://www.intelcia.com/fr/mentions-legales" target="_blank">Mentions légales</a>
                        </div>
                        <div class="footer-section">
                            <h3>NOUS CONTACTER</h3>
                            <a href="https://www.intelcia.com/fr/contactez-nous" target="_blank">Contact</a><br>
                            <a href="https://www.intelcia.com/fr/adresse" target="_blank">Adresse de nos sites</a>
                        </div>
                    </div>
                    <div class="footer-end">
                        © 2015 Intelcia group | Talent Academy<br>
                        <div class="social-links">
                            <a href="https://www.intelcia.com/it" target="_blank"><img src="data:image/png;base64,{intelcia_base64}" alt="Intelcia" style="height: 20px;"></a>
                            <a href="https://www.linkedin.com/company/intelcia/posts/?feedView=all" target="_blank"><img src="data:image/png;base64,{linkedin_base64}" alt="LinkedIn" style="height: 20px;"></a>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif menu == "2-Chargez Historique":
            st.sidebar.success("""
            ## 📖 info fichier d'upload'
            - **Upload 📂** :  Veuillez télécharger le fichier CSV contenant deux colonnes (date et volume) avec un point-virgule ";" comme délimiteur.
            ---
            """)
            st.header("📊 Chargement et préparation des données")
            uploaded_file = st.file_uploader("Charger Fichier CSV", type=["csv"])
            if uploaded_file is not None:
                st.session_state.filepath = uploaded_file
                st.success(f"Fichier chargé : {uploaded_file.name}")
                try:
                    temp_file = "temp_uploaded_file.csv"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    predictor = FTTHPredictor(seq_length=20, prediction_type=st.session_state.prediction_type)
                    data, data_scaled, scaler, weekday_weights = predictor.load_and_preprocess_data(
                        temp_file,
                        holidays_list=st.session_state.holidays if 'holidays' in st.session_state else [],
                        holiday_weights=st.session_state.holiday_weights if 'holiday_weights' in st.session_state else [],
                        outlier_dates=st.session_state.outliers if 'outliers' in st.session_state else [],
                        predict_holidays_as_zero=False,
                        predict_weekends_as_zero=False,
                        predict_saturday_as_zero=False,
                        predict_sunday_as_zero=False,
                        working_hours=(8, 18)
                    )
                    if data is not None:
                        st.subheader("")
                        import seaborn as sns
                        import matplotlib.pyplot as plt

                        data['day_name'] = data['date'].dt.day_name()
                        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        day_colors = {
                            'Monday': '#3498db',
                            'Tuesday': '#2ecc71',
                            'Wednesday': '#f1c40f',
                            'Thursday': '#e67e22',
                            'Friday': '#9b59b6',
                            'Saturday': '#e74c3c',
                            'Sunday': '#34495e'
                        }
                        palette = [day_colors[d] for d in ordered_days]
                        with st.expander("Box-plot par jour de la semaine"):
                            fig, ax = plt.subplots(figsize=(8,6))
                            sns.boxplot(
                                x='day_name',
                                y='volume',
                                data=data,
                                order=ordered_days,
                                palette=palette,
                                ax=ax
                            )
                            ax.set_title("Répartition du Volume par Jour de la Semaine")
                            ax.set_xlabel("Jour de la Semaine")
                            ax.set_ylabel("Volume")
                            st.pyplot(fig)

                    else:
                        st.error("Erreur lors du prétraitement des données.")
                except Exception as e:
                    logging.error(f"Erreur lors du chargement du fichier : {e}")
                    st.error(f"Erreur lors du chargement du fichier : {e}")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                st.info("Veuillez charger un fichier CSV pour continuer.")

        elif menu == "3-Paramétrages & Prév":
            st.header("🔮 Prévisions")
            st.sidebar.markdown("""
                ## ⚙️ Paramètres de Prévision
                - **Jours Fériés** : Indiquez les dates où l'activité est réduite ou nulle.
                - **Poids** : Ajustez l'impact d’un jour férié.
                - **Valeurs Aberrantes** : Détection et correction des anomalies.
                - **Prédictions à 0** : Option pour fixer à 0 certains jours.
                ---
            """, unsafe_allow_html=True)
            st.markdown('<div style="background-color:#085a89; border:1px solid #4CAF50; padding:10px; border-radius:5px; margin-bottom:10px;">', unsafe_allow_html=True)
            st.markdown("### Gestion des Jours Fériés")
            if 'holidays' not in st.session_state:
                st.session_state.holidays = []
            if 'holiday_weights' not in st.session_state:
                st.session_state.holiday_weights = []

            col1, col2 = st.columns(2)
            with col1:
                holiday_date = st.date_input("Ajouter un jour férié :", key="holiday_date")
            with col2:
                if st.button("Déclarer Jour Férié"):
                    formatted_holiday = holiday_date.strftime('%Y-%m-%d')
                    if formatted_holiday not in st.session_state.holidays:
                        st.session_state.holidays.append(formatted_holiday)
                        st.session_state.holiday_weights.append(100)
                        st.success(f"Jour férié ajouté : {formatted_holiday}")

            if st.session_state.holidays:
                st.subheader("Liste des Jours Fériés avec Poids")
                for idx, holiday in enumerate(st.session_state.holidays):
                    col_a, col_b, col_c = st.columns([3, 2, 1])
                    with col_a:
                        st.write(holiday)
                    with col_b:
                        weight = st.slider(f"Poids pour {holiday}", 0, 100, st.session_state.holiday_weights[idx], key=f"weight_{holiday}")
                        st.session_state.holiday_weights[idx] = weight
                    with col_c:
                        if st.button(f"Remove {holiday}", key=f"remove_{holiday}"):
                            st.session_state.holidays.pop(idx)
                            st.session_state.holiday_weights.pop(idx)
                            st.success(f"Jour férié supprimé : {holiday}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="background-color:#085a89; border:1px solid #4CAF50; padding:10px; border-radius:5px; margin-bottom:10px;">', unsafe_allow_html=True)
            st.markdown("### Gestion des Valeurs Hors Normes")

            if 'outliers' not in st.session_state:
                st.session_state.outliers = []

            col1, col2 = st.columns(2)
            with col1:
                outlier_date = st.date_input("Ajouter une valeur aberrante :", key="outlier_date")
            with col2:
                if st.button("Ajouter la valeur aberrante"):
                    formatted_outlier = outlier_date.strftime('%Y-%m-%d')
                    if formatted_outlier not in st.session_state.outliers:
                        st.session_state.outliers.append(formatted_outlier)
                        st.success(f"Valeur aberrante ajoutée : {formatted_outlier}")

            if st.session_state.outliers:
                st.subheader("Liste des Valeurs Aberrantes")
                for idx, outlier in enumerate(st.session_state.outliers):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(outlier)
                    with col_b:
                        if st.button(f"Remove {outlier}", key=f"remove_outlier_{outlier}"):
                            st.session_state.outliers.pop(idx)
                            st.success(f"Valeur aberrante supprimée : {outlier}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="background-color:#085a89; border:1px solid #4CAF50; padding:10px; border-radius:5px; margin-bottom:10px;">', unsafe_allow_html=True)
            st.markdown("### Granularité")
            if 'prediction_type' not in st.session_state:
                st.session_state.prediction_type = "Jour"
            selected_prediction_type = st.selectbox("Choisir le type de prédiction :", ["Jour", "tranche"], key="prediction_type")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="background-color:#085a89; border:1px solid #4CAF50; padding:10px; border-radius:5px; margin-bottom:10px;">', unsafe_allow_html=True)
            st.markdown("### Amplitude Horaires (De - à)")
            col1, col2 = st.columns(2)
            with col1:
                if 'working_hours_start' not in st.session_state:
                    st.session_state.working_hours_start = "08:00"
                selected_working_start = st.selectbox("Début :", [f"{h:02d}:{m:02d}" for h in range(24) for m in [0,15,30,45]], index=8, key="working_hours_start")
            with col2:
                if 'working_hours_end' not in st.session_state:
                    st.session_state.working_hours_end = "18:00"
                selected_working_end = st.selectbox("Fin :", [f"{h:02d}:{m:02d}" for h in range(24) for m in [0,15,30,45]], index=18, key="working_hours_end")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="background-color:#085a89; border:1px solid #4CAF50; padding:10px; border-radius:5px; margin-bottom:10px;">', unsafe_allow_html=True)
            st.markdown("### Gestion JF & Week-end")
            if 'predict_holidays_as_zero' not in st.session_state:
                st.session_state.predict_holidays_as_zero = False
            if 'predict_saturday_as_zero' not in st.session_state:
                st.session_state.predict_saturday_as_zero = False
            if 'predict_sunday_as_zero' not in st.session_state:
                st.session_state.predict_sunday_as_zero = False
            if 'predict_weekends_as_zero' not in st.session_state:
                st.session_state.predict_weekends_as_zero = False

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                predict_holidays_as_zero = st.checkbox("Prédire JF comme 0", value=False, key="predict_holidays_as_zero")
            with col2:
                predict_saturday_as_zero = st.checkbox("Prédire samedis comme 0", value=False, key="predict_saturday_as_zero")
            with col3:
                predict_sunday_as_zero = st.checkbox("Prédire dimanches comme 0", value=False, key="predict_sunday_as_zero")
            with col4:
                predict_weekends_as_zero = st.checkbox("Prédire week-ends comme 0", value=False, key="predict_weekends_as_zero")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="border-top: 6px solid #004080; margin-top: 15px; margin-bottom: 25px;"></div>', unsafe_allow_html=True)

            if "filepath" not in st.session_state or st.session_state.filepath is None:
                st.error("Veuillez d'abord charger un fichier dans la section 'Chargement des données'.")
            else:
                start_date = st.date_input("Date de Début Prévision :", key="start_date")
                end_date = st.date_input("Date de Fin Prévision :", key="end_date")

                if st.button("🚀 Lancer la prévision"):
                    try:
                        if start_date >= end_date:
                            raise ValueError("La date de début doit être avant la date de fin.")

                        seq_length = 5
                        all_holidays = [pd.Timestamp(holiday) for holiday in st.session_state.holidays]
                        holiday_weights = st.session_state.holiday_weights
                        outlier_dates = [pd.Timestamp(outlier) for outlier in st.session_state.outliers]
                        predictor = FTTHPredictor(seq_length, st.session_state.prediction_type)

                        temp_file = "temp_uploaded_file.csv"
                        with open(temp_file, "wb") as f:
                            f.write(st.session_state.filepath.getbuffer())

                        data, data_scaled, scaler, weekday_weights = predictor.load_and_preprocess_data(
                            temp_file,
                            holidays_list=all_holidays,
                            holiday_weights=holiday_weights,
                            outlier_dates=outlier_dates,
                            predict_holidays_as_zero=st.session_state.predict_holidays_as_zero,
                            predict_weekends_as_zero=st.session_state.predict_weekends_as_zero,
                            predict_saturday_as_zero=st.session_state.predict_saturday_as_zero,
                            predict_sunday_as_zero=st.session_state.predict_sunday_as_zero,
                            working_hours=(int(st.session_state.working_hours_start.split(':')[0]),
                                           int(st.session_state.working_hours_end.split(':')[0]))
                        )

                        if data is None:
                            raise ValueError("Prétraitement des données échoué.")
                        st.subheader("📊 Avancement...")

                        if len(data) <= seq_length:
                            seq_length = max(1, len(data) - 1)
                        train_data, test_data = predictor.split_train_test(data_scaled, train_ratio=0.8)
                        X_train_seq, y_train_seq = predictor.create_sequences(train_data)
                        X_test_seq, y_test_seq = predictor.create_sequences(test_data)

                        if len(X_train_seq) == 0 or len(y_train_seq) == 0:
                            raise ValueError("Pas assez de séquences créées. Veuillez fournir plus de points de données ou ajuster la longueur de la séquence.")

                        models = {}
                        progress_bar = st.progress(0)
                        progress_text = st.empty()

                        # Sélection de quelques modèles en exemple
                        if st.session_state.prediction_type == 'tranche':
                            model_types = ['svr']  # par exemple
                        else:
                            model_types = ['huber', 'elasticnet']  # ajustez selon vos besoins

                        total_models = len(model_types)
                        completed = 0

                        with ThreadPoolExecutor() as executor:
                            future_models = {executor.submit(predictor.train_model, X_train_seq, y_train_seq, m_type): m_type for m_type in model_types}
                            for future in as_completed(future_models):
                                m_type = future_models[future]
                                try:
                                    models[m_type] = future.result()
                                except Exception as e:
                                    logging.error(f"Erreur lors de l'entraînement de {m_type} : {e}")
                                    st.error(f"Erreur lors de l'entraînement de {m_type} : {e}")
                                completed += 1
                                progress = int((completed / total_models) * 100)
                                progress_bar.progress(progress)
                                progress_text.write(f"Progression : {progress}%")
                            progress_text.write("✅ Entraînement terminé !")
                            st.success("Tous les modèles ont été entraînés avec succès ! 🎉")

                        best_model_name = None
                        best_r2 = float('-inf')
                        results = {}
                        model_preds = []

                        with st.spinner("📊 Évaluation des modèles..."):
                            for m_name, model in models.items():
                                if model is None:
                                    continue
                                try:
                                    if m_name.lower() in ["arrima", "sarima"]:
                                        n_steps = X_test_seq.shape[0]
                                        y_pred = model.forecast(steps=n_steps)
                                    elif m_name.lower() in ["catboost", "svr", "adaboost", "huber", "ridge", "ensemble", "elasticnet", 'lightgbm']:
                                        X_test_reshaped = X_test_seq.reshape((X_test_seq.shape[0], -1))
                                        y_pred = model.predict(X_test_reshaped)
                                    else:
                                        y_pred = model.predict(X_test_seq)

                                    y_test_inv = scaler.inverse_transform(y_test_seq.reshape(-1, 1))
                                    y_pred_inv = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))
                                    y_pred_inv = np.clip(y_pred_inv, 0, None)

                                    r2 = r2_score(y_test_inv, y_pred_inv)
                                    mae = mean_absolute_error(y_test_inv, y_pred_inv)
                                    results[m_name] = {"R2": r2, "MAE": mae}
                                    model_preds.append(y_pred_inv.flatten())

                                    if r2 > best_r2:
                                        best_r2 = r2
                                        best_model_name = m_name
                                except Exception as e:
                                    logging.error(f"Erreur lors de la prédiction avec {m_name} : {e}")
                                    st.error(f"Erreur lors de la prédiction avec {m_name} : {e}")

                        if best_model_name:
                            future_dates, future_predictions = predictor.predict_future(
                                models[best_model_name], data_scaled, scaler, start_date, end_date, all_holidays,
                                holiday_weights, outlier_dates,
                                st.session_state.predict_holidays_as_zero,
                                st.session_state.predict_weekends_as_zero,
                                st.session_state.predict_saturday_as_zero,
                                st.session_state.predict_sunday_as_zero,
                                (int(st.session_state.working_hours_start.split(':')[0]), int(st.session_state.working_hours_end.split(':')[0])),
                                model_type=best_model_name, weekday_weights=weekday_weights
                            )
                            st.success("✅ Prédiction terminée. Résultats sauvegardés.")
                            st.write(f"Résultats : Meilleur modèle {best_model_name} avec R² : {best_r2 * 100:.2f}% et MAE : {results[best_model_name]['MAE']:.2f}")

                            df_predictions = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})
                            df_predictions["Day"] = df_predictions["Date"].dt.date

                            # Correction post-prediction
                            holiday_dates = [pd.to_datetime(h).date() for h in st.session_state.holidays]
                            for idx, d in enumerate(future_dates):
                                if st.session_state.predict_saturday_as_zero and d.weekday() == 5:
                                    future_predictions[idx] = 0
                                if st.session_state.predict_sunday_as_zero and d.weekday() == 6:
                                    future_predictions[idx] = 0
                                if st.session_state.predict_weekends_as_zero and d.weekday() >= 5:
                                    future_predictions[idx] = 0
                                if st.session_state.predict_holidays_as_zero and d.date() in holiday_dates:
                                    future_predictions[idx] = 0

                            df_predictions = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})
                            df_history = pd.DataFrame({
                                "Date": list(data["date"]) + list(future_dates),
                                "Volume": list(data["volume"]) + list(future_predictions),
                                "Type": ["Historique"] * len(data) + ["Prévision"] * len(future_dates)
                            })

                            fig_forecast = go.Figure()
                            fig_forecast.add_trace(go.Scatter(
                                x=df_predictions["Date"], y=df_predictions["Forecast"],
                                mode="lines+markers", name="Forecast",
                                marker=dict(color="lightgreen")
                            ))
                            fig_forecast.update_layout(
                                title="Prévisions Interactives",
                                xaxis_title="Date",
                                yaxis_title="Volume",
                                paper_bgcolor="#0a2832",
                                plot_bgcolor="#0a2832",
                                font=dict(color="white")
                            )
                            st.plotly_chart(fig_forecast, width=600)

                            df_tranche = df_predictions.copy()
                            df_tranche["Hour"] = df_tranche["Date"].dt.hour

                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                                historical_start = data["date"].min().date()
                                historical_end = data["date"].max().date()
                                df_dates = pd.DataFrame({
                                    "Type": ["Historique", "Prévision"],
                                    "Date début": [str(historical_start), str(start_date)],
                                    "Date fin": [str(historical_end), str(end_date)]
                                })
                                df_dates.to_excel(writer, sheet_name="Dates", index=False)

                                # Graph "Historique vs Prévision"
                                fig, ax = plt.subplots(figsize=(10, 5))
                                df_history[df_history["Type"] == "Historique"].plot(
                                    x="Date", y="Volume", ax=ax, color="blue", label="Historique"
                                )
                                df_history[df_history["Type"] == "Prévision"].plot(
                                    x="Date", y="Volume", ax=ax, color="green", label="Prévision"
                                )
                                ax.set_title("Historique vs Prévision")
                                ax.legend()

                                graph_bytes = io.BytesIO()
                                plt.savefig(graph_bytes, format="png", bbox_inches="tight")
                                plt.close(fig)
                                graph_bytes.seek(0)
                                img_data = graph_bytes.read()

                                workbook = writer.book
                                worksheet_graph = workbook.add_worksheet("Graph")
                                worksheet_graph.insert_image("A1", "graph.png", {'image_data': io.BytesIO(img_data)})

                                df_history.to_excel(writer, sheet_name="Historique & Graph", index=False)
                                worksheet_history = writer.sheets["Historique & Graph"]
                                worksheet_history.insert_image("G2", "graph.png", {'image_data': io.BytesIO(img_data)})

                                df_daily = data.groupby(data["date"].dt.date)["volume"].sum().reset_index()
                                df_daily.columns = ["Date", "Volume"]
                                df_daily.to_excel(writer, sheet_name="Volume Jour", index=False)

                            output.seek(0)
                            excel_data = output.getvalue()
                            timestamp = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
                            st.download_button(
                                label="📥 Télécharger les Prédictions",
                                data=excel_data,
                                file_name=f"HIMA_prediction_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col1:
                                # CE.png
                                banner = load_image_from_url("https://raw.githubusercontent.com/HIMA2702/HIMA/main/CE.png")
                                if banner:
                                    new_size = (banner.width // 2, banner.height // 2)
                                    banner_resized = banner.resize(new_size, Image.Resampling.LANCZOS)
                                    st.image(banner_resized, caption="© 2025 HIMA_By_WFM", width=new_size[0])
                                else:
                                    logging.error(f"Erreur lors du chargement du bandeau CE.png")

                    except Exception as e:
                        st.error(f"Erreur lors de la prévision : {e}")
                    finally:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

if __name__ == "__main__":
    # Initialisation de quelques variables de session au lancement
    if 'holidays' not in st.session_state:
        st.session_state.holidays = []
    if 'holiday_weights' not in st.session_state:
        st.session_state.holiday_weights = []
    if 'outliers' not in st.session_state:
        st.session_state.outliers = []
    if 'predictions_csv' not in st.session_state:
        st.session_state.predictions_csv = None
    if 'prediction_type' not in st.session_state:
        st.session_state.prediction_type = "Jour"
    if 'working_hours_start' not in st.session_state:
        st.session_state.working_hours_start = "08:00"
    if 'working_hours_end' not in st.session_state:
        st.session_state.working_hours_end = "18:00"
    if 'predict_holidays_as_zero' not in st.session_state:
        st.session_state.predict_holidays_as_zero = False
    if 'predict_saturday_as_zero' not in st.session_state:
        st.session_state.predict_saturday_as_zero = False
    if 'predict_sunday_as_zero' not in st.session_state:
        st.session_state.predict_sunday_as_zero = False
    if 'predict_weekends_as_zero' not in st.session_state:
        st.session_state.predict_weekends_as_zero = False

    app = Application()
    app.run()
