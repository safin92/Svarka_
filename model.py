import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

class WeldingModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.scaler = MinMaxScaler()
        self.models = {}
        self.metrics = {}

    def prepare_data(self):
        X = self.df[['IW', 'IF', 'VW', 'FP']]
        y_depth = self.df['Depth']
        y_width = self.df['Width']

        # Масштабирование данных
        X_scaled = self.scaler.fit_transform(X)

        # Разделение данных на обучающую и тестовую выборки
        self.X_train_depth, self.X_test_depth, self.y_train_depth, self.y_test_depth = train_test_split(X_scaled, y_depth, test_size=0.2, random_state=42)
        self.X_train_width, self.X_test_width, self.y_train_width, self.y_test_width = train_test_split(X_scaled, y_width, test_size=0.2, random_state=42)

    def train_models(self):
        # Градиентный бустинг для глубины
        gb_model_depth = GradientBoostingRegressor(random_state=42)
        gb_model_depth.fit(self.X_train_depth, self.y_train_depth)
        self.models['Gradient Boosting (Depth)'] = gb_model_depth

        # Градиентный бустинг для ширины
        gb_model_width = GradientBoostingRegressor(random_state=42)
        gb_model_width.fit(self.X_train_width, self.y_train_width)
        self.models['Gradient Boosting (Width)'] = gb_model_width

    def evaluate_models(self):
        for name, model in self.models.items():
            if 'Depth' in name:
                y_pred = model.predict(self.X_test_depth)
                self.metrics[name] = {
                    'MSE': mean_squared_error(self.y_test_depth, y_pred),
                    'R²': r2_score(self.y_test_depth, y_pred)
                }
            else:
                y_pred = model.predict(self.X_test_width)
                self.metrics[name] = {
                    'MSE': mean_squared_error(self.y_test_width, y_pred),
                    'R²': r2_score(self.y_test_width, y_pred)
                }

    def get_metrics(self):
        return self.metrics