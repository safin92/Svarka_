from flask import Flask, render_template, request
from model import WeldingModel
import numpy as np

app = Flask(__name__)

# Загрузите и подготовьте модель
data_path = 'ebw_data.csv'  # Укажите путь к вашему CSV-файлу
welding_model = WeldingModel(data_path)
welding_model.prepare_data()
welding_model.train_models()
welding_model.evaluate_models()

@app.route('/')
def index():
    metrics = welding_model.get_metrics()
    return render_template('index.html', metrics=metrics, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы
    IW = float(request.form['IW'])
    IF = float(request.form['IF'])
    VW = float(request.form['VW'])
    FP = float(request.form['FP'])

    # Подготовка данных для предсказания
    input_data = np.array([[IW, IF, VW, FP]])
    scaled_data = welding_model.scaler.transform(input_data)

    # Выполнение предсказаний с использованием градиентного бустинга
    depth_pred = welding_model.models['Gradient Boosting (Depth)'].predict(scaled_data)[0]
    width_pred = welding_model.models['Gradient Boosting (Width)'].predict(scaled_data)[0]

    metrics = welding_model.get_metrics()
    return render_template('index.html', metrics=metrics, prediction={'depth': depth_pred, 'width': width_pred})

if __name__ == '__main__':
    app.run(debug=True)
