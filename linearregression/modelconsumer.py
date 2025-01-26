import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modelhandler import ModelHandler, LocalModelHandler, AzureModelHandler

model_handler: ModelHandler = AzureModelHandler("linear_regression")

def predict(model: LinearRegression, surface: int, rooms:int, bathrooms:int, elevator: bool, terrace: bool, garage: bool, postal_code:int) -> np.ndarray:
    new_data = {
        "surface": [surface],
        "rooms": [rooms],
        "bathrooms": [bathrooms],
        "elevator": [int(elevator)],
        "terrace": [int(terrace)],
        "garage": [int(garage)],
        f"postal_code_{postal_code}": [1]
    }

    # Convertir a DataFrame de pandas
    new_df = pd.DataFrame(new_data)

    # Asegurarse de que todas las columnas estén presentes, rellenando con ceros las que falten
    new_df = new_df.reindex(columns=features, fill_value=0)

    # Reordenar las columnas para que coincidan con el conjunto de entrenamiento
    new_df = new_df[features]

    # Hacer predicciones con el modelo cargado
    predictions = model.predict(new_df)

    # Mostrar las predicciones
    print("Predictions:", predictions)


if __name__ == '__main__':
    model, features = model_handler.load_model()
    predict(
        model = model,
        surface = 80,
        rooms = 3,
        bathrooms = 1,
        elevator = True,
        terrace = False,
        garage = False,
        postal_code = 36211
    )