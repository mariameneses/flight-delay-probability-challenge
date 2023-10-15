"""
Module for prediction model of the probability of delay for a flight
"""

from typing import Tuple, Union, List
from datetime import datetime
from os.path import exists

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


class DelayModel:
    " Implement prediction model "
    def __init__(
        self
    ):
        self.model_path = './challenge/models/delay_pred.json'
        if exists(self.model_path):
            self._model = xgb.XGBClassifier()
            self._model.load_model(self.model_path)
        else:
            self._model = None
        self._threshold_in_minutes = 15
        self._top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def _get_min_diff(
            self,
            element: pd.Series
        ) -> float:
        """
        Get difference in minutes between Fecha-O and Fecha-I

        Args:
            element (pd.Series): DataFrame's row whose index is the DataFrame's
            columns. 

        Returns:
            float : calculated difference.
        """
        fecha_o = datetime.strptime(element['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(element['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
        )
        for col in self._top_10_features:
            if col not in features.columns:
                features.loc[:, col] = 0
        features = features[self._top_10_features]
        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data['delay'] = np.where(
                data['min_diff'] > self._threshold_in_minutes, 1, 0
            )
            target = data[[target_column]]
            return features, target
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, _, y_train, _ = train_test_split(
            features, target.iloc[:, 0],
            test_size=0.33, random_state=42
        )
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0/n_y1
        model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        model.fit(x_train, y_train)
        model.save_model(self.model_path)
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        y_preds = self._model.predict(features)
        return y_preds.tolist()
