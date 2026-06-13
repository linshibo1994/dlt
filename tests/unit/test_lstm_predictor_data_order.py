#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from backend.app.predictors.deep_learning.models.lstm_predictor import LSTMPredictor


def _sample_descending_data():
    return pd.DataFrame(
        [
            {"issue": "26003", "date": "2026-01-07", "front_balls": "03,04,05,06,07", "back_balls": "03,04"},
            {"issue": "26002", "date": "2026-01-05", "front_balls": "02,03,04,05,06", "back_balls": "02,03"},
            {"issue": "26001", "date": "2026-01-03", "front_balls": "01,02,03,04,05", "back_balls": "01,02"},
        ]
    )


def test_lstm_training_data_is_chronological():
    predictor = LSTMPredictor(config={"sequence_length": 2})

    ordered = predictor._order_lottery_dataframe(_sample_descending_data(), ascending=True)

    assert ordered["issue"].tolist() == ["26001", "26002", "26003"]


def test_lstm_latest_window_keeps_latest_periods_in_chronological_order():
    predictor = LSTMPredictor(config={"sequence_length": 2})

    latest_window = predictor._latest_chronological_window(_sample_descending_data())

    assert latest_window["issue"].tolist() == ["26002", "26003"]
