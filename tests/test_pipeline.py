"""Автоматичні тести основних модулів застосунку.

Запуск:  python -m unittest discover tests -v
"""

import unittest

import numpy as np
import pandas as pd

from src import config, data_loader, models, preprocessing


def make_sample(n=600, seed=0):
    """Невеликий штучний набір даних зі структурою AI4I для перевірки логіки."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    speed = rng.normal(1540, 180, n)
    torque = np.clip(40 - (speed - 1540) * 0.05 + rng.normal(0, 4, n), 4, 76)
    wear = rng.integers(0, 254, n)
    osf = (wear * torque > 11000).astype(int)
    df = pd.DataFrame({
        "udi": range(n), "product_id": [f"L{i}" for i in range(n)],
        "type": rng.choice(["L", "M", "H"], n),
        "air_temp": air, "process_temp": air + 10 + rng.normal(0, 1, n),
        "rot_speed": speed, "torque": torque, "tool_wear": wear,
        "failure": osf, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": osf, "RNF": 0,
    })
    return df


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = make_sample()

    def test_cleaning_removes_duplicates_and_conflicts(self):
        dirty = pd.concat([self.df, self.df.iloc[:5]])
        dirty.loc[dirty.index[10], "failure"] = 1  # несправність без типу
        dirty.loc[dirty.index[10], "OSF"] = 0
        clean, _ = preprocessing.clean_data(dirty)
        self.assertLessEqual(len(clean), len(self.df) - 1)
        self.assertNotIn("udi", clean.columns)

    def test_feature_engineering(self):
        data = preprocessing.add_features(self.df)
        row = data.iloc[0]
        self.assertAlmostEqual(row["temp_diff"], row["process_temp"] - row["air_temp"])
        self.assertAlmostEqual(row["power"], row["torque"] * row["rot_speed"] * 2 * np.pi / 60)

    def test_correlated_features_are_removed(self):
        data = preprocessing.add_features(self.df)
        corr = preprocessing.correlation_matrix(data)
        selected, dropped = preprocessing.select_features(corr)
        for a in selected:
            for b in selected:
                if a != b:
                    self.assertLess(abs(corr.loc[a, b]), config.CORR_THRESHOLD)
        self.assertTrue(dropped)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clean, _ = preprocessing.clean_data(make_sample(800))
        cls.data = preprocessing.add_features(clean)
        cls.features = ["type_code", "rot_speed", "torque", "tool_wear", "temp_diff"]
        cls.result = models.train_model(cls.data, cls.features)

    def test_metrics_in_range(self):
        for value in self.result["metrics"].values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        self.assertGreater(self.result["metrics"]["ROC-AUC"], 0.8)

    def test_predict_new_object(self):
        new = preprocessing.add_features(pd.DataFrame([{
            "type": "L", "air_temp": 300, "process_temp": 310, "rot_speed": 1300,
            "torque": 65, "tool_wear": 230}]))
        out = models.predict(self.result, new)
        self.assertEqual(out.iloc[0]["Прогноз"], "Несправність")


class TestRealDataset(unittest.TestCase):
    def test_dataset_structure(self):
        if not data_loader.dataset_exists():
            self.skipTest("Набір даних ще не завантажено")
        df = data_loader.read_raw()
        self.assertEqual(len(df), 10000)
        self.assertTrue(set(config.COLUMN_MAP.values()).issubset(df.columns))


if __name__ == "__main__":
    unittest.main()
