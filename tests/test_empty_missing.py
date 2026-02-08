import os
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import DataLoader
from src.inventory_analysis import ProductAnalyzer
from src.recommendations import RecommendationsEngine
from src.trend_analysis import TrendAnalyzer


class DummyTrendAnalyzer:
    def detect_seasonality(self):
        return {}

    def calculate_growth_rates(self):
        return {}


class DummyProductAnalyzer:
    def get_product_rankings(self):
        return pd.DataFrame()

    def get_category_analysis(self):
        return pd.DataFrame()

    def analyze_price_sensitivity(self):
        return {}


class TestDataLoader(unittest.TestCase):
    def _write_temp_csv(self, df: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()
        df.to_csv(tmp.name, index=False)
        return tmp.name

    def test_missing_columns_raises(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "product_id": [1],
            "product_name": ["A"],
        })
        path = self._write_temp_csv(df)
        try:
            loader = DataLoader(path)
            with self.assertRaises(ValueError):
                loader.load_data()
        finally:
            os.unlink(path)

    def test_invalid_numeric_raises(self):
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "product_id": [1],
            "product_name": ["A"],
            "category": ["C"],
            "quantity_sold": ["x"],
            "unit_price": [10.0],
            "revenue": [10.0],
        })
        path = self._write_temp_csv(df)
        try:
            loader = DataLoader(path)
            with self.assertRaises(ValueError):
                loader.load_data()
        finally:
            os.unlink(path)


class TestProductAnalyzer(unittest.TestCase):
    def test_empty_dataframes_return_empty(self):
        analyzer = ProductAnalyzer(pd.DataFrame())
        self.assertTrue(analyzer.get_product_rankings().empty)
        self.assertTrue(analyzer.get_category_analysis().empty)
        self.assertEqual(analyzer.analyze_price_sensitivity(), {})

    def test_missing_columns_return_empty(self):
        df = pd.DataFrame({"product_id": [1], "quantity_sold": [2]})
        analyzer = ProductAnalyzer(df)
        self.assertTrue(analyzer.get_product_rankings().empty)
        self.assertTrue(analyzer.get_category_analysis().empty)
        self.assertEqual(analyzer.analyze_price_sensitivity(), {})


class TestTrendAnalyzer(unittest.TestCase):
    def test_empty_dataframes_return_empty(self):
        analyzer = TrendAnalyzer(pd.DataFrame())
        self.assertTrue(analyzer.get_daily_trends().empty)
        self.assertTrue(analyzer.get_weekly_trends().empty)
        self.assertTrue(analyzer.get_monthly_trends().empty)

        seasonality = analyzer.detect_seasonality()
        self.assertIsNone(seasonality["best_day"])
        self.assertIsNone(seasonality["worst_day"])
        self.assertEqual(seasonality["weekend_boost"], 0.0)

        self.assertTrue(analyzer.simple_forecast().empty)


class TestRecommendationsEngine(unittest.TestCase):
    def test_empty_inputs_are_safe(self):
        engine = RecommendationsEngine(
            pd.DataFrame(),
            DummyTrendAnalyzer(),
            DummyProductAnalyzer(),
        )
        all_recs = engine.generate_all_recommendations()

        self.assertEqual(all_recs["inventory"], [])
        self.assertEqual(all_recs["pricing"], [])
        self.assertEqual(all_recs["marketing"], [])
        self.assertEqual(all_recs["operational"], [])
        self.assertEqual(all_recs["forecasting"], [])

        summary = all_recs["summary"]
        self.assertEqual(summary.get("business_health"), "Unknown")
        self.assertEqual(summary.get("key_metrics"), {})
        self.assertEqual(summary.get("top_performer"), {})


if __name__ == "__main__":
    unittest.main()
