"""
Trend Analysis Module
Analyzes sales trends over time, identifies patterns, and forecasts demand.
"""

from datetime import timedelta
from typing import Dict, List

import pandas as pd


class TrendAnalyzer:
    """
    Performs time-series analysis on retail sales data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.daily_sales = None
        self.weekly_sales = None
        self.monthly_sales = None

        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

    def _has_columns(self, columns: List[str]) -> bool:
        return all(col in self.data.columns for col in columns)

    def analyze_all_trends(self) -> Dict:
        """
        Run complete trend analysis pipeline.

        Returns:
            Dictionary containing all trend insights
        """
        return {
            'daily_trends': self.get_daily_trends(),
            'weekly_trends': self.get_weekly_trends(),
            'monthly_trends': self.get_monthly_trends(),
            'growth_analysis': self.calculate_growth_rates(),
            'seasonality': self.detect_seasonality(),
            'top_days': self.get_best_worst_days()
        }

    def get_daily_trends(self) -> pd.DataFrame:
        """Aggregate sales by day."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=['date', 'total_revenue', 'total_quantity', 'unique_products'])
        if not self._has_columns(['date', 'revenue', 'quantity_sold', 'product_id']):
            return pd.DataFrame(columns=['date', 'total_revenue', 'total_quantity', 'unique_products'])

        daily = self.data.groupby('date').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum',
            'product_id': 'nunique'
        }).reset_index()
        daily.columns = ['date', 'total_revenue', 'total_quantity', 'unique_products']
        self.daily_sales = daily
        return daily

    def get_weekly_trends(self) -> pd.DataFrame:
        """Aggregate sales by week."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=['week', 'total_revenue', 'total_quantity'])
        if not self._has_columns(['date', 'revenue', 'quantity_sold']):
            return pd.DataFrame(columns=['week', 'total_revenue', 'total_quantity'])

        self.data['year_week'] = self.data['date'].dt.strftime('%Y-W%U')
        weekly = self.data.groupby('year_week').agg({
            'revenue': 'sum',
            'quantity_sold': 'sum'
        }).reset_index()
        weekly.columns = ['week', 'total_revenue', 'total_quantity']
        self.weekly_sales = weekly
        return weekly

    def get_monthly_trends(self) -> pd.DataFrame:
        """Aggregate sales by month."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(columns=['year', 'month', 'total_revenue', 'total_quantity', 'month_year'])
        if not self._has_columns(['year', 'month', 'revenue', 'quantity_sold']):
            return pd.DataFrame(columns=['year', 'month', 'total_revenue', 'total_quantity', 'month_year'])

        monthly = self.data.groupby(['year', 'month']).agg({
            'revenue': 'sum',
            'quantity_sold': 'sum'
        }).reset_index()
        monthly['month_year'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
        monthly.columns = ['year', 'month', 'total_revenue', 'total_quantity', 'month_year']
        self.monthly_sales = monthly
        return monthly

    def calculate_growth_rates(self) -> Dict:
        """Calculate month-over-month growth rates."""
        monthly = self.get_monthly_trends()

        if len(monthly) < 2:
            return {'error': 'Insufficient data for growth calculation'}

        monthly['revenue_growth'] = monthly['total_revenue'].pct_change() * 100
        monthly['quantity_growth'] = monthly['total_quantity'].pct_change() * 100

        avg_revenue_growth = monthly['revenue_growth'].mean()
        avg_quantity_growth = monthly['quantity_growth'].mean()

        return {
            'monthly_data': monthly,
            'avg_revenue_growth': round(avg_revenue_growth, 2),
            'avg_quantity_growth': round(avg_quantity_growth, 2),
            'trend_direction': 'upward' if avg_revenue_growth > 0 else 'downward'
        }

    def detect_seasonality(self) -> Dict:
        """Detect weekly seasonality patterns."""
        if self.data is None or self.data.empty:
            return {
                'daily_averages': pd.DataFrame(columns=['day_of_week', 'revenue', 'quantity_sold']),
                'best_day': None,
                'worst_day': None,
                'weekend_boost': 0.0
            }
        if not self._has_columns(['day_of_week', 'revenue', 'quantity_sold']):
            return {
                'daily_averages': pd.DataFrame(columns=['day_of_week', 'revenue', 'quantity_sold']),
                'best_day': None,
                'worst_day': None,
                'weekend_boost': 0.0
            }

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = self.data.groupby('day_of_week').agg({
            'revenue': 'mean',
            'quantity_sold': 'mean'
        }).reindex(day_order)

        best_day = daily_avg['revenue'].idxmax()
        worst_day = daily_avg['revenue'].idxmin()

        return {
            'daily_averages': daily_avg.reset_index(),
            'best_day': best_day,
            'worst_day': worst_day,
            'weekend_boost': self._calculate_weekend_boost()
        }

    def _calculate_weekend_boost(self) -> float:
        """Calculate average weekend vs weekday performance."""
        if self.data is None or self.data.empty or 'day_of_week' not in self.data.columns:
            return 0.0

        self.data['is_weekend'] = self.data['day_of_week'].isin(['Saturday', 'Sunday'])
        weekend_avg = self.data[self.data['is_weekend']]['revenue'].mean()
        weekday_avg = self.data[~self.data['is_weekend']]['revenue'].mean()

        if weekday_avg > 0:
            boost = ((weekend_avg - weekday_avg) / weekday_avg) * 100
            return round(boost, 2)
        return 0.0

    def get_best_worst_days(self) -> Dict:
        """Identify best and worst performing days."""
        daily = self.get_daily_trends()
        if daily is None or daily.empty:
            return {
                'best_day': {'date': None, 'revenue': 0, 'quantity': 0},
                'worst_day': {'date': None, 'revenue': 0, 'quantity': 0}
            }

        best_day = daily.loc[daily['total_revenue'].idxmax()]
        worst_day = daily.loc[daily['total_revenue'].idxmin()]

        return {
            'best_day': {
                'date': best_day['date'].strftime('%Y-%m-%d') if hasattr(best_day['date'], 'strftime') else str(best_day['date']),
                'revenue': round(best_day['total_revenue'], 2),
                'quantity': int(best_day['total_quantity'])
            },
            'worst_day': {
                'date': worst_day['date'].strftime('%Y-%m-%d') if hasattr(worst_day['date'], 'strftime') else str(worst_day['date']),
                'revenue': round(worst_day['total_revenue'], 2),
                'quantity': int(worst_day['total_quantity'])
            }
        }

    def simple_forecast(self, days_ahead: int = 30) -> pd.DataFrame:
        """
        Simple moving average forecast.

        Args:
            days_ahead: Number of days to forecast

        Returns:
            DataFrame with forecasted values
        """
        daily = self.get_daily_trends()
        if daily is None or daily.empty:
            return pd.DataFrame(columns=['date', 'forecasted_revenue', 'type'])

        daily['ma_7'] = daily['total_revenue'].rolling(window=7).mean()
        ma_values = daily['ma_7'].dropna()
        if ma_values.empty:
            return pd.DataFrame(columns=['date', 'forecasted_revenue', 'type'])

        last_avg = ma_values.iloc[-1]
        last_date = daily['date'].max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]

        forecast = pd.DataFrame({
            'date': future_dates,
            'forecasted_revenue': [last_avg] * days_ahead,
            'type': 'forecast'
        })

        return forecast