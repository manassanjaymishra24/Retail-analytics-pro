"""
Recommendations Engine Module
Generates actionable business insights and recommendations based on analytics.
"""

from typing import Dict, List

import pandas as pd


class RecommendationsEngine:
    """
    Generates strategic recommendations based on sales analytics.
    """

    def __init__(self, data: pd.DataFrame, trend_analyzer, product_analyzer):
        self.data = data.copy()
        self.trend_analyzer = trend_analyzer
        self.product_analyzer = product_analyzer
        self.recommendations = []

    def _has_columns(self, columns: List[str]) -> bool:
        return all(col in self.data.columns for col in columns)

    def _safe_first_row(self, df: pd.DataFrame) -> Dict:
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    def _safe_series_idxmax(self, series: pd.Series):
        if series is None or series.empty:
            return None
        return series.idxmax()

    def generate_all_recommendations(self) -> Dict:
        """
        Generate comprehensive business recommendations.

        Returns:
            Dictionary with categorized recommendations
        """
        return {
            'inventory': self.generate_inventory_recommendations(),
            'pricing': self.generate_pricing_recommendations(),
            'marketing': self.generate_marketing_recommendations(),
            'operational': self.generate_operational_recommendations(),
            'forecasting': self.generate_forecast_recommendations(),
            'summary': self.generate_executive_summary()
        }

    def generate_inventory_recommendations(self) -> List[Dict]:
        """Generate inventory management recommendations."""
        recommendations = []

        rankings = self.product_analyzer.get_product_rankings()
        if rankings is None or rankings.empty:
            return recommendations

        # High-velocity items - ensure stock
        top_products = rankings.head(3)
        for _, product in top_products.iterrows():
            recommendations.append({
                'priority': 'High',
                'category': 'Stock Management',
                'action': f"Maintain safety stock for {product['product_name']}",
                'rationale': f"Top 20% performer generating ${product['revenue']:,.2f} revenue",
                'impact': 'Prevent stockouts on high-revenue items',
                'best_practice': 'Apply continuous review system for fast-movers'
            })

        # Slow-moving inventory
        bottom_products = rankings.tail(3)
        for _, product in bottom_products.iterrows():
            recommendations.append({
                'priority': 'Medium',
                'category': 'Inventory Optimization',
                'action': f"Review inventory levels for {product['product_name']}",
                'rationale': f"Low velocity: only ${product['revenue']:,.2f} revenue",
                'impact': 'Reduce carrying costs and free up capital',
                'best_practice': 'Consider promotional pricing or bundle offers'
            })

        # Category balancing
        category_perf = self.product_analyzer.get_category_analysis()
        if category_perf is not None and not category_perf.empty:
            top_category = category_perf.iloc[0]
            recommendations.append({
                'priority': 'High',
                'category': 'Assortment Planning',
                'action': f"Expand {top_category['category']} category selection",
                'rationale': f"Category generates {top_category['revenue_share']:.1f}% of total revenue",
                'impact': 'Capitalize on proven demand patterns',
                'best_practice': 'Use demand forecasting to guide assortment decisions'
            })

        return recommendations

    def generate_pricing_recommendations(self) -> List[Dict]:
        """Generate pricing strategy recommendations."""
        recommendations = []

        price_analysis = self.product_analyzer.analyze_price_sensitivity() or {}
        correlation = price_analysis.get('price_quantity_correlation')

        if correlation is not None and correlation < -0.3:
            recommendations.append({
                'priority': 'High',
                'category': 'Pricing Strategy',
                'action': 'Implement dynamic pricing for price-sensitive products',
                'rationale': f"Strong price sensitivity detected (correlation: {correlation:.2f})",
                'impact': 'Optimize revenue through demand-based pricing',
                'best_practice': 'Adjust prices during peak demand periods'
            })

        seasonality = self.trend_analyzer.detect_seasonality() or {}
        weekend_boost = seasonality.get('weekend_boost')
        if weekend_boost is not None and weekend_boost > 10:
            recommendations.append({
                'priority': 'Medium',
                'category': 'Pricing Strategy',
                'action': 'Test weekend premium pricing',
                'rationale': f"{weekend_boost:.1f}% higher sales on weekends",
                'impact': 'Capture increased willingness to pay during peak periods',
                'best_practice': 'Leverage demand forecasting for promotional timing'
            })

        return recommendations

    def generate_marketing_recommendations(self) -> List[Dict]:
        """Generate marketing and promotion recommendations."""
        recommendations = []

        seasonality = self.trend_analyzer.detect_seasonality() or {}
        worst_day = seasonality.get('worst_day')
        if worst_day is None:
            return recommendations

        recommendations.append({
            'priority': 'High',
            'category': 'Promotional Strategy',
            'action': f"Schedule promotions on {worst_day}s",
            'rationale': f"{worst_day} shows lowest natural demand",
            'impact': 'Smooth demand curve and improve weekly revenue consistency',
            'best_practice': 'Use demand forecasting to optimize promotional timing'
        })

        growth = self.trend_analyzer.calculate_growth_rates() or {}
        if growth.get('avg_revenue_growth', 0) > 5:
            recommendations.append({
                'priority': 'High',
                'category': 'Marketing Investment',
                'action': 'Increase marketing spend to capitalize on growth trend',
                'rationale': f"Revenue growing at {growth['avg_revenue_growth']:.1f}% monthly",
                'impact': 'Accelerate market share capture during growth phase',
                'best_practice': 'Align marketing campaigns with demand forecasts'
            })

        return recommendations

    def generate_operational_recommendations(self) -> List[Dict]:
        """Generate operational efficiency recommendations."""
        recommendations = []

        if not self._has_columns(['day_of_week', 'revenue']):
            return recommendations

        daily_avg = self.data.groupby('day_of_week')['revenue'].mean()
        peak_day = self._safe_series_idxmax(daily_avg)
        if peak_day is None:
            return recommendations

        recommendations.append({
            'priority': 'High',
            'category': 'Workforce Planning',
            'action': f"Increase staffing on {peak_day}s",
            'rationale': f"{peak_day} generates highest average revenue (${daily_avg[peak_day]:,.2f})",
            'impact': 'Improve customer service during peak periods',
            'best_practice': 'Integrate demand forecasting with workforce management'
        })

        recommendations.append({
            'priority': 'Medium',
            'category': 'Supply Chain',
            'action': 'Share demand forecasts with key suppliers',
            'rationale': 'Improve lead time accuracy and reduce stockouts',
            'impact': 'Strengthen supplier relationships and ensure availability',
            'best_practice': 'Foster cross-functional collaboration'
        })

        return recommendations

    def generate_forecast_recommendations(self) -> List[Dict]:
        """Generate demand forecasting improvement recommendations."""
        recommendations = []

        recommendations.append({
            'priority': 'High',
            'category': 'Forecasting Process',
            'action': 'Implement automated data validation pipeline',
            'rationale': 'Clean data is foundation of accurate forecasting',
            'impact': 'Reduce forecast errors by 20-50%',
            'best_practice': 'Start with clean sales data - remove outliers'
        })

        recommendations.append({
            'priority': 'High',
            'category': 'Forecasting Models',
            'action': 'Segment products by demand pattern and apply specific models',
            'rationale': 'Different products require different forecasting approaches',
            'impact': 'Improve accuracy for seasonal and promotional items',
            'best_practice': 'Use time series for stable items, causal models for promotions'
        })

        recommendations.append({
            'priority': 'Medium',
            'category': 'Forecast Governance',
            'action': 'Establish monthly forecast accuracy reviews (MAPE tracking)',
            'rationale': 'Continuous improvement requires measurement',
            'impact': 'Identify systematic biases and refine models',
            'best_practice': 'Measure forecast accuracy and act on deviations'
        })

        return recommendations

    def generate_executive_summary(self) -> Dict:
        """Generate high-level executive summary."""
        if not self._has_columns(['revenue', 'quantity_sold', 'date']):
            return {
                'business_health': 'Unknown',
                'key_metrics': {},
                'top_performer': {},
                'critical_actions': []
            }

        stats = {
            'total_revenue': self.data['revenue'].sum(),
            'total_units': self.data['quantity_sold'].sum(),
            'avg_daily_revenue': self.data.groupby('date')['revenue'].sum().mean(),
            'growth_rate': (self.trend_analyzer.calculate_growth_rates() or {}).get('avg_revenue_growth', 0)
        }

        top_product = self._safe_first_row(self.product_analyzer.get_product_rankings())

        return {
            'business_health': 'Growing' if stats['growth_rate'] > 0 else 'Stable',
            'key_metrics': {
                'total_revenue': f"${stats['total_revenue']:,.2f}",
                'total_units_sold': f"{stats['total_units']:,}",
                'avg_daily_revenue': f"${stats['avg_daily_revenue']:,.2f}",
                'monthly_growth': f"{stats['growth_rate']:.1f}%"
            },
            'top_performer': {
                'product': top_product.get('product_name', 'Unknown'),
                'revenue': f"${top_product.get('revenue', 0):,.2f}"
            },
            'critical_actions': [
                'Monitor inventory levels for top 3 products',
                'Implement weekend pricing strategy',
                'Review forecast accuracy monthly'
            ]
        }