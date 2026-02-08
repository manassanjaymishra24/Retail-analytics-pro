"""
Product Performance & Inventory Analysis Module
Analyzes individual product performance and provides inventory insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class ProductAnalyzer:
    """
    Analyzes product-level performance metrics and inventory status.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.product_summary = None

    def _has_columns(self, columns: List[str]) -> bool:
        return all(col in self.data.columns for col in columns)
        
    def analyze_all_products(self) -> Dict:
        """
        Run complete product analysis pipeline.
        
        Returns:
            Dictionary containing all product insights
        """
        return {
            'product_rankings': self.get_product_rankings(),
            'category_performance': self.get_category_analysis(),
            'top_performers': self.get_top_performers(),
            'under_performers': self.get_under_performers(),
            'price_analysis': self.analyze_price_sensitivity(),
            'product_trends': self.get_product_trends()
        }
    
    def get_product_rankings(self) -> pd.DataFrame:
        """
        Rank all products by key metrics.
        
        Returns:
            DataFrame with product performance rankings
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        if not self._has_columns(['product_id', 'product_name', 'category', 'quantity_sold', 'revenue', 'unit_price']):
            return pd.DataFrame()

        summary = self.data.groupby(['product_id', 'product_name', 'category']).agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
            'unit_price': 'first'
        }).reset_index()
        
        # Calculate additional metrics
        summary['avg_order_value'] = np.divide(
            summary['revenue'],
            summary['quantity_sold'],
            out=np.zeros_like(summary['revenue'], dtype=float),
            where=summary['quantity_sold'] != 0
        )
        summary['revenue_rank'] = summary['revenue'].rank(ascending=False)
        summary['quantity_rank'] = summary['quantity_sold'].rank(ascending=False)
        
        # Calculate performance score (composite)
        summary['performance_score'] = (
            (summary['revenue_rank'].max() - summary['revenue_rank'] + 1) * 0.6 +
            (summary['quantity_rank'].max() - summary['quantity_rank'] + 1) * 0.4
        )
        
        summary = summary.sort_values('performance_score', ascending=False)
        self.product_summary = summary
        
        return summary
    
    def get_category_analysis(self) -> pd.DataFrame:
        """Analyze performance by category."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        if not self._has_columns(['category', 'revenue', 'quantity_sold', 'product_id']):
            return pd.DataFrame()
        category_stats = self.data.groupby('category').agg({
            'revenue': ['sum', 'mean'],
            'quantity_sold': ['sum', 'mean'],
            'product_id': 'nunique'
        }).reset_index()
        
        # Flatten column names
        category_stats.columns = ['category', 'total_revenue', 'avg_daily_revenue', 
                                 'total_quantity', 'avg_daily_quantity', 'product_count']
        
        # Calculate category share
        category_stats['revenue_share'] = (category_stats['total_revenue'] / 
                                          category_stats['total_revenue'].sum() * 100).round(2)
        
        return category_stats.sort_values('total_revenue', ascending=False)
    
    def get_top_performers(self, n: int = 5) -> Dict:
        """
        Identify top performing products.
        
        Args:
            n: Number of top products to return
            
        Returns:
            Dictionary with top products by revenue and quantity
        """
        rankings = self.get_product_rankings()
        if rankings is None or rankings.empty:
            return {'by_revenue': [], 'by_quantity': []}
        
        top_revenue = rankings.nlargest(n, 'revenue')[['product_name', 'revenue', 'quantity_sold']]
        top_quantity = rankings.nlargest(n, 'quantity_sold')[['product_name', 'quantity_sold', 'revenue']]
        
        return {
            'by_revenue': top_revenue.to_dict('records'),
            'by_quantity': top_quantity.to_dict('records')
        }
    
    def get_under_performers(self, threshold_percentile: float = 25) -> pd.DataFrame:
        """
        Identify under-performing products.
        
        Args:
            threshold_percentile: Percentile threshold for under-performance
            
        Returns:
            DataFrame of under-performing products
        """
        rankings = self.get_product_rankings()
        if rankings is None or rankings.empty:
            return pd.DataFrame()
        threshold = np.percentile(rankings['performance_score'], threshold_percentile)
        
        under_performers = rankings[rankings['performance_score'] <= threshold].copy()
        under_performers['improvement_potential'] = 'Consider promotion or review'
        
        return under_performers
    
    def analyze_price_sensitivity(self) -> Dict:
        """Analyze relationship between price and quantity sold."""
        if self.data is None or self.data.empty:
            return {}
        if not self._has_columns(['product_id', 'unit_price', 'quantity_sold', 'product_name']):
            return {}
        product_stats = self.data.groupby('product_id').agg({
            'unit_price': 'first',
            'quantity_sold': 'sum',
            'product_name': 'first'
        }).reset_index()
        
        # Calculate correlation
        correlation = product_stats['unit_price'].corr(product_stats['quantity_sold'])
        if pd.isna(correlation):
            correlation = None
        
        # Segment products by price
        if product_stats['unit_price'].nunique() < 3:
            product_stats['price_segment'] = 'All'
        else:
            try:
                product_stats['price_segment'] = pd.qcut(
                    product_stats['unit_price'],
                    q=3,
                    labels=['Budget', 'Mid-range', 'Premium'],
                    duplicates='drop'
                )
            except ValueError:
                product_stats['price_segment'] = 'All'
        
        segment_performance = product_stats.groupby('price_segment', observed=False).agg({
            'quantity_sold': 'mean',
            'unit_price': 'mean'
        }).reset_index()
        
        return {
            'price_quantity_correlation': round(correlation, 3) if correlation is not None else None,
            'segment_analysis': segment_performance.to_dict('records'),
            'insights': self._generate_price_insights(correlation) if correlation is not None else "Insufficient data for price insights"
        }
    
    def _generate_price_insights(self, correlation: float) -> str:
        """Generate insights based on price correlation."""
        if correlation is None:
            return "Insufficient data for price insights"
        if correlation < -0.5:
            return "Strong negative correlation: Higher prices significantly reduce sales volume"
        elif correlation < -0.2:
            return "Moderate negative correlation: Price sensitivity detected"
        elif correlation > 0.2:
            return "Positive correlation: Higher prices associated with higher sales (premium effect)"
        else:
            return "Weak correlation: Price is not the main driver of sales volume"
    
    def get_product_trends(self) -> pd.DataFrame:
        """Analyze individual product trends over time."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        if not self._has_columns(['product_id', 'product_name', 'year', 'month', 'revenue', 'quantity_sold']):
            return pd.DataFrame()
        # Calculate month-over-month growth for each product
        monthly_product = self.data.groupby(['product_id', 'product_name', 'year', 'month']).agg({
            'revenue': 'sum',
            'quantity_sold': 'sum'
        }).reset_index()
        
        # Calculate growth rate for each product
        def calc_growth(group):
            group = group.sort_values(['year', 'month'])
            group['revenue_growth'] = group['revenue'].pct_change() * 100
            return group
        
        monthly_product = monthly_product.groupby('product_id', group_keys=False).apply(calc_growth)
        
        # Get latest growth rates
        latest_growth = monthly_product.groupby('product_id').last().reset_index()
        latest_growth = latest_growth[['product_id', 'product_name', 'revenue', 'revenue_growth']]
        latest_growth['trend'] = latest_growth['revenue_growth'].apply(
            lambda x: 'Growing' if x > 5 else ('Declining' if x < -5 else 'Stable')
        )
        
        return latest_growth.sort_values('revenue_growth', ascending=False)
    
    def generate_inventory_recommendations(self) -> List[Dict]:
        """
        Generate inventory management recommendations.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        rankings = self.get_product_rankings()
        if rankings is None or rankings.empty:
            return recommendations
        
        # High performers - ensure stock
        top_products = rankings.head(3)
        for _, product in top_products.iterrows():
            recommendations.append({
                'product': product['product_name'],
                'priority': 'High',
                'action': 'Ensure adequate inventory',
                'reason': f"Top performer with ${product['revenue']:,.2f} revenue"
            })
        
        # Under-performers - review
        bottom_products = rankings.tail(3)
        for _, product in bottom_products.iterrows():
            recommendations.append({
                'product': product['product_name'],
                'priority': 'Medium',
                'action': 'Review pricing or promotion strategy',
                'reason': f"Low performance: ${product['revenue']:,.2f} revenue"
            })
        
        return recommendations
