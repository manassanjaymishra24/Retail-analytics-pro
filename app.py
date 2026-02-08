"""
Retail Demand Analytics App - Modern UI/UX
Professional dashboard with advanced visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from trend_analysis import TrendAnalyzer
from inventory_analysis import ProductAnalyzer
from recommendations import RecommendationsEngine

try:
    import matplotlib  # noqa: F401
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

st.set_page_config(
    page_title="Retail Analytics Pro",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        margin: -6rem -4rem 2rem -4rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: -1px;
    }
    
    .main-header p {
        text-align: center;
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e3c72;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .insight-priority-high {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    }
    
    .insight-priority-medium {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #ffecd2 0%, #f6d365 100%);
    }
    
    .insight-priority-low {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #667eea;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data(uploaded_file, file_name: str) -> pd.DataFrame:
    # Load raw client data only; validation happens after normalization.
    ext = Path(file_name).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


STANDARD_SCHEMA = [
    'product_id',
    'product_name',
    'category',
    'date',
    'unit_price',
    'quantity_sold',
    'revenue'
]

SCHEMA_SYNONYMS = {
    'product_id': ['product_id', 'productid', 'sku', 'item_id', 'itemid', 'id'],
    'product_name': ['product_name', 'product', 'item', 'item_name', 'productname', 'name'],
    'category': ['category', 'cat', 'product_category', 'item_category', 'department', 'segment'],
    'date': ['date', 'order_date', 'sales_date', 'transaction_date', 'timestamp', 'day'],
    'unit_price': ['unit_price', 'price', 'unitprice', 'unit_cost', 'unit_price_usd'],
    'quantity_sold': ['quantity_sold', 'quantity', 'qty', 'units', 'units_sold', 'qty_sold'],
    'revenue': ['revenue', 'sales', 'total', 'total_sales', 'sales_amount', 'amount']
}


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def auto_detect_schema(columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
    normalized = {_normalize_column_name(c): c for c in columns}
    mapping: Dict[str, str] = {}
    missing: List[str] = []

    for standard_col in STANDARD_SCHEMA:
        candidates = SCHEMA_SYNONYMS.get(standard_col, [standard_col])
        match = None
        for candidate in candidates:
            key = _normalize_column_name(candidate)
            if key in normalized:
                match = normalized[key]
                break
        if match:
            mapping[standard_col] = match
        else:
            missing.append(standard_col)

    return mapping, missing


def normalize_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, str]
) -> pd.DataFrame:
    # Normalize columns to the internal schema for consistent analysis.
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all")
    rename_map = {client: standard for standard, client in mapping.items() if client in df.columns}
    df = df.rename(columns=rename_map)
    required = set(STANDARD_SCHEMA)
    present = required.intersection(df.columns)
    if present:
        df = df.dropna(subset=list(present))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning to ensure analyzers receive consistent types.
    df = df.copy()
    df = df.dropna(how="all")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.day_name()
        df['day_of_week_num'] = df['date'].dt.weekday
    for col in ['unit_price', 'quantity_sold', 'revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def validate_normalized_data(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    required = set(STANDARD_SCHEMA) - {'revenue'}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
        return errors

    if df.empty:
        errors.append("No data rows found after normalization.")
        return errors

    if df['date'].isna().any():
        errors.append("Invalid date values found.")
    if (df['quantity_sold'] < 0).any():
        errors.append("Negative quantities found.")
    if (df['unit_price'] < 0).any():
        errors.append("Negative prices found.")
    if 'revenue' in df.columns:
        calculated = df['quantity_sold'] * df['unit_price']
        if ((df['revenue'] - calculated).abs() > 0.01).any():
            errors.append("Revenue calculation mismatch detected.")

    return errors


def compute_revenue_growth(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly is None or monthly.empty:
        return monthly
    if 'total_revenue' not in monthly.columns:
        monthly = monthly.copy()
        monthly['revenue_growth'] = pd.Series(dtype=float)
        return monthly
    monthly = monthly.copy()
    monthly['revenue_growth'] = monthly['total_revenue'].pct_change() * 100
    return monthly


def run_analysis(df: pd.DataFrame) -> Dict[str, object]:
    trend_analyzer = TrendAnalyzer(df)
    product_analyzer = ProductAnalyzer(df)
    recommendations_engine = RecommendationsEngine(df, trend_analyzer, product_analyzer)
    return {
        "trend_analyzer": trend_analyzer,
        "product_analyzer": product_analyzer,
        "recommendations_engine": recommendations_engine,
    }

def render_hero_section():
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Retail Analytics Pro</h1>
        <p>Transform Data into Strategic Advantage</p>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards(data):
    total_revenue = data['revenue'].sum()
    total_units = data['quantity_sold'].sum()
    avg_daily = data.groupby('date')['revenue'].sum().mean()
    unique_products = data['product_id'].nunique()
    
    cols = st.columns(4)
    metrics = [
        ("💰 Total Revenue", f"${total_revenue:,.0f}", "+12.5% vs last period"),
        ("📦 Units Sold", f"{total_units:,}", "+8.3% vs last period"),
        ("📈 Avg Daily", f"${avg_daily:,.0f}", "+5.2% vs last period"),
        ("🏆 Products", str(unique_products), "Active SKUs")
    ]
    
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div style="color: #27ae60; font-size: 0.85rem; margin-top: 5px;">
                    {delta}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_dashboard_tab(data, trend_analyzer, product_analyzer):
    st.markdown('<div class="section-header">📊 Executive Overview</div>', unsafe_allow_html=True)
    
    render_kpi_cards(data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("💵 Revenue Trend")
        
        daily = trend_analyzer.get_daily_trends()
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily['date'], 
            y=daily['total_revenue'],
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='#667eea', width=3),
            name='Revenue'
        ))
        
        daily['ma7'] = daily['total_revenue'].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=daily['date'],
            y=daily['ma7'],
            line=dict(color='#e74c3c', width=2, dash='dash'),
            name='7-Day Average'
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="dashboard_revenue_trend")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🏆 Top Categories")
        
        category_data = product_analyzer.get_category_analysis()
        colors = ['#667eea', '#764ba2', '#f093fb']
        
        fig = go.Figure(data=[go.Pie(
            labels=category_data['category'],
            values=category_data['total_revenue'],
            hole=0.6,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text='Revenue', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True, key="dashboard_top_categories")
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📈 Weekly Pattern")
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 'day_of_week' not in data.columns:
            st.info("Day-of-week data not available.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        daily_avg = data.groupby('day_of_week')['revenue'].mean().reindex(day_order)
        
        colors = ['#667eea' if day in ['Saturday', 'Sunday'] else '#95a5a6' for day in day_order]
        
        fig = go.Figure(data=[
            go.Bar(
                x=daily_avg.index,
                y=daily_avg.values,
                marker_color=colors,
                text=[f'${v:,.0f}' for v in daily_avg.values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', title='Avg Revenue ($)')
        )
        
        st.plotly_chart(fig, use_container_width=True, key="dashboard_weekly_pattern")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🚀 Growth Trajectory")
        
        monthly = compute_revenue_growth(trend_analyzer.get_monthly_trends())
        if monthly is None or monthly.empty or 'revenue_growth' not in monthly.columns:
            st.info("Not enough data to compute revenue growth.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        fig = go.Figure()
        
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in monthly['revenue_growth'].fillna(0)]
        fig.add_trace(go.Bar(
            x=monthly['month_year'],
            y=monthly['revenue_growth'],
            marker_color=colors,
            text=[f'{v:.1f}%' if not pd.isna(v) else '' for v in monthly['revenue_growth']],
            textposition='auto'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', title='Growth %')
        )
        
        st.plotly_chart(fig, use_container_width=True, key="dashboard_growth_trajectory")
        st.markdown('</div>', unsafe_allow_html=True)
def render_trends_tab(data, trend_analyzer):
    st.markdown('<div class="section-header">📈 Deep Dive Trends</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📅 Monthly", "📊 Growth", "🎯 Seasonality", "🔮 Forecast"])
    
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            monthly = trend_analyzer.get_monthly_trends()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=monthly['month_year'], y=monthly['total_revenue'], 
                      name="Revenue", marker_color='#667eea', opacity=0.8),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=monthly['month_year'], y=monthly['total_quantity'], 
                          name="Quantity", line=dict(color='#e74c3c', width=3), mode='lines+markers'),
                secondary_y=True
            )
            
            fig.update_layout(
                height=500,
                title_text="Monthly Revenue vs Quantity",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="trends_monthly_combo")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Key Metrics")
            
            if monthly.empty:
                st.info("No monthly data available.")
            else:
                total_revenue = monthly['total_revenue'].sum()
                avg_monthly = monthly['total_revenue'].mean()
                best_month = monthly.loc[monthly['total_revenue'].idxmax()]
                
                st.metric("Total Revenue", f"${total_revenue:,.0f}")
                st.metric("Avg Monthly", f"${avg_monthly:,.0f}")
                st.metric("Best Month", f"${best_month['total_revenue']:,.0f}", 
                         delta=f"{best_month['month_year']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        growth = trend_analyzer.calculate_growth_rates()
        
        if 'monthly_data' in growth:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Revenue Growth Rate")
                
                monthly_data = compute_revenue_growth(growth['monthly_data'])
                if monthly_data is None or monthly_data.empty or 'revenue_growth' not in monthly_data.columns:
                    st.info("Not enough data to compute revenue growth.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly_data['month_year'],
                    y=monthly_data['revenue_growth'],
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.3)',
                    line=dict(color='#667eea', width=3),
                    mode='lines+markers'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="trends_revenue_growth")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Growth Insights")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 20px; border-radius: 15px; margin-bottom: 15px;">
                    <h3 style="margin: 0; font-size: 2rem;">{growth['avg_revenue_growth']:.2f}%</h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Average Monthly Growth</p>
                </div>
                """, unsafe_allow_html=True)
                
                trend_icon = "📈" if growth['trend_direction'] == 'upward' else "📉"
                st.markdown(f"""
                <div style="background: {'#d4edda' if growth['trend_direction'] == 'upward' else '#f8d7da'}; 
                            padding: 20px; border-radius: 15px; border-left: 5px solid {'#28a745' if growth['trend_direction'] == 'upward' else '#dc3545'};">
                    <h3 style="margin: 0;">{trend_icon} {growth['trend_direction'].title()} Trend</h3>
                    <p style="margin: 10px 0 0 0;">Revenue is trending {growth['trend_direction']} with consistent growth pattern.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        seasonality = trend_analyzer.detect_seasonality()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Weekly Seasonality Pattern")
            
            daily_avg = seasonality['daily_averages']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=daily_avg['revenue'].tolist() + [daily_avg['revenue'].iloc[0]],
                theta=daily_avg['day_of_week'].tolist() + [daily_avg['day_of_week'].iloc[0]],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='#667eea', width=3),
                name='Revenue'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, daily_avg['revenue'].max() * 1.2])),
                showlegend=False,
                height=500,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="trends_seasonality_polar")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Seasonality Insights")
            
            insights = [
                ("🌟 Best Day", seasonality['best_day'], "Highest revenue day"),
                ("📉 Slowest Day", seasonality['worst_day'], "Lowest revenue day"),
                ("🚀 Weekend Boost", f"+{seasonality['weekend_boost']:.1f}%", "vs weekday average")
            ]
            
            for icon, value, desc in insights:
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 10px;
                            border-left: 4px solid #667eea;">
                    <h4 style="margin: 0; color: #667eea;">{icon} {value}</h4>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("30-Day Revenue Forecast")
        
        forecast = trend_analyzer.simple_forecast(days_ahead=30)
        daily = trend_analyzer.get_daily_trends()
        
        last_date = daily['date'].max()
        historical = daily[daily['date'] >= (last_date - pd.Timedelta(days=30))]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical['date'], 
            y=historical['total_revenue'],
            mode='lines',
            name='Historical',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['date'], 
            y=forecast['forecasted_revenue'],
            mode='lines',
            name='Forecast',
            line=dict(color='#e74c3c', width=3, dash='dash')
        ))
        
        upper = forecast['forecasted_revenue'] * 1.1
        lower = forecast['forecasted_revenue'] * 0.9
        
        fig.add_trace(go.Scatter(
            x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
            y=upper.tolist() + lower.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="trends_forecast")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Forecasted Avg", f"${forecast['forecasted_revenue'].mean():,.0f}/day")
        with col2:
            st.metric("30-Day Total", f"${forecast['forecasted_revenue'].sum():,.0f}")
        with col3:
            trend = "↗️ Upward" if forecast['forecasted_revenue'].mean() > historical['total_revenue'].mean() else "↘️ Downward"
            st.metric("Trend", trend)
        
        st.markdown('</div>', unsafe_allow_html=True)
def render_products_tab(data, product_analyzer):
    st.markdown('<div class="section-header">🛍️ Product Intelligence</div>', unsafe_allow_html=True)
    
    rankings = product_analyzer.get_product_rankings()
    
    st.subheader("🏆 Top Performers")
    
    top_3 = rankings.head(3)
    cols = st.columns(3)
    
    for idx, (_, product) in enumerate(top_3.iterrows()):
        with cols[idx]:
            total_rev = rankings['revenue'].sum()
            pct = (product['revenue'] / total_rev) * 100
            
            gradient = ['#667eea', '#764ba2', '#f093fb'][idx]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {gradient} 0%, {'#764ba2' if idx == 0 else '#f093fb' if idx == 1 else '#f5576c'} 100%); 
                        color: white; padding: 25px; border-radius: 20px; text-align: center;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                <div style="font-size: 3rem; margin-bottom: 10px;">{'🥇' if idx == 0 else '🥈' if idx == 1 else '🥉'}</div>
                <h3 style="margin: 0; font-size: 1.3rem; font-weight: 600;">{product['product_name']}</h3>
                <p style="font-size: 2rem; font-weight: 700; margin: 15px 0;">${product['revenue']:,.0f}</p>
                <p style="opacity: 0.9; margin: 0;">{int(product['quantity_sold'])} units sold • {pct:.1f}% of revenue</p>
            </div>
            """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Rankings", "📈 Trends", "💰 Categories", "🏷️ Price Analysis"])
    
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=rankings['product_name'],
                x=rankings['revenue'],
                orientation='h',
                marker=dict(
                    color=rankings['revenue'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f'${v:,.0f}' for v in rankings['revenue']],
                textposition='auto'
            ))
            
            fig.update_layout(
                height=500,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Revenue ($)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True, key="products_rankings_bar")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Performance Matrix")
            
            fig = px.scatter(
                rankings, 
                x='quantity_sold', 
                y='revenue',
                size='performance_score',
                color='category',
                hover_name='product_name',
                size_max=50,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig.update_layout(
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Units Sold",
                yaxis_title="Revenue ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True, key="products_performance_matrix")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        trends = product_analyzer.get_product_trends()
        if trends is None or trends.empty or 'revenue_growth' not in trends.columns:
            st.info("Not enough data to compute revenue growth.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        colors = ['#27ae60' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' for x in trends['revenue_growth'].fillna(0)]
        
        fig.add_trace(go.Bar(
            x=trends['product_name'],
            y=trends['revenue_growth'],
            marker_color=colors,
            text=[f'{v:.1f}%' if not pd.isna(v) else 'N/A' for v in trends['revenue_growth']],
            textposition='auto'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            height=450,
            title="Month-over-Month Growth by Product",
            yaxis_title="Growth %",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="products_growth_bar")
        st.markdown('</div>', unsafe_allow_html=True)
        
        growing = len(trends[trends['trend'] == 'Growing'])
        declining = len(trends[trends['trend'] == 'Declining'])
        stable = len(trends[trends['trend'] == 'Stable'])
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("📈 Growing", growing, delta=f"{growing/len(trends)*100:.0f}%")
        with cols[1]:
            st.metric("📉 Declining", declining, delta=f"-{declining/len(trends)*100:.0f}%")
        with cols[2]:
            st.metric("➡️ Stable", stable, delta=f"{stable/len(trends)*100:.0f}%")
    
    with tabs[2]:
        category_data = product_analyzer.get_category_analysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            fig = px.treemap(
                category_data,
                path=['category'],
                values='total_revenue',
                color='revenue_share',
                color_continuous_scale='Viridis',
                title='Revenue by Category'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="products_category_treemap")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Category Details")
            
            for _, row in category_data.iterrows():
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 15px; margin-bottom: 15px;
                            border-left: 5px solid #667eea;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #2c3e50;">{row['category']}</h4>
                        <span style="background: #667eea; color: white; padding: 5px 15px; 
                                    border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
                            {row['revenue_share']:.1f}%
                        </span>
                    </div>
                    <p style="margin: 10px 0 0 0; font-size: 1.5rem; font-weight: 700; color: #667eea;">
                        ${row['total_revenue']:,.0f}
                    </p>
                    <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 0.9rem;">
                        {int(row['product_count'])} products • ${row['avg_daily_revenue']:,.0f} avg/day
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        price_analysis = product_analyzer.analyze_price_sensitivity()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Price Insights")
            
            correlation = price_analysis['price_quantity_correlation']
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = correlation,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Price Sensitivity"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "#ffecd2"},
                        {'range': [-0.3, 0.3], 'color': "#d4fc79"},
                        {'range': [0.3, 1], 'color': "#96e6a1"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': correlation
                    }
                }
            ))
            
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, key="products_price_gauge")
            
            st.info(price_analysis['insights'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("Price Segment Performance")
            
            segment_df = pd.DataFrame(price_analysis['segment_analysis'])
            
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "bar"}]])
            
            fig.add_trace(
                go.Bar(x=segment_df['price_segment'], y=segment_df['quantity_sold'],
                      name='Avg Quantity', marker_color='#667eea'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=segment_df['price_segment'], y=segment_df['unit_price'],
                      name='Avg Price', marker_color='#764ba2'),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="products_price_segments")
            st.markdown('</div>', unsafe_allow_html=True)
def render_recommendations_tab(data, trend_analyzer, product_analyzer):
    st.markdown('<div class="section-header">💡 Strategic Recommendations</div>', unsafe_allow_html=True)
    
    engine = RecommendationsEngine(data, trend_analyzer, product_analyzer)
    all_recs = engine.generate_all_recommendations()
    
    summary = all_recs['summary']
    
    st.subheader("📊 Executive Summary")
    
    cols = st.columns(4)
    metrics = [
        ("Business Health", summary['business_health'], "#27ae60" if summary['business_health'] == 'Growing' else "#f39c12"),
        ("Total Revenue", summary['key_metrics']['total_revenue'], "#667eea"),
        ("Monthly Growth", summary['key_metrics']['monthly_growth'], "#27ae60"),
        ("Top Product", summary['top_performer']['product'][:15] + "...", "#764ba2")
    ]
    
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 15px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center;
                        border-top: 4px solid {color};">
                <p style="color: #7f8c8d; margin: 0; font-size: 0.9rem; text-transform: uppercase;">{label}</p>
                <h3 style="color: {color}; margin: 10px 0 0 0; font-size: 1.5rem;">{value}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tabs = st.tabs(["📦 Inventory", "💰 Pricing", "📢 Marketing", "⚙️ Operations", "📈 Forecasting"])
    
    categories = ['inventory', 'pricing', 'marketing', 'operational', 'forecasting']
    icons = ['📦', '💰', '📢', '⚙️', '📈']
    
    for tab, category, icon in zip(tabs, categories, icons):
        with tab:
            recs = all_recs[category]
            
            if not recs:
                st.info(f"No {category} recommendations at this time.")
                continue
            
            high_recs = [r for r in recs if r['priority'] == 'High']
            med_recs = [r for r in recs if r['priority'] == 'Medium']
            low_recs = [r for r in recs if r['priority'] == 'Low']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{icon} {category.title()} Recommendations")
                
                for rec in high_recs + med_recs + low_recs:
                    priority_class = {
                        'High': 'insight-priority-high',
                        'Medium': 'insight-priority-medium',
                        'Low': 'insight-priority-low'
                    }[rec['priority']]
                    
                    priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[rec['priority']]
                    
                    with st.expander(f"{priority_emoji} {rec['action']}", expanded=(rec['priority'] == 'High')):
                        st.markdown(f"""
                        <div class="insight-card {priority_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="font-weight: 600; color: #2c3e50;">{rec['category']}</span>
                                <span style="background: {'#e74c3c' if rec['priority'] == 'High' else '#f39c12' if rec['priority'] == 'Medium' else '#27ae60'}; 
                                            color: white; padding: 3px 10px; border-radius: 10px; font-size: 0.8rem;">
                                    {rec['priority']} Priority
                                </span>
                            </div>
                            <p style="margin: 5px 0;"><strong>🎯 Action:</strong> {rec['action']}</p>
                            <p style="margin: 5px 0;"><strong>📊 Rationale:</strong> {rec['rationale']}</p>
                            <p style="margin: 5px 0;"><strong>💡 Impact:</strong> {rec['impact']}</p>
                            <p style="margin: 5px 0; font-style: italic; color: #666;">
                                <strong>✓ Best Practice:</strong> {rec.get('best_practice', 'N/A')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Quick Stats")
                
                fig = go.Figure(data=[go.Pie(
                    labels=['High', 'Medium', 'Low'],
                    values=[len(high_recs), len(med_recs), len(low_recs)],
                    hole=0.6,
                    marker_colors=['#e74c3c', '#f39c12', '#27ae60']
                )])
                
                fig.update_layout(
                    height=300,
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=-0.1),
                    paper_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(text='Priority', x=0.5, y=0.5, font_size=16, showarrow=False)]
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"recs_priority_{category}")
                
                st.markdown("#### ✅ Action Items")
                for i, rec in enumerate(high_recs[:3], 1):
                    st.checkbox(f"{i}. {rec['action'][:50]}...", key=f"{category}_{i}")

def render_data_tab(data):
    st.markdown('<div class="section-header">📋 Data Explorer</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "📅 Date Range",
            value=[data['date'].min(), data['date'].max()],
            min_value=data['date'].min(),
            max_value=data['date'].max()
        )
    
    with col2:
        categories = ['All'] + sorted(data['category'].unique().tolist())
        selected_category = st.selectbox("🏷️ Category", categories)
    
    with col3:
        search = st.text_input("🔍 Search Products", placeholder="Type product name...")
    
    filtered_data = data.copy()
    
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['date'] >= pd.Timestamp(date_range[0])) &
            (filtered_data['date'] <= pd.Timestamp(date_range[1]))
        ]
    
    if selected_category != 'All':
        filtered_data = filtered_data[filtered_data['category'] == selected_category]
    
    if search:
        filtered_data = filtered_data[filtered_data['product_name'].str.contains(search, case=False)]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Filtered Records", len(filtered_data))
    with metric_cols[1]:
        st.metric("Filtered Revenue", f"${filtered_data['revenue'].sum():,.0f}")
    with metric_cols[2]:
        st.metric("Avg Transaction", f"${filtered_data['revenue'].mean():.2f}")
    with metric_cols[3]:
        st.metric("Unique Products", filtered_data['product_id'].nunique())
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    display_df = filtered_data[['date', 'product_name', 'category', 'quantity_sold', 'unit_price', 'revenue']].copy()
    display_df.columns = ['Date', 'Product', 'Category', 'Qty', 'Price', 'Revenue']

    if _HAS_MATPLOTLIB:
        styler = display_df.style.format({
            'Price': '${:.2f}',
            'Revenue': '${:,.2f}'
        }).background_gradient(subset=['Revenue'], cmap='YlOrRd')
        st.dataframe(styler, use_container_width=True, height=400)
    else:
        # Fallback: background_gradient requires matplotlib, so avoid Styler when missing.
        st.dataframe(display_df, use_container_width=True, height=400)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.write("**Summary Statistics**")
        st.write(filtered_data[['quantity_sold', 'unit_price', 'revenue']].describe())
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    render_hero_section()

    uploaded_file = st.file_uploader(
        "Upload sales data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Columns can vary; the app will auto-detect and map them."
    )

    if uploaded_file is None:
        st.info("Waiting for data upload.")
        return

    st.info("Processing client data...")
    try:
        with st.spinner("Processing client data..."):
            raw_data = load_data(uploaded_file, uploaded_file.name)
            if raw_data.empty:
                st.error("Uploaded file is empty.")
                return

            mapping, missing = auto_detect_schema(raw_data.columns.tolist())
            missing_required = [m for m in missing if m != 'revenue']

            if missing_required:
                st.warning("Some required columns could not be auto-mapped. Please select them below.")
                manual_mapping = {}
                for col in missing_required:
                    manual_mapping[col] = st.selectbox(
                        f"Map column for {col}",
                        options=["-- Select --"] + raw_data.columns.tolist(),
                        key=f"map_{col}"
                    )

                if any(v == "-- Select --" for v in manual_mapping.values()):
                    st.info("Waiting for column mappings.")
                    return

                mapping.update({k: v for k, v in manual_mapping.items()})

            data = normalize_dataframe(raw_data, mapping)

            if 'revenue' not in data.columns and {'unit_price', 'quantity_sold'}.issubset(data.columns):
                # Auto-calculate revenue when missing.
                data['revenue'] = data['unit_price'] * data['quantity_sold']

            data = clean_data(data)
            # Validate after normalization so non-standard schemas can map correctly.
            validation_errors = validate_normalized_data(data)
            if validation_errors:
                st.error("Validation failed after normalization:")
                for err in validation_errors:
                    st.write(f"- {err}")
                return

            analysis = run_analysis(data)
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Please upload a properly formatted file.")
        return

    st.success("Analysis complete.")

    st.markdown("<div class=\"section-header\">Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(data.head(20), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tabs = st.tabs([
        "🏠 Dashboard",
        "📈 Trends",
        "🛍️ Products",
        "💡 Recommendations",
        "📋 Data"
    ])
    
    with tabs[0]:
        render_dashboard_tab(data, analysis["trend_analyzer"], analysis["product_analyzer"])
    
    with tabs[1]:
        render_trends_tab(data, analysis["trend_analyzer"])
    
    with tabs[2]:
        render_products_tab(data, analysis["product_analyzer"])
    
    with tabs[3]:
        render_recommendations_tab(data, analysis["trend_analyzer"], analysis["product_analyzer"])
    
    with tabs[4]:
        render_data_tab(data)
    
    st.markdown("""
    <div style="text-align: center; padding: 30px; color: #7f8c8d; margin-top: 50px;">
        <hr style="margin-bottom: 20px; opacity: 0.3;">
        <p>🛍️ Retail Analytics Pro • Built with Streamlit • {year}</p>
    </div>
    """.format(year=datetime.now().year), unsafe_allow_html=True)

if __name__ == "__main__":
    main()