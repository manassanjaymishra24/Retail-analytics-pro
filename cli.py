#!/usr/bin/env python3
"""
Retail Demand Analytics - CLI Version
Command-line interface for quick analysis without Streamlit.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from trend_analysis import TrendAnalyzer
from inventory_analysis import ProductAnalyzer
from recommendations import RecommendationsEngine

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_metric(label, value):
    print(f"  {label:<25} {value}")

def main():
    parser = argparse.ArgumentParser(
        description='Retail Demand Analytics CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --file PATH_TO_DATA.csv --full
  python cli.py --summary
  python cli.py --recommendations
        """
    )
    
    parser.add_argument('--file', required=True,
                       help='Path to sales data CSV')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary statistics only')
    parser.add_argument('--trends', action='store_true',
                       help='Show trend analysis')
    parser.add_argument('--products', action='store_true',
                       help='Show product performance')
    parser.add_argument('--recommendations', action='store_true',
                       help='Show business recommendations')
    parser.add_argument('--full', action='store_true',
                       help='Show complete analysis (all sections)')
    
    args = parser.parse_args()
    
    # If no specific flags, show summary
    if not any([args.summary, args.trends, args.products, args.recommendations, args.full]):
        args.summary = True
    
    if args.full:
        args.summary = args.trends = args.products = args.recommendations = True
    
    try:
        # Load data
        print("\n📊 Loading data...")
        loader = DataLoader(args.file)
        data = loader.load_data()
        stats = loader.get_summary_stats()
        
        # Initialize analyzers
        trend_analyzer = TrendAnalyzer(data)
        product_analyzer = ProductAnalyzer(data)
        
        # Summary Section
        if args.summary:
            print_header("BUSINESS SUMMARY")
            print_metric("Total Revenue", f"${stats['total_revenue']:,.2f}")
            print_metric("Total Units Sold", f"{stats['total_units']:,}")
            print_metric("Date Range", f"{stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
            print_metric("Active Products", stats['total_products'])
            print_metric("Categories", stats['total_categories'])
            print_metric("Avg Daily Revenue", f"${stats['total_revenue']/len(data['date'].unique()):,.2f}")
        
        # Trends Section
        if args.trends:
            print_header("TREND ANALYSIS")
            
            growth = trend_analyzer.calculate_growth_rates()
            print_metric("Monthly Growth Rate", f"{growth['avg_revenue_growth']:.2f}%")
            print_metric("Trend Direction", growth['trend_direction'].upper())
            
            seasonality = trend_analyzer.detect_seasonality()
            print_metric("Best Sales Day", seasonality['best_day'])
            print_metric("Weekend Boost", f"{seasonality['weekend_boost']:.1f}%")
            
            best_worst = trend_analyzer.get_best_worst_days()
            print_metric("Best Day Revenue", f"${best_worst['best_day']['revenue']:,.2f} ({best_worst['best_day']['date']})")
            print_metric("Worst Day Revenue", f"${best_worst['worst_day']['revenue']:,.2f} ({best_worst['worst_day']['date']})")
        
        # Products Section
        if args.products:
            print_header("TOP PRODUCTS")
            rankings = product_analyzer.get_product_rankings()
            
            for i, row in rankings.head(5).iterrows():
                print(f"  {int(row['performance_score']):>2}. {row['product_name']:<20} ${row['revenue']:>10,.2f} ({int(row['quantity_sold'])} units)")
            
            print_header("CATEGORY PERFORMANCE")
            categories = product_analyzer.get_category_analysis()
            for _, row in categories.iterrows():
                print_metric(f"{row['category']}", f"${row['total_revenue']:,.2f} ({row['revenue_share']:.1f}%)")
        
        # Recommendations Section
        if args.recommendations:
            print_header("STRATEGIC RECOMMENDATIONS")
            engine = RecommendationsEngine(data, trend_analyzer, product_analyzer)
            recs = engine.generate_all_recommendations()
            
            # Group by priority
            high_priority = []
            medium_priority = []
            
            for category, items in recs.items():
                if category == 'summary':
                    continue
                for item in items:
                    if item['priority'] == 'High':
                        high_priority.append((category, item))
                    elif item['priority'] == 'Medium':
                        medium_priority.append((category, item))
            
            if high_priority:
                print("\n  🔴 HIGH PRIORITY:")
                for cat, item in high_priority[:5]:
                    print(f"     • [{cat.upper()}] {item['action']}")
                    print(f"       Reason: {item['rationale']}")
            
            if medium_priority:
                print("\n  🟡 MEDIUM PRIORITY:")
                for cat, item in medium_priority[:3]:
                    print(f"     • [{cat.upper()}] {item['action']}")
        
        print("\n" + "="*60)
        print("  Analysis Complete ✅")
        print("="*60 + "\n")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found - {args.file}")
        print("   Run with --help for usage information.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
