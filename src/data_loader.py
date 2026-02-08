"""
Data Loader Module
Handles loading, validation, and preprocessing of retail sales data.
"""

from pathlib import Path
from typing import IO, Optional

import pandas as pd


class DataLoader:
    """
    Handles all data ingestion operations for the retail analytics app.
    """

    REQUIRED_COLUMNS = [
        'date', 'product_id', 'product_name', 'category',
        'quantity_sold', 'unit_price', 'revenue'
    ]

    def __init__(self, source: IO, source_name: Optional[str] = None):
        self.source = source
        self.source_name = source_name
        self.data = None
        self.validation_errors = []

    def _get_source_label(self) -> str:
        if self.source_name:
            return self.source_name
        name = getattr(self.source, "name", None)
        return name or "uploaded_file"

    def _read_source(self) -> pd.DataFrame:
        source = self.source
        name = self.source_name or getattr(self.source, "name", "")
        ext = Path(name).suffix.lower()
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(source)
        if ext == ".csv" or ext == "":
            return pd.read_csv(source)
        raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

    def load_data(self) -> pd.DataFrame:
        """
        Load and validate CSV data.

        Returns:
            pd.DataFrame: Cleaned sales data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        try:
            # Load raw data
            self.data = self._read_source()
            print(f"Loaded {len(self.data)} records from {self._get_source_label()}")

            # Validate structure
            self._validate_columns()
            self._validate_data_types()
            self._validate_business_rules()

            if self.validation_errors:
                error_msg = "\n".join(self.validation_errors)
                raise ValueError(f"Data validation failed:\n{error_msg}")

            # Clean and preprocess
            self._preprocess_data()

            return self.data

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self._get_source_label()}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}") from e

    def _validate_columns(self):
        """Check if all required columns are present."""
        missing = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing:
            self.validation_errors.append(f"Missing columns: {', '.join(missing)}")

    def _validate_data_types(self):
        """Validate and convert data types."""
        try:
            if 'date' not in self.data.columns:
                self.validation_errors.append("Missing date column for type conversion")
                return

            # Convert date
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            if self.data['date'].isna().any():
                self.validation_errors.append("Invalid date values found")

            # Ensure numeric columns
            numeric_cols = ['quantity_sold', 'unit_price', 'revenue']
            for col in numeric_cols:
                if col not in self.data.columns:
                    self.validation_errors.append(f"Missing {col} column for type conversion")
                    continue
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                if self.data[col].isna().any():
                    self.validation_errors.append(f"Invalid numeric values found in {col}")

        except Exception as e:
            self.validation_errors.append(f"Data type conversion error: {str(e)}")

    def _validate_business_rules(self):
        """Check business logic constraints."""
        required = ['quantity_sold', 'unit_price', 'revenue']
        if not all(col in self.data.columns for col in required):
            self.validation_errors.append("Missing columns for business rule validation")
            return

        if (self.data['quantity_sold'] < 0).any():
            self.validation_errors.append("Negative quantities found")

        if (self.data['unit_price'] < 0).any():
            self.validation_errors.append("Negative prices found")

        calculated_revenue = self.data['quantity_sold'] * self.data['unit_price']
        revenue_diff = (self.data['revenue'] - calculated_revenue).abs()
        if (revenue_diff > 0.01).any():
            self.validation_errors.append("Revenue calculation mismatch detected")

    def _preprocess_data(self):
        """Clean and enrich data."""
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        if len(self.data) < initial_count:
            print(f"Removed {initial_count - len(self.data)} duplicate rows")

        # Add derived columns
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['week'] = self.data['date'].dt.isocalendar().week
        self.data['day_of_week'] = self.data['date'].dt.day_name()
        self.data['month_name'] = self.data['date'].dt.month_name()

        # Sort by date
        self.data = self.data.sort_values('date').reset_index(drop=True)

    def get_summary_stats(self) -> dict:
        """Return quick summary statistics."""
        if self.data is None:
            return {}

        return {
            'total_records': len(self.data),
            'date_range': (self.data['date'].min(), self.data['date'].max()),
            'total_products': self.data['product_id'].nunique(),
            'total_categories': self.data['category'].nunique(),
            'total_revenue': self.data['revenue'].sum(),
            'total_units': self.data['quantity_sold'].sum()
        }
