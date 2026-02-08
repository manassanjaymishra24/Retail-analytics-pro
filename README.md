# 📊 Retail Demand Analytics App

A **production-ready retail analytics dashboard** that transforms raw sales data into **actionable business insights**.  
Built with **Python, Pandas, Streamlit, and Plotly**, the app supports **dynamic client data uploads**, **automatic schema detection**, and **end-to-end analysis** — with **no bundled sample datasets**.

---

## 🚀 Key Highlights

- 🔄 Client-driven data uploads (CSV / Excel)
- 🧠 Automatic column mapping & schema normalization
- 📈 Time-series trend and growth analysis
- 🛍️ Product & category performance insights
- 🔮 Demand forecasting
- 💡 Strategic business recommendations
- 🎨 Modern, executive-grade UI/UX

---

## 🎯 Features

### 📥 Data Ingestion
- Upload CSV or Excel files directly through the UI
- Automatic detection of column names (schema-agnostic)
- Built-in data validation and cleaning
- Revenue auto-calculated if missing

### 📊 Trend Analysis
- Daily, weekly, and monthly revenue trends
- Growth rate analysis
- Seasonality detection
- Moving-average forecasting

### 🛍️ Product Intelligence
- Top and bottom product rankings
- Category-level revenue analysis
- Price sensitivity and performance scoring
- Product growth classification (Growing / Stable / Declining)

### 💡 Strategic Recommendations
- Inventory optimization suggestions
- Pricing strategy insights
- Marketing and operational recommendations
- Priority-based action items (High / Medium / Low)

### 🧭 Interactive Dashboard
- Executive KPI cards
- Plotly-powered interactive charts
- Filterable data explorer
- Exportable filtered datasets

---

## 🏗️ Project Structure
retail-demand-analytics/
├── src/
│ ├── data_loader.py
│ ├── trend_analysis.py
│ ├── inventory_analysis.py
│ ├── recommendations.py
│ └── init.py
├── tests/
├── app.py
├── cli.py
├── README.md
├── requirements.txt
└── .gitignore

> ⚠️ No bundled datasets — the app runs entirely on **user-uploaded data**.

---

## 📂 Expected Data Format

The app automatically maps column names, but the following **logical fields** are required:

| Logical Field    | Required | Notes |
|------------------|----------|------|
| `product_id`    | ✅ | SKU / Item ID |
| `product_name`  | ✅ | Product name |
| `category`      | ✅ | Product category |
| `date`          | ✅ | Order / sales date |
| `unit_price`    | ✅ | Price per unit |
| `quantity_sold` | ✅ | Units sold |
| `revenue`       | ❌ | Auto-calculated if missing |

Column names **do not need to match exactly** — the app auto-detects common variants.

---

## 🚀 Quick Start

### 1️⃣ Create Virtual Environment


python -m venv venv
### 2️⃣ Activate Environment

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate
### 3️⃣ Install Dependencies
pip install -r requirements.txt
### 4️⃣ Run the App
streamlit run app.py

Open the browser link shown in the terminal.

### 🧠 How the App Works

User uploads a sales dataset

Columns are auto-detected and normalized

Data is validated and cleaned

Analytics engines compute trends, forecasts, and insights

Interactive dashboards and recommendations are rendered

### 🛠 Tech Stack

Python

Pandas & NumPy

Streamlit

Plotly

Matplotlib (optional, used for advanced table styling)

### 📌 Project Status

✅ Core analytics complete

✅ Production-ready UI/UX

✅ Dynamic client data support

🚧 Advanced forecasting models (planned enhancement)

### 🎓 Use Cases

Retail sales analysis

Business intelligence dashboards

Data analytics portfolios

Interview & internship projects

Internal analytics tools

### 📄 License

This project is intended for educational and portfolio use.
You are free to extend and adapt it for personal or academic projects.

### ⭐ Final Note

This project is designed to reflect real-world analytics workflows, not toy datasets.
It emphasizes data robustness, modular design, and business relevance.
