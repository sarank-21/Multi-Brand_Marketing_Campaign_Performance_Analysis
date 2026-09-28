# 🎯 Multi-Brand Marketing Campaign Performance Analysis & Prediction

## 📌 About the Project

This project analyzes **153,252 marketing campaigns** from three beauty brands (**Nykaa, Purplle and Tira**) and predicts two things for a new campaign: its **expected Revenue** (regression) and whether it will end in **Profit or Loss** (classification). Raw brand CSVs are cleaned, merged, encoded and used to train multiple ML models, and the best ones are served through an interactive **Streamlit** app with a prediction page and an analysis dashboard. It helps marketing teams judge a campaign plan *before* spending budget, and understand which campaign types, channels and segments perform best.

---

## 🛠️ Development Process

### 1️⃣ Data Collection
- Loaded three separate brand datasets: `nykaa_campaign_data_with_nulls.csv`, `purplle_campaign_data_with_nulls.csv` and `tira_campaign_data_with_nulls.csv`.
- Each file contains campaign details such as `Campaign_Type`, `Target_Audience`, `Duration`, `Channel_Used`, `Impressions`, `Clicks`, `Leads`, `Conversions`, `Revenue`, `Acquisition_Cost`, `ROI`, `Language`, `Engagement_Score`, `Customer_Segment` and `Date`.

### 2️⃣ Data Cleaning & Preprocessing
- Checked null percentages and duplicates for every brand file.
- **Categorical nulls** (`Campaign_Type`, `Target_Audience`, `Language`, `Customer_Segment`) were filled with the **mode** of each brand.
- **Numerical nulls**: `Duration`, `Impressions`, `Engagement_Score` filled with the **mean**; `Clicks`, `Leads`, `Conversions`, `Revenue`, `Acquisition_Cost` filled with the **median** (chosen after inspecting box plots).
- `Date` converted with `pd.to_datetime` and back-filled; rows with missing `Campaign_ID` or `Channel_Used` were dropped.

### 3️⃣ Outlier Handling
- Applied the **IQR method** (Q1 − 1.5×IQR, Q3 + 1.5×IQR) on `Clicks`, `Leads`, `Conversions`, `Revenue`, `Acquisition_Cost` and `ROI`.
- Outliers were **capped** rather than removed so no campaign rows were lost; box plots were checked before and after.

### 4️⃣ Feature Engineering
- Recalculated **ROI** as `Revenue / (Acquisition_Cost × Conversions) − 1`, since `Acquisition_Cost` is a *per-conversion* cost.
- Created the target **`ROI_Flag`**: `Profit` if ROI ≥ 0, otherwise `Loss`.
- Merged the three brand files into one `final_df.csv` (153,252 rows).

### 5️⃣ Data Transformation
- **Label encoding** (`LabelEncoder`) for `Campaign_Type`, `Target_Audience`, `Language`, `Customer_Segment`; encoders saved to `label_encoders.pkl` so the app uses identical mappings.
- **Multi-label encoding** (`MultiLabelBinarizer`) for `Channel_Used` → `Channel_Email`, `Channel_Facebook`, `Channel_Google`, `Channel_Instagram`, `Channel_WhatsApp`, `Channel_YouTube`.
- **StandardScaler** applied to the 17 model features.

### 6️⃣ Imbalance Handling
- Target split is **~78% Profit / ~22% Loss** (119,493 vs 33,759).
- Used a **stratified** train/test split, `class_weight="balanced"` where supported, and judged models on **F1 of the Loss class** instead of plain accuracy.

### 7️⃣ Leakage Prevention
- `ROI`, `ROI_Flag`, `Revenue` (for classification), `Campaign_ID` and `Date` were excluded from the features, because ROI is derived from Revenue.
- Both models use the **same 17 features**, verified with an assertion in the notebook.

### 8️⃣ Model Building
- **Regression (Revenue):** Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
- **Classification (Profit/Loss):** Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting, AdaBoost, XGBoost (SVM optional via `RUN_SVM`).
- 80/20 split with `random_state=42`.

### 9️⃣ Model Evaluation
- Regression: R², MAE, MSE, RMSE. Classification: accuracy (train/test gap), precision, recall, F1.
- **5-fold stratified cross-validation** confirmed stability (mean F1 ≈ 0.784, std ≈ 0.0015).
- Ran an 8-campaign **sanity check** (see [Sample Test Cases](#-sample-test-cases)).

### 🔟 Dashboard Development
- Built a multi-page **Streamlit** app (`Main.py`) with Home, Prediction and Analysis pages using `st.session_state` navigation.
- Dropdown options come directly from the trained encoders, so every choice is guaranteed to be transformable.

### 1️⃣1️⃣ Visualization & Analysis
- Plotly charts for ROI by campaign type, revenue/ROI by channel, engagement by segment, spend/clicks vs revenue and a correlation heatmap.

### 1️⃣2️⃣ Performance Optimization
- `@st.cache_data` for the dataset and `@st.cache_resource` for all models, scalers and encoders, so they load once per session.

---

## ✨ Key Features

### 💰 Revenue Prediction
Predicts expected campaign revenue in ₹ using the best regression model (Gradient Boosting).

### ✅ Profit / Loss Prediction
Classifies a campaign as Profit or Loss using the best classifier (Random Forest).

### 📊 KPI Dashboard
Home page cards for Total Campaigns, Avg ROI, Total Revenue, Profit Rate and Target Audiences.

### 🥧 Profit vs Loss Distribution
Pie chart showing the balance of profitable and loss-making campaigns.

### 📶 Campaign Analysis
Five analysis tabs: campaign type, channel, segment, top/low performers and relationships.

### 🧹 Robust Data Cleaning
Mode/mean/median imputation with IQR outlier capping per brand.

### 🔀 Multi-Channel Support
A campaign can use several channels at once, handled through multi-label encoding.

### 🛡️ Input Validation
Required-field checks plus warnings when Clicks > Impressions, Leads > Clicks or Conversions > Leads.

### ⚡ Cached Resources
Data and models are cached so pages respond quickly.

### 🎨 Custom UI
Dark-themed KPI cards, styled buttons, colour-coded result cards (green for Profit, red for Loss).

---

## 🔍 Features (Detailed)

### 🔎 Prediction Page
- 11 required inputs plus a multi-select for channels (WhatsApp, YouTube, Google, Facebook, Instagram, Email).
- Outputs **Predicted Revenue** (clipped at ₹0) and **Predicted Outcome** (Profit ✅ / Loss ❌) in styled cards.
- Acquisition Cost is entered *per conversion*, matching how the models were trained.

### 📶 Analysis Page
- **By Campaign Type** – average ROI bar chart.
- **By Channel** – total revenue and average ROI per channel (channels exploded from the multi-value column).
- **By Segment** – engagement score box plot per customer segment.
- **Top & Low Performers** – rank by Revenue or ROI with a Top-N slider (5–20).
- **Relationships** – Spend vs Revenue, Clicks vs Revenue scatter plots and a correlation heatmap.

### 🏠 Home Page
- Live KPI cards computed from the cleaned dataset, a 100-row dataset preview and the Profit vs Loss pie chart.

---

## 🧰 Tech Stack

### 🖥️ Frontend / UI
- **Streamlit** – web app, session-state page navigation, tabs, widgets
- **HTML/CSS** – custom KPI and result cards via `st.markdown`

### 🧠 Machine Learning
- **scikit-learn** – models, `LabelEncoder`, `MultiLabelBinarizer`, `StandardScaler`, metrics, cross-validation
- **XGBoost** – gradient-boosted regressor and classifier

### 📊 Data Processing & Analysis
- **Pandas** – loading, cleaning, merging, group-by analysis
- **NumPy** – IQR calculations and clipping

### 📈 Data Visualization
- **Plotly Express** – bar, pie, box, scatter and heatmap charts

### ⚙️ Backend / Core Logic
- **Joblib** – saving/loading models, scalers, encoders and column lists
- **pathlib** – file path handling
- **warnings** – suppress non-critical warnings

### 🛠️ Development Tools
- **Jupyter Notebook / Google Colab** – cleaning and modelling
- **Git & GitHub** – version control

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sarank-21/Multi-Brand_Marketing_Campaign_Performance_Analysis.git
cd Multi-Brand_Marketing_Campaign_Performance_Analysis
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
Key libraries: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `xgboost`, `joblib`.

### 4️⃣ Prepare the Dataset
Place the cleaned dataset at:
```
CSV/final_df.csv
```
To rebuild it, run `Multi_Brand_Marketing_Campaign_Data_Cleaning.ipynb` on the three raw brand CSVs.

### 5️⃣ Train the Models
Run `Multi_Brand_Marketing_Campaign_Model.ipynb` (*Restart and run all*). It generates the 7 files the app needs, next to `Main.py`:
```
best_classification_model.pkl   classification_scaler.pkl   model_columns_class.pkl
best_Regression_model.pkl       Regression_scaler.pkl       model_columns_regression.pkl
label_encoders.pkl
```

### 6️⃣ Run the Application
```bash
streamlit run Main.py
```

### 7️⃣ Optional: Clear Cache
```bash
streamlit cache clear
```

---

## 🧪 Sample Test Cases

Use these real campaigns from the held-out test split to check the app or the notebook. Enter the values on the **Prediction** page.

### ✅ Should predict **Profit**

| Field | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| Campaign Type | SEO | Paid Ads | Paid Ads | Email |
| Target Audience | Premium Shoppers | Working Women | Youth | Premium Shoppers |
| Duration | 15 | 14 | 23 | 5 |
| Channels | Email | YouTube, Facebook | WhatsApp, Google, YouTube | YouTube |
| Impressions | 90648 | 55820 | 76381 | 67277 |
| Clicks | 5309 | 5087 | 8599 | 7652 |
| Leads | 2712 | 2937 | 4847 | 3436 |
| Conversions | 1769 | 1280 | 1621 | 777 |
| Acquisition Cost | 168.01 | 66.28 | 170.21 | 176.84 |
| Language | Hindi | Tamil | English | Hindi |
| Engagement Score | 10.8 | 16.67 | 19.73 | 18.98 |
| Customer Segment | College Students | College Students | Working Women | Tier 2 City Customers |

### ❌ Should predict **Loss**

| Field | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Campaign Type | Social Media | Social Media | Influencer | Social Media |
| Target Audience | Youth | Working Women | Premium Shoppers | College Students |
| Duration | 30 | 8 | 30 | 25 |
| Channels | Google, YouTube | Instagram | Instagram | YouTube, Email |
| Impressions | 42242 | 15390 | 11518 | 15328 |
| Clicks | 1050 | 633 | 828 | 1661 |
| Leads | 217 | 352 | 224 | 623 |
| Conversions | 70 | 264 | 775 | 231 |
| Acquisition Cost | 850.34 | 862.96 | 862.96 | 862.96 |
| Language | Hindi | Hindi | Bengali | Hindi |
| Engagement Score | 3.17 | 8.12 | 10.19 | 16.41 |
| Customer Segment | Working Women | Premium Shoppers | Working Women | Premium Shoppers |

---

## 📊 Model Performance

| Task | Best Model | Key Metrics |
|---|---|---|
| Revenue (Regression) | Gradient Boosting | R² 0.7447 · MAE ₹134,492 · RMSE ₹183,315 |
| Profit / Loss (Classification) | Random Forest | Accuracy 0.9035 · Recall (Loss) 0.8283 · F1 (Loss) 0.7908 |

---

## 💼 Use Case

1. **Pre-launch Campaign Evaluation** – Check whether a planned campaign is likely to make a profit before committing budget.
2. **Budget Allocation** – Compare acquisition cost against predicted revenue to decide where spend is worth it.
3. **Channel Strategy** – See which channels drive the most revenue and ROI, and combine them wisely.
4. **Audience Targeting** – Understand engagement across customer segments and languages.
5. **Performance Review** – Identify the top and bottom campaigns by Revenue or ROI.
6. **Multi-Brand Benchmarking** – Compare Nykaa, Purplle and Tira campaign patterns in one place.

---

## 🚀 Future Enhancements

1. **Explainability** – Add SHAP/LIME to show why a campaign is predicted as Profit or Loss.
2. **Hyperparameter Tuning** – GridSearch/Optuna to push regression R² beyond 0.74.
3. **Funnel Features** – Reintroduce CTR, Lead Rate and Conversion Rate as engineered features.
4. **Brand as a Feature** – Include the brand in the model for brand-specific predictions.
5. **Database Integration** – Store predictions and campaign history in MySQL/PostgreSQL.
6. **Automated Reporting** – Export analysis and predictions as PDF/Excel reports.
7. **Batch Prediction** – Upload a CSV of planned campaigns and score them all at once.
8. **Cloud Deployment** – Host on Streamlit Community Cloud or a container platform.

---

## 🔄 How It Works

```
┌───────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI  (Main.py)                    │
│         🏠 home   →   🔎 Prediction   |   📶 Analysis          │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│               SESSION STATE  (st.session_state.page)          │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     APPLICATION LOGIC LAYER                   │
│      options_for()  ·  style_fig()  ·  input validation       │
└──────────────┬──────────────────┬──────────────────┬──────────┘
               │                  │                  │
               ▼                  ▼                  ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│    DATA PIPELINE     │ │     ML PIPELINE      │ │    OUTPUT ENGINE     │
│ load_data()          │ │ LabelEncoder         │ │ Predicted Revenue ₹  │
│ drop dead rows       │ │ MultiLabelBinarizer  │ │ Profit / Loss card   │
│ final_df.csv         │ │ StandardScaler       │ │ KPI cards + charts   │
│                      │ │ Regressor (GB)       │ │                      │
│                      │ │ Classifier (RF)      │ │                      │
└──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    ▼
┌───────────────────────────────────────────────────────────────┐
│                     CACHED RESOURCES LAYER                    │
│   @st.cache_data  → load_data()                               │
│   @st.cache_resource → load_models()  (.pkl files)            │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                       ARTIFACT LAYER                          │
│  best_Regression_model.pkl · best_classification_model.pkl    │
│  Regression_scaler.pkl · classification_scaler.pkl            │
│  model_columns_regression.pkl · model_columns_class.pkl       │
│  label_encoders.pkl                                           │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                        ANALYSIS LAYER                         │
│  Campaign Type · Channel · Segment · Top/Low · Relationships  │
└───────────────────────────────────────────────────────────────┘
```

---

## 📘 Project Overview

This project is a supervised machine learning system for marketing analytics that studies over 153K campaigns across Nykaa, Purplle and Tira. After per-brand cleaning (mode, mean and median imputation with IQR capping), the data is merged, label-encoded, multi-label encoded for channels and scaled into 17 leakage-free features. A Gradient Boosting regressor estimates campaign revenue (R² ≈ 0.74), while a class-weighted Random Forest classifies each campaign as Profit or Loss (accuracy ≈ 90%, Loss-class F1 ≈ 0.79) on a ~78/22 imbalanced target. The Streamlit app exposes both models through a validated prediction form and adds a dashboard with KPI cards and Plotly analyses of campaign type, channel, segment, top/low performers and correlations. Together they help marketers plan spend, pick channels and avoid loss-making campaigns before launch.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
