# 🎯 Multi-Brand Marketing Campaign Performance Analysis & Prediction

## 📌 About the Project

This project analyzes marketing campaign performance across three e-commerce brands — **Nykaa, Purplle, and Tira** — and builds machine learning models to predict campaign outcomes before they run. It combines a full data cleaning and feature-engineering pipeline with a **regression model that forecasts Revenue** and a **classification model that flags a campaign as Profit or Loss**, all wrapped in an interactive Streamlit dashboard. Marketing teams can use it to explore historical performance by channel, campaign type, and customer segment, and to simulate a new campaign's expected revenue and profitability before committing budget to it.

---

## 🛠️ Development Process

1. **Data Collection**
   - Loaded three separate raw datasets — `nykaa_campaign_data_with_nulls.csv`, `purplle_campaign_data_with_nulls.csv`, `tira_campaign_data_with_nulls.csv` — each containing intentional null values to simulate real-world messy data.
   - Inspected each brand's dataset independently with `.info()`, `.isnull().mean()`, and `.duplicated().sum()` before touching it.

2. **Data Cleaning & Preprocessing**
   - Handled categorical nulls (`Campaign_Type`, `Target_Audience`, `Language`, `Customer_Segment`) by imputing each brand's own mode (e.g., Nykaa's `Campaign_Type` nulls filled with `"Paid Ads"`, Purplle's with `"Influencer"`, Tira's with `"SEO"`).
   - Converted `Date` to datetime and back-filled missing dates with `bfill`.
   - Dropped rows with missing `Campaign_ID` or `Channel_Used`, since these can't be safely imputed.
   - Filled numeric nulls per-column based on distribution shape: mean-imputed `Duration`, `Impressions`, and `Engagement_Score`; median-imputed `Clicks`, `Leads`, `Conversions`, `Revenue`, and `Acquisition_Cost`.

3. **Outlier Handling**
   - Applied the IQR method (`Q1`, `Q3`, `1.5×IQR` bounds) to `Clicks`, `Leads`, `Conversions`, `Revenue`, `Acquisition_Cost`, and `ROI` for every brand.
   - Used `np.clip()` to cap outliers at the bounds rather than dropping rows, preserving dataset size while controlling extreme values.

4. **Feature Engineering**
   - Engineered `ROI` as `(Revenue / (Acquisition_Cost × Conversions)) − 1` for each brand after outlier treatment on the raw fields.
   - Derived the target label `ROI_Flag` (`"Profit"` if `ROI >= 0`, else `"Loss"`) to power the classification task.
   - Merged the three cleaned brand datasets with `pd.concat()` into a single `final_df` and reset the index.

5. **Multi-Label Encoding**
   - Since a campaign can run on more than one channel at once, `Channel_Used` (a comma-separated string like `"WhatsApp, Google"`) was split and encoded with `MultiLabelBinarizer` into six binary columns — one per channel (`WhatsApp`, `YouTube`, `Google`, `Facebook`, `Instagram`, `Email`) — instead of one-hot encoding a single categorical column.
   - Encoded `Campaign_Type`, `Target_Audience`, `Language`, and `Customer_Segment` with `LabelEncoder`, saving all four fitted encoders together in `label_encoders.pkl` for reuse at inference time.

6. **Model Building — Revenue Regression**
   - Feature set: all engineered columns except `Campaign_ID`, `ROI`, `ROI_Flag`, `Revenue` (the target), `Language`, `Campaign_Type`, and `Date`.
   - Scaled features with `StandardScaler` and benchmarked six regressors — Linear Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
   - Selected the model with the highest R² score on the held-out test split and persisted it as `best_Regression_model.pkl` alongside `Regression_scaler.pkl` and the training column order (`model_columns_regression.pkl`).

7. **Model Building — Profit/Loss Classification**
   - Feature set: all engineered columns except `Campaign_ID`, `ROI`, `ROI_Flag` (the target), `Language`, `Campaign_Type`, and `Date` — deliberately **keeping Revenue as a feature while excluding ROI itself**, since ROI is derived directly from Revenue and would leak the answer.
   - Benchmarked nine classifiers with `class_weight="balanced"` where supported — Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, AdaBoost, and XGBoost.
   - Selected the model with the highest F1 score and persisted it as `best_classification_model.pkl` alongside `classification_scaler.pkl` and `model_columns_class.pkl`.

8. **Dashboard Development**
   - Built a three-page Streamlit app (`Home`, `Prediction`, `Analysis`) driven by `st.session_state.page`, with `@st.cache_data` for the cleaned dataset and `@st.cache_resource` for the loaded models/scalers/encoders so they load once per session.

9. **Visualization & Analysis**
   - Used Plotly Express throughout for bar, box, scatter, pie, and correlation-heatmap charts, styled with a consistent dark theme (`paper_bgcolor`, `plot_bgcolor`, custom hover labels).

10. **Prediction System**
    - Built a live prediction form that validates every required field, applies the same encoders/scalers used in training, and returns both the predicted Revenue and the predicted Profit/Loss outcome side by side.

---

## ✨ Key Features

### 🔎 Live Campaign Prediction
Enter a hypothetical campaign's details and instantly get a predicted Revenue figure and a Profit/Loss classification, computed from the same pipeline used in training.

### 📊 KPI Dashboard
The home page surfaces total campaigns, average ROI, total revenue, profit rate, and target-audience count as live metric cards computed from the cleaned dataset.

### 📶 Multi-Angle Performance Analysis
A tabbed analysis view breaks performance down by Campaign Type, Channel, Customer Segment, Top & Bottom performers, and cross-variable relationships.

### 🗂️ Multi-Brand Data Pipeline
Three independently-sourced, independently-cleaned brand datasets (Nykaa, Purplle, Tira) are merged into one unified dataset without losing brand-specific null-handling nuances.

### 🏷️ Multi-Label Channel Encoding
Channels are modeled as a true multi-label field via `MultiLabelBinarizer`, correctly reflecting that a single campaign can run across several channels at once.

### 🧮 Dual Model Architecture
One regression model (Revenue) and one classification model (Profit/Loss) run side-by-side from a shared feature pipeline, giving both a number and a verdict for every prediction.

### 🚫 Leakage-Aware Feature Selection
ROI is deliberately excluded from both models' feature sets — it's used only to derive the classification label — so no model ever sees information it's trying to predict.

### ⚡ Cached Data & Models
`st.cache_data` and `st.cache_resource` keep the dataset and the six persisted model artifacts (`.pkl` files) from being reloaded on every interaction.

### 🎨 Custom Dark-Themed UI
Hand-styled KPI cards, prediction result cards, and buttons (via injected CSS) give the dashboard a consistent, polished look beyond Streamlit's defaults.

### 🔗 Correlation & Relationship Explorer
An interactive correlation heatmap plus spend-vs-revenue and clicks-vs-revenue scatter plots let users visually probe what actually drives campaign outcomes.

---

## 🧩 Features (Detailed)

### Home Page
- Displays five KPI cards: Total Campaigns, Average ROI, Total Revenue, Profit Rate, and number of unique Target Audiences.
- Shows a searchable/scrollable preview (`st.dataframe`) of the first 100 rows of the cleaned dataset.
- Renders a Profit vs Loss donut/pie chart based on the engineered `ROI_Flag` column.

### Prediction Page
- Two-column input form covering all 12 raw campaign attributes: Campaign Type, Target Audience, Duration, Channel Used (multi-select), Impressions, Clicks, Leads, Conversions, Acquisition Cost, Language, Engagement Score, Customer Segment, and Date.
- Validates that every field is filled before running a prediction, listing exactly which fields are missing if not.
- Deliberately does **not** ask the user for ROI, since ROI is derived from Revenue — the very thing being predicted — which would make it both leakage and impossible for a real user to supply.
- Builds two parallel encoded feature vectors (one reindexed to the regression model's training columns, one to the classification model's), scales each with its own saved scaler, and displays the predicted Revenue and Profit/Loss outcome as two result cards, color-coded green for Profit and red for Loss.

### Analysis Page
- **By Campaign Type** — bar chart of average ROI per campaign type.
- **By Channel** — side-by-side bar charts of total revenue and average ROI per channel, after exploding the multi-label `Channel_Used` field.
- **By Segment** — box plot of Engagement Score distribution across customer segments.
- **Top & Low Performers** — user-adjustable ranking (by Revenue or ROI) and count (5–20) showing the best- and worst-performing campaigns side by side.
- **Relationships** — scatter plots of Acquisition Cost vs Revenue and Clicks vs Revenue (both colored by ROI), plus a correlation heatmap across all key numeric fields.

---

## 🧰 Tech Stack

### 🖥️ Frontend / UI
- **Streamlit** — multi-page dashboard framework with session-state-driven navigation
- **Custom CSS** — injected via `st.markdown(..., unsafe_allow_html=True)` for buttons, KPI cards, and result cards

### 🧠 Machine Learning
- **scikit-learn** — `LabelEncoder`, `MultiLabelBinarizer`, `StandardScaler`, `train_test_split`, `LinearRegression`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `GaussianNB`, `AdaBoostClassifier`
- **XGBoost** — `XGBRegressor`, `XGBClassifier`
- **joblib** — persisting and loading trained models, scalers, encoders, and column orders

### 📊 Data Processing & Analysis
- **pandas** — cleaning, merging, grouping, and feature engineering across the three brand datasets
- **NumPy** — IQR outlier bound calculation and `np.clip()`-based capping

### 📈 Data Visualization
- **Plotly Express** — bar, box, scatter, pie, and correlation-heatmap charts across both notebooks and the app

### 🚀 Deployment & Optimization
- **`st.cache_data`** — caches the loaded/cleaned dataset
- **`st.cache_resource`** — caches loaded models, scalers, and encoders

### 🛠️ Development Tools
- **Jupyter / Google Colab Notebooks** — `Multi_Brand_Marketing_Campaign_Data_Cleaning.ipynb` (cleaning + feature engineering) and `Multi_Brand_Marketing_Campaign_EDA.ipynb` (EDA + encoding + model training)

---

## ⚙️ Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sarank-21/Multi-Brand_Marketing_Campaign_Performance_Analysis.git
   cd D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis
   ```

2. **Create a Virtual Environment**
   ```bash
   # Windows
   python -m venv mmc
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `xgboost`, `joblib`

4. **Prepare the Dataset & Model Artifacts**
   - Place the cleaned dataset at `CSV/final_df.csv` (path used in `app.py`).
   - Ensure the following model artifacts (produced by the EDA notebook) sit alongside `app.py`: `best_classification_model.pkl`, `classification_scaler.pkl`, `best_Regression_model.pkl`, `Regression_scaler.pkl`, `model_columns_class.pkl`, `model_columns_regression.pkl`, `label_encoders.pkl`.

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

6. **Optional: Clear Cache**
   If you update the dataset or retrain a model, clear Streamlit's cache from the app's menu (⋮ → "Clear cache") or restart the app so `@st.cache_data`/`@st.cache_resource` reload the new files.

---

## 💼 Use Case

1. **Pre-Launch Budget Planning** — Marketing teams can simulate a planned campaign's inputs before spending money, to see its predicted Revenue and likely Profit/Loss outcome.
2. **Channel Strategy** — Compare average ROI and total revenue across WhatsApp, YouTube, Google, Facebook, Instagram, and Email to decide where to allocate budget.
3. **Segment Targeting** — Use the Engagement Score breakdown by Customer Segment to identify which audiences respond best to campaigns.
4. **Cross-Brand Benchmarking** — Since Nykaa, Purplle, and Tira data are unified, campaign types and channels can be benchmarked across brands rather than in isolation.
5. **Performance Triage** — Quickly surface the top and bottom N campaigns by Revenue or ROI to investigate what's working and what isn't.
6. **Spend Sensitivity Analysis** — Explore the Acquisition Cost vs Revenue scatter plot and correlation heatmap to understand diminishing returns on spend.

---

## 🚀 Future Enhancements

1. Add SHAP or LIME-based explainability so users can see *why* a given campaign was predicted to be Profit or Loss.
2. Persist prediction history to a database for tracking predicted vs. actual outcomes over time.
3. Add hyperparameter tuning (GridSearchCV/Optuna) on top of the current model comparison to squeeze out further R²/F1 gains.
4. Support real-time ad-platform API integration (Google Ads, Meta) to pull live campaign metrics instead of static CSVs.
5. Add automated PDF/Excel report generation summarizing a batch of predictions.
6. Introduce user authentication and per-user saved campaign scenarios.
7. Extend the multi-brand pipeline to onboard new brands without manual per-brand null-value handling.
8. Add confidence intervals or prediction uncertainty bands to the Revenue forecast.

---

## 🏗️ How It Works

```
                    ┌─────────────────────────────┐
                    │      Streamlit UI Pages      │
                    │   Home | Prediction | Analysis│
                    └───────────────┬─────────────┘
                                    │
                    ┌───────────────▼─────────────┐
                    │   st.session_state.page      │
                    │      (navigation state)      │
                    └───────────────┬─────────────┘
                                    │
                    ┌───────────────▼─────────────┐
                    │   Application Logic Layer    │
                    └──┬──────────────┬─────────┬──┘
                       │              │         │
           ┌───────────▼───┐  ┌───────▼─────┐ ┌─▼──────────────┐
           │  Data Pipeline │  │ ML Pipeline │ │  Output Engine  │
           │  load_data()   │  │load_models()│ │ predicted_revenue│
           │  (clean CSV)   │  │ (6 .pkl's)  │ │ predicted_label  │
           └───────┬────────┘  └──────┬──────┘ └─────────┬───────┘
                    │                  │                  │
           ┌────────▼──────────────────▼──────────────────▼───────┐
           │            Cached Resources Layer                    │
           │   @st.cache_data (final_df) · @st.cache_resource     │
           │   (clf_model, clf_scaler, reg_model, reg_scaler,     │
           │    label_encoders, column orders)                    │
           └──────────────────────────┬────────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │      Analysis Layer      │
                          │  Plotly Express charts:  │
                          │  bar · box · scatter ·   │
                          │  pie · correlation heatmap│
                          └──────────────────────────┘
```

---

## 📖 Project Overview

This project is an end-to-end supervised machine learning system built on multi-brand e-commerce marketing campaign data from Nykaa, Purplle, and Tira. It combines a regression model that forecasts campaign Revenue with a separate classification model that predicts whether a campaign will land in Profit or Loss, using a shared, leakage-aware feature pipeline built on `StandardScaler`-normalized, `LabelEncoder`/`MultiLabelBinarizer`-encoded campaign attributes. The underlying data went through brand-specific null imputation, IQR-based outlier capping, and ROI feature engineering before being merged into a single 150,000+ row dataset. Both models were selected from a bake-off of six regressors and nine classifiers respectively, chosen by R² and F1 score. The resulting artifacts power a three-page Streamlit dashboard that lets users explore historical performance by channel, segment, and campaign type, and run live "what-if" predictions for a hypothetical new campaign. For a marketing team, this turns campaign planning from a purely retrospective reporting exercise into a forward-looking budgeting and channel-selection tool.

---

⭐ **If you find this project useful, give it a star on GitHub and share your feedback!**
