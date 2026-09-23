import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Brand Marketing Campaign Performance Analysis",
    page_icon="🎯",
    layout="wide",
)

if "page" not in st.session_state:
    st.session_state.page = "home"

CHANNELS = ["WhatsApp", "YouTube", "Google", "Facebook", "Instagram", "Email"]

# --------------------------------------------------
# LOAD DATA & MODELS
# --------------------------------------------------
DATA_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\CSV\final_df.csv"
CLF_MODEL_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\best_classification_model.pkl"
CLF_SCALER_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\classification_scaler.pkl"
REG_MODEL_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\best_Regression_model.pkl"
REG_SCALER_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\Regression_scaler.pkl"
CLF_COLUMNS_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\model_columns_class.pkl"
REG_COLUMNS_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\model_columns_regression.pkl"
LABEL_ENCODERS_PATH = r"D:\PROJECTS\Anna_Project_3\Multi-Brand_Marketing_Campaign_Performance_Analysis\label_encoders.pkl"

LABEL_COLUMNS = ['Campaign_Type', 'Target_Audience', 'Language', 'Customer_Segment']


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Drop the leftover pandas index column that shows up if the CSV was
    # saved without index=False (appears as "Unnamed: 0").
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed(: 0)?$")]
    return df


@st.cache_resource
def load_models():
    clf_model = joblib.load(CLF_MODEL_PATH)
    clf_scaler = joblib.load(CLF_SCALER_PATH)
    clf_columns = joblib.load(CLF_COLUMNS_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    reg_model = joblib.load(REG_MODEL_PATH)
    reg_scaler = joblib.load(REG_SCALER_PATH)
    reg_columns = joblib.load(REG_COLUMNS_PATH)
    return clf_model, clf_scaler, clf_columns, label_encoders, reg_model, reg_scaler, reg_columns


try:
    clean_df = load_data()
    (clf_model, clf_scaler, clf_columns, label_encoders,
     reg_model, reg_scaler, reg_columns) = load_models()
    MODELS_READY = True
except Exception as e:
    MODELS_READY = False
    LOAD_ERROR = str(e)

# --------------------------------------------------
# SHARED STYLING
# NOTE: every line below starts at column 0 on purpose. Streamlit's
# markdown renderer treats any line indented 4+ spaces as a code block,
# so HTML/CSS passed to st.markdown must never be pretty-printed with
# Python-style indentation -- even nested tags stay flush left.
# --------------------------------------------------
st.markdown("""
<style>
div.stButton > button {
width: 200px;
height: 55px;
background: rgba(255, 255, 255, 0.05);
color: #e5e7eb;
border: 1px solid rgba(255, 255, 255, 0.15);
border-radius: 12px;
font-size: 16px;
font-weight: 600;
transition: all 0.25s ease;
}
div.stButton > button:hover {
background: rgba(96, 165, 250, 0.12);
border: 1px solid #60a5fa;
color: #60a5fa;
transform: scale(1.03);
box-shadow: 0 0 18px rgba(96, 165, 250, 0.18);
}
div.stButton > button:active {
transform: scale(0.98);
}
div.stButton > button:focus {
outline: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Multi-Brand Marketing Campaign Performance Analysis")

if not MODELS_READY:
    st.error(
        "Couldn't load data/model files. Update the paths at the top of this "
        f"script to point to your files.\n\nDetails: {LOAD_ERROR}"
    )
    st.stop()

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if st.session_state.page == "home":

    col1, col2, col3, col4 = st.columns(4)
    with col2:
        if st.button("🔎 Prediction"):
            st.session_state.page = "Prediction"
            st.rerun()
    with col3:
        if st.button("📶 Analysis"):
            st.session_state.page = "Analysis"
            st.rerun()

    # Calculate values
    total_campaigns = len(clean_df)

    avg_roi = (
        clean_df["ROI"].mean()
        if "ROI" in clean_df.columns
        else 0
    )

    total_revenue = (
        clean_df["Revenue"].sum()
        if "Revenue" in clean_df.columns
        else 0
    )

    profit_rate = (
        (clean_df["ROI_Flag"].astype(str).str.lower() == "profit").mean() * 100
        if "ROI_Flag" in clean_df.columns
        else 0
    )

    target_audience = (
        clean_df["Target_Audience"].nunique()
        if "Target_Audience" in clean_df.columns
        else 0
    )

    profit_accent = "#4ade80" if profit_rate >= 50 else "#f87171"

    # KPI CSS -- flush left, see note above
    st.markdown("""
<style>
.metric-container {
display: flex;
width: 100%;
gap: 20px;
margin-bottom: 25px;
}
.metric-card {
flex: 1;
min-width: 0;
text-align: center;
padding: 18px 10px;
border-radius: 14px;
background: #1a1d26;
border: 1px solid #2a2e3a;
border-top: 4px solid var(--accent);
box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
.metric-icon {
font-size: 20px;
margin-bottom: 6px;
}
.metric-label {
font-size: 13px;
font-weight: 500;
color: #9aa0ac;
margin-bottom: 8px;
white-space: nowrap;
}
.metric-value {
font-size: 30px;
font-weight: 700;
color: var(--accent);
line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

    # KPI CARDS -- flush left, see note above
    html = f"""
<div class="metric-container">
<div class="metric-card" style="--accent:#60a5fa">
<div class="metric-icon">📊</div>
<div class="metric-label">Total Campaigns</div>
<div class="metric-value">{total_campaigns:,}</div>
</div>
<div class="metric-card" style="--accent:#c084fc">
<div class="metric-icon">📈</div>
<div class="metric-label">Avg ROI</div>
<div class="metric-value">{avg_roi:.2f}</div>
</div>
<div class="metric-card" style="--accent:#fbbf24">
<div class="metric-icon">💰</div>
<div class="metric-label">Total Revenue</div>
<div class="metric-value">₹{total_revenue:,.0f}</div>
</div>
<div class="metric-card" style="--accent:{profit_accent}">
<div class="metric-icon">🎯</div>
<div class="metric-label">Profit Rate</div>
<div class="metric-value">{profit_rate:.1f}%</div>
</div>
<div class="metric-card" style="--accent:#2dd4bf">
<div class="metric-icon">👥</div>
<div class="metric-label">Target Audiences</div>
<div class="metric-value">{target_audience}</div>
</div>
</div>
"""

    st.markdown(html, unsafe_allow_html=True)

    st.subheader("Dataset Overview")
    st.dataframe(clean_df.head(100), use_container_width=True)

    if "ROI_Flag" in clean_df.columns:
        fig = px.pie(
            clean_df,
            names="ROI_Flag",
            title="Profit vs Loss Distribution",
            color_discrete_sequence=["#4ade80", "#f87171"],
        )
        fig.update_layout(title_x=0.34,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"),
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e5e7eb")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif st.session_state.page == "Prediction":

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.subheader("📋 Marketing Campaign Details")
    st.caption("All fields are required.")

    col1, col2 = st.columns(2)
    with col1:
        Campaign_Type = st.selectbox(
            "Campaign Type",
            ["Paid Ads", "Influencer", "SEO", "Email", "Social Media"],
            index=None,
            placeholder="Select campaign type",
        )
        Target_Audience = st.selectbox(
            "Target Audience",
            ["Premium Shoppers", "Working Women", "Youth", "Tier 2 City Customers", "College Students"],
            index=None,
            placeholder="Select target audience",
        )
        Duration = st.number_input(
            "Duration (days)", min_value=0.0, max_value=365.0, value=None, placeholder="Enter duration"
        )
        Channel_Used = st.multiselect("Channel Used", CHANNELS)
        Impressions = st.number_input("Impressions", min_value=0.0, value=None, placeholder="Enter impressions")
        Clicks = st.number_input("Clicks", min_value=0.0, value=None, placeholder="Enter clicks")
        Leads = st.number_input("Leads", min_value=0.0, value=None, placeholder="Enter leads")
    with col2:
        Conversions = st.number_input("Conversions", min_value=0.0, value=None, placeholder="Enter conversions")
        Acquisition_Cost = st.number_input(
            "Acquisition Cost", min_value=0.0, value=None, placeholder="Enter acquisition cost"
        )
        Language = st.selectbox(
            "Language", ["Hindi", "Bengali", "Tamil", "English"], index=None, placeholder="Select language"
        )
        Engagement_Score = st.number_input(
            "Engagement Score", min_value=0.0, max_value=50.0, value=None, placeholder="Enter engagement score"
        )
        Customer_Segment = st.selectbox(
            "Customer Segment",
            ["Youth", "College Students", "Working Women", "Premium Shoppers", "Tier 2 City Customers"],
            index=None,
            placeholder="Select customer segment",
        )
        Date = st.date_input("Date", value=None)

    if st.button("🔎 Predict"):

        # ---- Validate every field before doing anything else ----
        required_fields = {
            "Campaign Type": Campaign_Type,
            "Target Audience": Target_Audience,
            "Duration": Duration,
            "Impressions": Impressions,
            "Clicks": Clicks,
            "Leads": Leads,
            "Conversions": Conversions,
            "Acquisition Cost": Acquisition_Cost,
            "Language": Language,
            "Engagement Score": Engagement_Score,
            "Customer Segment": Customer_Segment,
            "Date": Date,
        }

        missing = [name for name, val in required_fields.items() if val is None]
        if not Channel_Used:
            missing.append("Channel Used")

        if missing:
            st.error("Please fill in the following before predicting: " + ", ".join(missing))
            st.stop()

        # NOTE: ROI is intentionally NOT collected — it's derived from
        # Revenue, which is what's being predicted, so asking for it here
        # would be both leakage and impossible for the user to know.
        raw_input = {
            "Campaign_Type": Campaign_Type,
            "Target_Audience": Target_Audience,
            "Duration": Duration,
            "Impressions": Impressions,
            "Clicks": Clicks,
            "Leads": Leads,
            "Conversions": Conversions,
            "Acquisition_Cost": Acquisition_Cost,
            "Language": Language,
            "Engagement_Score": Engagement_Score,
            "Customer_Segment": Customer_Segment,
        }
        for ch in CHANNELS:
            raw_input[f"Channel_{ch}"] = 1 if ch in Channel_Used else 0

        input_df = pd.DataFrame([raw_input])

        # ---- Build REGRESSION input ----
        input_df_reg = input_df.copy()
        for col in LABEL_COLUMNS:
            input_df_reg[col] = label_encoders[col].transform(input_df_reg[col])
        input_encoded_reg = input_df_reg.reindex(columns=reg_columns, fill_value=0)

        # ---- Build CLASSIFICATION input ----
        input_df_clf = input_df.copy()
        for col in LABEL_COLUMNS:
            input_df_clf[col] = label_encoders[col].transform(input_df_clf[col])
        input_encoded_clf = input_df_clf.reindex(columns=clf_columns, fill_value=0)

        # ---- Revenue (regression) ----
        input_scaled_reg = reg_scaler.transform(input_encoded_reg)
        predicted_revenue = reg_model.predict(input_scaled_reg)[0]

        # ---- Profit / Loss (classification) ----
        input_scaled_clf = clf_scaler.transform(input_encoded_clf)
        predicted_class = clf_model.predict(input_scaled_clf)[0]
        predicted_label = "Profit" if predicted_class == 1 else "Loss"

        st.divider()
        # ----------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------
        result_accent = "#4ade80" if predicted_label == "Profit" else "#f87171"

        result_bg = (
            "rgba(74, 222, 128, 0.10)"
            if predicted_label == "Profit"
            else "rgba(248, 113, 113, 0.10)"
        )

        # Result card CSS -- flush left, see note at top of file
        st.markdown("""
<style>
.result-container {
display: flex;
gap: 20px;
width: 100%;
margin-top: 20px;
}
.result-card {
flex: 1;
padding: 24px;
background: #1a1d26;
border: 1px solid #2a2e3a;
border-radius: 14px;
text-align: center;
box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
.result-icon {
font-size: 28px;
margin-bottom: 8px;
}
.result-label {
font-size: 14px;
font-weight: 500;
color: #9aa0ac;
margin-bottom: 10px;
}
.result-value {
font-size: 32px;
font-weight: 700;
color: #60a5fa;
}
.result-outcome {
font-size: 32px;
font-weight: 700;
color: var(--result-accent);
}
</style>
""", unsafe_allow_html=True)

        # Result cards -- flush left, see note at top of file
        result_html = f"""
<div class="result-container">
<div class="result-card">
<div class="result-icon">💰</div>
<div class="result-label">Predicted Revenue</div>
<div class="result-value">₹{predicted_revenue:,.2f}</div>
</div>
<div class="result-card" style="--result-accent:{result_accent}; background:{result_bg};">
<div class="result-icon">{"✅" if predicted_label == "Profit" else "❌"}</div>
<div class="result-label">Predicted Outcome</div>
<div class="result-outcome">{predicted_label}</div>
</div>
</div>
"""
        st.markdown(result_html, unsafe_allow_html=True)

# --------------------------------------------------
# ANALYSIS PAGE
# --------------------------------------------------
elif st.session_state.page == "Analysis":

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.subheader("📶 Campaign Performance Analysis")

    def style_fig(fig):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
        )
        return fig

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["By Campaign Type", "By Channel", "By Segment","Top & Low Performers", "Relationships"]
    )

    with tab1:
        agg = clean_df.groupby("Campaign_Type", as_index=False)["ROI"].mean()
        fig1 = px.bar(agg, x="Campaign_Type", y="ROI", color="ROI", title="Average ROI by Campaign Type")
        fig1.update_layout(title_x=0.3,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
        st.plotly_chart(style_fig(fig1), use_container_width=True)

    with tab2:
        exploded = clean_df.assign(Channel=clean_df["Channel_Used"].astype(str).str.split(", ")).explode("Channel")
        agg2 = exploded.groupby("Channel", as_index=False).agg(
            Revenue=("Revenue", "sum"),
            Avg_ROI=("ROI", "mean"),
        )
        c1, c2 = st.columns(2)
        with c1:
            fig2 = px.bar(agg2, x="Channel", y="Revenue", title="Total Revenue by Channel")
            fig2.update_layout(title_x=0.3,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
            st.plotly_chart(style_fig(fig2), use_container_width=True)
        with c2:
            fig2b = px.bar(agg2, x="Channel", y="Avg_ROI", color="Avg_ROI", title="Average ROI by Channel")
            fig2b.update_layout(title_x=0.25,title_font=dict(size=30),
                          width=550,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
            st.plotly_chart(style_fig(fig2b), use_container_width=True)

    with tab3:
        fig3 = px.box(clean_df, x="Customer_Segment", y="Engagement_Score", title="Engagement Score by Customer Segment")
        fig3.update_layout(title_x=0.3,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
        st.plotly_chart(style_fig(fig3), use_container_width=True)

    with tab4:
        rank_metric = st.selectbox("Rank campaigns by", ["Revenue", "ROI"], index=0)
        top_n = st.slider("How many to show", 5, 20, 10)

        display_cols = [c for c in ["Campaign_ID", "Campaign_Type", "Channel_Used", "Brand", "Revenue", "ROI"]
                         if c in clean_df.columns]

        top_campaigns = clean_df.nlargest(top_n, rank_metric)[display_cols]
        low_campaigns = clean_df.nsmallest(top_n, rank_metric)[display_cols]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Top {top_n} campaigns by {rank_metric}**")
            st.dataframe(top_campaigns, use_container_width=True, hide_index=True)
        with c2:
            st.markdown(f"**Bottom {top_n} campaigns by {rank_metric}**")
            st.dataframe(low_campaigns, use_container_width=True, hide_index=True)

    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            fig4 = px.scatter(
                clean_df, x="Acquisition_Cost", y="Revenue", color="ROI",
                title="Spend vs Revenue (colored by ROI)", opacity=0.6,
            )
            fig4.update_layout(title_x=0.2,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
            st.plotly_chart(style_fig(fig4), use_container_width=True)
        with c2:
            fig5 = px.scatter(
                clean_df, x="Clicks", y="Revenue", color="ROI",
                title="Clicks vs Revenue (colored by ROI)", opacity=0.6,
            )
            fig5.update_layout(title_x=0.2,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
            st.plotly_chart(style_fig(fig5), use_container_width=True)

        numeric_cols = [c for c in
                        ["Impressions", "Clicks", "Leads", "Conversions", "Acquisition_Cost", "Revenue", "ROI", "Engagement_Score"]
                        if c in clean_df.columns]
        corr = clean_df[numeric_cols].corr()
        fig6 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation: Spend, Clicks, Revenue & ROI",
        )
        fig6.update_layout(title_x=0.3,title_font=dict(size=30),
                          width=500,     # increase width
                          height=550  ,    # increase height
                              hoverlabel=dict(
                              bgcolor="#5EFABE",
                              font_size=14,
                              font_color="black"))
        st.plotly_chart(style_fig(fig6), use_container_width=True)