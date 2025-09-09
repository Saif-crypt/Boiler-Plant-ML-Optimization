import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Boiler Efficiency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load data ---
df_data = pd.read_csv('boiler_ml_anomalies.csv', parse_dates=['Date'])
df_predictions = pd.read_csv('efficiency_predictions.csv', parse_dates=['Date'])
df_feat_imp = pd.read_csv('feature_importance.csv')

st.title("🔥 Boiler Efficiency Monitoring Dashboard")

# Sidebar - Date filter and anomaly toggle
st.sidebar.header("Filters & Options")
min_date, max_date = df_data['Date'].min(), df_data['Date'].max()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])
show_anomalies_only = st.sidebar.checkbox("Show Anomalies Only", value=False)

# Filter data
filtered_data = df_data[
    (df_data['Date'] >= pd.to_datetime(start_date)) & (df_data['Date'] <= pd.to_datetime(end_date))
]
if show_anomalies_only:
    filtered_data = filtered_data[filtered_data['anomaly'] == -1]

filtered_pred = df_predictions[
    (df_predictions['Date'] >= pd.to_datetime(start_date)) & (df_predictions['Date'] <= pd.to_datetime(end_date))
]

# Layout: KPIs at top
st.header("Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Efficiency", f"{filtered_data['Efficiency'].mean():.2f} %")
anomaly_count = filtered_data[filtered_data['anomaly'] == -1].shape[0]
col2.metric("Anomaly Days Detected", str(anomaly_count))
col3.metric("Data Points", str(filtered_data.shape[0]))

st.markdown("---")

# Efficiency Actual vs Predicted line chart with smoother style
st.header("Efficiency: Actual vs Predicted")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(filtered_pred['Date'], filtered_pred['Actual_Efficiency'], marker='o', linestyle='-', color='#1f77b4', label='Actual')
ax.plot(filtered_pred['Date'], filtered_pred['Predicted_Efficiency'], marker='x', linestyle='--', color='#ff7f0e', label='Predicted')
ax.set_xlabel("Date")
ax.set_ylabel("Efficiency (%)")
ax.set_title("Boiler Efficiency Over Time")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

st.markdown("---")

# Feature importance sorted, horizontal bar plot with distinct color
st.header("Top Feature Importance")
df_feat_imp_sorted = df_feat_imp.sort_values(by='Importance', ascending=True)
fig2, ax2 = plt.subplots(figsize=(10, 8))
bars = ax2.barh(df_feat_imp_sorted['Feature'], df_feat_imp_sorted['Importance'], color='#2ca02c')
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance for Boiler Efficiency Prediction')
ax2.grid(axis='x', linestyle='--', alpha=0.4)
st.pyplot(fig2)

st.markdown("---")

# Data table with anomaly indicator column for clarity
st.header("Detailed Data Table with Anomalies")

# Optionally, add a new anomaly status column for clearer display
filtered_data['Anomaly Status'] = filtered_data['anomaly'].map({-1: 'Anomaly', 1: 'Normal'})

st.dataframe(filtered_data, height=400)

# CSV Download
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_boiler_data.csv",
    mime="text/csv"
)

st.markdown("""
---
### Operational Recommendations  
- Investigate highlighted anomaly days for root causes.  
- Use prediction trends to plan maintenance schedules.  
- Optimize key features identified by importance analysis for improved efficiency.
""")
