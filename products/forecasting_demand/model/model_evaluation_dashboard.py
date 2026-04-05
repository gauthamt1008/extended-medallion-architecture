import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import matplotlib.pyplot as plt

# ---------------------------
# LOAD DATA FROM PLATINUM
# ---------------------------

builder = (
    SparkSession.builder
        .appName("ForecastDashboard")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

preds = (
    spark.read.format("delta")
    .load(r".\dashboard_data\demand_predictions")
    .toPandas()
)

metrics = (
    spark.read.format("delta")
    .load(r".\dashboard_data\forecast_metrics")
    .toPandas()
)

importance = (
    spark.read.format("delta")
    .load(r".\dashboard_data\forecast_feature_importance")
    .toPandas()
)

preds["datetime"] = pd.to_datetime(preds["datetime"])


# ---------------------------
# DASHBOARD
# ---------------------------

st.title("Taxi Demand Forecast")

st.sidebar.header("Controls")

zone_ids = sorted(preds["pu_location_id"].unique())

selected_zone = st.sidebar.selectbox(
    "Select zone",
    zone_ids
)

zone_data = preds[preds["pu_location_id"] == selected_zone]


# ---------------------------
# Forecast chart
# ---------------------------

st.subheader("Forecast vs Actual")

fig, ax = plt.subplots(figsize=(12,4))

ax.plot(
    zone_data["datetime"],
    zone_data["trip_count"],
    label="Actual"
)

ax.plot(
    zone_data["datetime"],
    zone_data["prediction"],
    label="Predicted"
)

ax.set_ylabel("Trips per hour")
ax.legend()

st.pyplot(fig)


# ---------------------------
# Metrics table
# ---------------------------

st.subheader("Model performance")

st.dataframe(metrics.sort_values("mae"))


# ---------------------------
# Feature importance
# ---------------------------

st.subheader("Feature importance")

fig2, ax2 = plt.subplots()

ax2.barh(
    importance["feature"],
    importance["importance"]
)

ax2.invert_yaxis()

st.pyplot(fig2)