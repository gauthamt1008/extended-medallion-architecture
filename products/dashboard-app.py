import streamlit as st
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NYC Taxi Analytics",
    layout="wide"
)

st.title("NYC Taxi Analytics Dashboard")

# --------------------------------------------------
# SPARK SESSION
# --------------------------------------------------

@st.cache_resource
def get_spark():

    builder = (
        SparkSession.builder
        .appName("Dashboard")

        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.cores", "4")

        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")

        .config("spark.driver.maxResultSize", "2g")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    return spark


spark = get_spark()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    fact_year = spark.read.format("delta").load(
        r"..\03_storage_gold\facts\fact_trip_yearly_summary"
    ).toPandas()

    fact_payment = spark.read.format("delta").load(
        r"..\03_storage_gold\facts\fact_payment_summary"
    ).toPandas()

    fact_distance = spark.read.format("delta").load(
        r"..\03_storage_gold\facts\fact_distance_summary"
    ).toPandas()

    fact_vendor = spark.read.format("delta").load(
        r"..\03_storage_gold\facts\fact_vendor_summary"
    ).toPandas()

    fact_time = spark.read.format("delta").load(
        r"..\03_storage_gold\facts\fact_trip_time_summary"
    ).toPandas()

    fact_location = spark.read.format("delta").load(
        r"..\03_storage_gold\facts\fact_location_summary"
    ).toPandas()

    dim_payment = spark.read.format("delta").load(
        r"..\03_storage_gold\dimensions\dim_payment_type"
    ).toPandas()

    dim_vendor = spark.read.format("delta").load(
        r"..\03_storage_gold\dimensions\dim_vendor"
    ).toPandas()

    dim_zone = spark.read.format("delta").load(
        r"..\03_storage_gold\dimensions\dim_location"
    ).toPandas()

    return (
        fact_year,
        fact_payment,
        fact_distance,
        fact_vendor,
        fact_time,
        fact_location,
        dim_payment,
        dim_vendor,
        dim_zone
    )


(
    fact_year,
    fact_payment,
    fact_distance,
    fact_vendor,
    fact_time,
    fact_location,
    dim_payment,
    dim_vendor,
    dim_zone
) = load_data()

# --------------------------------------------------
# JOIN DIMENSIONS
# --------------------------------------------------

fact_payment = fact_payment.merge(
    dim_payment,
    on="payment_type",
    how="left"
)

fact_vendor = fact_vendor.merge(
    dim_vendor,
    on="vendor_id",
    how="left"
)

location_df = fact_location.merge(

    dim_zone,

    left_on="pu_location_id",

    right_on="location_id",

    how="left"
)

# --------------------------------------------------
# FILTER
# --------------------------------------------------

st.sidebar.header("Filters")

years = sorted(fact_year["trip_year"].unique())

selected_year = st.sidebar.multiselect(

    "Select Year",

    years,

    default=years
)

fact_year = fact_year[
    fact_year["trip_year"].isin(selected_year)
]

fact_payment = fact_payment[
    fact_payment["trip_year"].isin(selected_year)
]

fact_vendor = fact_vendor[
    fact_vendor["trip_year"].isin(selected_year)
]

fact_time = fact_time[
    fact_time["trip_year"].isin(selected_year)
]

fact_distance = fact_distance[
    fact_distance["trip_year"].isin(selected_year)
]

location_df = location_df[
    location_df["trip_year"].isin(selected_year)
]

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

st.header("Overall Performance")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Trips",

    f"{int(fact_year['total_trips'].sum()):,}"
)

c2.metric(
    "Total Revenue",

    f"${round(fact_year['total_revenue'].sum(),2):,}"
)

c3.metric(
    "Avg Trip Value",

    round(fact_year["avg_trip_value"].mean(), 2)
)

# --------------------------------------------------
# REVENUE TREND
# --------------------------------------------------

st.header("Revenue Trend")

fact_year = fact_year.sort_values("trip_year")

fig = px.line(

    fact_year,

    x="trip_year",

    y="total_revenue",

    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PAYMENT ANALYSIS
# --------------------------------------------------

st.header("Payment Behaviour")

fig = px.bar(

    fact_payment,

    x="payment_name",

    y="total_trips",

    color="trip_year",

    barmode="group"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# DISTANCE ANALYSIS
# --------------------------------------------------

st.header("Trip Distance Pattern")

distance_summary = (

    fact_distance

    .groupby("distance_bucket")["total_trips"]

    .sum()

    .reset_index()
)

fig = px.pie(

    distance_summary,

    names="distance_bucket",

    values="total_trips"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# VENDOR PERFORMANCE
# --------------------------------------------------

st.header("Vendor Performance")

fig = px.bar(

    fact_vendor,

    x="vendor_name",

    y="total_revenue",

    color="trip_year",

    barmode="group"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# TIME ANALYSIS
# --------------------------------------------------

st.header("Peak Demand Hours")

hourly = (

    fact_time

    .groupby("pickup_hour")["total_trips"]

    .sum()

    .reset_index()

    .sort_values("pickup_hour")
)

fig = px.line(

    hourly,

    x="pickup_hour",

    y="total_trips",

    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# LOCATION ANALYSIS
# --------------------------------------------------

st.header("Location Intelligence")

top_zones = (

    location_df

    .groupby("zone")[["total_revenue","total_trips"]]

    .sum()

    .reset_index()

    .sort_values("total_revenue", ascending=False)

    .head(10)
)

fig = px.bar(

    top_zones,

    x="zone",

    y="total_revenue",

    hover_data=["total_trips"]
)

st.plotly_chart(fig, use_container_width=True)

borough_summary = (

    location_df

    .groupby("borough")["total_trips"]

    .sum()

    .reset_index()

    .sort_values("total_trips", ascending=False)
)

fig = px.bar(

    borough_summary,

    x="borough",

    y="total_trips"
)

st.plotly_chart(fig, use_container_width=True)

premium_zones = (

    location_df

    .groupby("zone")["avg_trip_value"]

    .mean()

    .reset_index()

    .sort_values("avg_trip_value", ascending=False)

    .head(10)
)

fig = px.bar(

    premium_zones,

    x="zone",

    y="avg_trip_value"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    ---
    <div style="text-align:right">
    - By Gautham T.<br>
    Data Source: NYC TLC Trip Record Data
    </div>
    """,
    unsafe_allow_html=True
)