import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NYC Taxi Demand Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    .insight-box {
        background: #fff8e6;
        border-left: 3px solid #f5a623;
        border-radius: 0 6px 6px 0;
        padding: 0.75rem 1rem;
        font-size: 13px;
        color: #555;
        margin-top: 0.5rem;
    }
    .data-note {
        background: #e8f4fd;
        border-left: 3px solid #2196F3;
        border-radius: 0 6px 6px 0;
        padding: 0.6rem 1rem;
        font-size: 13px;
        color: #444;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = r".\model\demand_forecast_lgbm.pkl"
pkg        = pickle.load(open(MODEL_PATH, "rb"))
model      = pkg["model"]
feature_cols = pkg["features"]


# --------------------------------------------------
# SPARK + DATA  (cached)
# --------------------------------------------------

@st.cache_resource
def get_spark():
    builder = (
        SparkSession.builder.appName("DemandIntelligence")
            .config("spark.sql.extensions",            "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.driver.memory",             "8g")
            .config("spark.executor.memory",           "8g")
            .config("spark.driver.cores",              "4")
            .config("spark.sql.shuffle.partitions",    "200")
            .config("spark.sql.adaptive.enabled",      "true")
            .config("spark.driver.maxResultSize",      "2g")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


@st.cache_data
def load_data():
    spark = get_spark()

    demand = (
        spark.read.format("delta")
        .load(r"..\..\04_storage_platinum\demand_series")
        .toPandas()
    )
    demand["datetime"] = (
        pd.to_datetime(demand["trip_date"]) +
        pd.to_timedelta(demand["pickup_hour"], unit="h")
    )
    demand = demand.sort_values(["pu_location_id", "datetime"]).reset_index(drop=True)

    dim_location = (
        spark.read.format("delta")
        .load(r"..\..\03_storage_gold\dimensions\dim_location")
        .toPandas()
    )

    return demand, dim_location


demand_series, dim_location = load_data()

demand_series = demand_series.merge(
    dim_location[["location_id", "zone", "borough"]],
    left_on="pu_location_id", right_on="location_id",
    how="left"
)

DATA_THROUGH       = demand_series["datetime"].max()
DATA_THROUGH_LABEL = DATA_THROUGH.strftime("%B %Y")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("NYC Taxi\nDemand Intelligence")
st.sidebar.markdown(
    f"<div class='data-note' style='margin-top:0.5rem'>"
    f"Historical data through <strong>{DATA_THROUGH_LABEL}</strong>.<br>"
    f"Seasonal patterns are stable across years."
    f"</div>",
    unsafe_allow_html=True
)

tab_choice = st.sidebar.radio(
    "View",
    ["Seasonal Patterns", "Location Hotspots", "Capacity Planning"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='section-label'>Zone filter</div>", unsafe_allow_html=True)

all_zones    = (
    demand_series[["pu_location_id", "zone"]]
    .drop_duplicates().sort_values("zone")
)
zone_options = dict(zip(all_zones["zone"], all_zones["pu_location_id"]))

selected_zone_name = st.sidebar.selectbox("Zone", list(zone_options.keys()))
selected_zone_id   = zone_options[selected_zone_name]


# --------------------------------------------------
# HELPER: autoregressive prediction
# --------------------------------------------------

def predict_demand(zone_id, datetimes, seed_df):
    """
    Autoregressively predict trip_count for a list of future datetimes.
    seed_df: real historical rows used for initial lag lookups.
    Returns: list of float predictions (same length as datetimes).
    """
    buf   = seed_df.copy().reset_index(drop=True)
    preds = []

    for dt in datetimes:
        lag_1h   = float(buf["trip_count"].iloc[-1])
        lag_24h  = float(buf["trip_count"].iloc[-24])  if len(buf) >= 24  else lag_1h
        lag_168h = float(buf["trip_count"].iloc[-168]) if len(buf) >= 168 else lag_1h

        # PySpark dayofweek: 1=Sun … 7=Sat
        dow_pandas = dt.dayofweek
        dow_spark  = (dow_pandas + 1) % 7 + 1

        row = {
            "pu_location_id": int(zone_id),
            "pickup_hour":    dt.hour,
            "day_of_week":    dow_spark,
            "month":          dt.month,
            "trip_year":      dt.year,
            "is_weekend":     1 if dow_pandas in [5, 6] else 0,
            "is_rush_hour":   1 if (7 <= dt.hour <= 10 or 16 <= dt.hour <= 19) else 0,
            "is_night":       1 if (dt.hour >= 22 or dt.hour <= 5) else 0,
            "lag_1h":         lag_1h,
            "lag_24h":        lag_24h,
            "lag_168h":       lag_168h,
            "avg_fare":       float(buf["avg_fare"].iloc[-24:].mean()),
            "avg_distance":   float(buf["avg_distance"].iloc[-24:].mean()),
            "avg_passengers": float(buf["avg_passengers"].iloc[-24:].mean()),
        }

        pred = float(np.clip(
            model.predict(pd.DataFrame([row])[feature_cols])[0], 0, None
        ))
        preds.append(pred)

        buf = pd.concat([buf, pd.DataFrame([{
            "trip_count":     pred,
            "avg_fare":       row["avg_fare"],
            "avg_distance":   row["avg_distance"],
            "avg_passengers": row["avg_passengers"],
        }])], ignore_index=True)

    return preds


# ================================================================
#  TAB A — SEASONAL PATTERNS
# ================================================================

if tab_choice == "Seasonal Patterns":

    st.title("Seasonal Demand Patterns")
    st.markdown(
        "<div class='data-note'>"
        "Seasonal cycles change slowly — last year's patterns are valid guides for "
        "this year's planning. Weekday/weekend splits, holiday spikes, and "
        "summer/winter swings are structural, not data-lag-sensitive."
        "</div>",
        unsafe_allow_html=True
    )

    zone_df = (
        demand_series[demand_series["pu_location_id"] == selected_zone_id]
        .sort_values("datetime").copy()
    )

    # ── A1: Hour-of-day profile ──────────────────────────────────
    st.subheader("Hour-of-day demand profile")
    st.markdown(
        "<div class='section-label'>Average trips per hour — weekday vs weekend</div>",
        unsafe_allow_html=True
    )

    hourly = (
        zone_df
        .groupby(["pickup_hour", "is_weekend"])["trip_count"]
        .mean().reset_index()
    )
    hourly["day_type"] = hourly["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    fig_hour = go.Figure()
    for day_type, color in [("Weekday", "#2196F3"), ("Weekend", "#FF9800")]:
        grp = hourly[hourly["day_type"] == day_type]
        fig_hour.add_trace(go.Scatter(
            x=grp["pickup_hour"], y=grp["trip_count"].round(1),
            mode="lines+markers", name=day_type,
            line=dict(color=color, width=2.5), marker=dict(size=5),
            hovertemplate="Hour %{x}:00 — %{y:.0f} trips<extra>" + day_type + "</extra>"
        ))

    fig_hour.update_layout(
        height=320, hovermode="x unified",
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Avg trips/hour",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    peak_wday  = hourly[hourly["day_type"] == "Weekday"].nlargest(1, "trip_count").iloc[0]
    peak_wend  = hourly[hourly["day_type"] == "Weekend"].nlargest(1, "trip_count").iloc[0]
    st.markdown(
        f"<div class='insight-box'>"
        f"<strong>Peak hours:</strong> Weekday peaks at "
        f"<strong>{int(peak_wday.pickup_hour)}:00</strong> ({peak_wday.trip_count:.0f} avg trips) — "
        f"Weekend peaks at <strong>{int(peak_wend.pickup_hour)}:00</strong> "
        f"({peak_wend.trip_count:.0f} avg trips)."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── A2: Typical week pattern ─────────────────────────────────
    st.subheader("Typical week demand pattern")
    st.markdown(
        "<div class='section-label'>"
        "Average hourly demand across a full week — based on all historical data"
        "</div>",
        unsafe_allow_html=True
    )

    zone_df["dow_num"]  = pd.to_datetime(zone_df["trip_date"]).dt.dayofweek  # 0=Mon
    zone_df["dow_name"] = pd.to_datetime(zone_df["trip_date"]).dt.strftime("%a")
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    typical_week = (
        zone_df
        .groupby(["dow_num", "dow_name", "pickup_hour"])["trip_count"]
        .mean().reset_index()
        .sort_values(["dow_num", "pickup_hour"])
    )
    typical_week["label"] = typical_week["dow_name"] + " " + typical_week["pickup_hour"].apply(lambda h: f"{h:02d}:00")

    fig_week = go.Figure()
    colors_week = {"Mon": "#1565C0", "Tue": "#1976D2", "Wed": "#1E88E5",
                "Thu": "#42A5F5", "Fri": "#2196F3", "Sat": "#FF9800", "Sun": "#F44336"}

    for dow in dow_order:
        grp = typical_week[typical_week["dow_name"] == dow]
        fig_week.add_trace(go.Scatter(
            x=grp["pickup_hour"], y=grp["trip_count"].round(1),
            mode="lines", name=dow,
            line=dict(color=colors_week[dow], width=1.8),
            hovertemplate=f"{dow} %{{x}}:00 — %{{y:.0f}} trips<extra></extra>"
        ))

    fig_week.update_layout(
        height=340, hovermode="x unified",
        xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=2),
        yaxis_title="Avg trips/hour",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_week, use_container_width=True)

    busiest = typical_week.nlargest(1, "trip_count").iloc[0]
    quietest = typical_week.nsmallest(1, "trip_count").iloc[0]
    st.markdown(
        f"<div class='insight-box'>"
        f"Busiest slot: <strong>{busiest.dow_name} {int(busiest.pickup_hour):02d}:00</strong> "
        f"({busiest.trip_count:.0f} avg trips/hr) — "
        f"Quietest: <strong>{quietest.dow_name} {int(quietest.pickup_hour):02d}:00</strong> "
        f"({quietest.trip_count:.0f} avg trips/hr). "
        f"This pattern repeats reliably and can be used directly for weekly scheduling."
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── A3: Monthly seasonality ──────────────────────────────────
    st.subheader("Monthly demand seasonality")

    monthly = (
        zone_df.groupby("month")["trip_count"].mean().reset_index()
    )
    monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
    avg_demand = monthly["trip_count"].mean()

    bar_colors = [
        "#E53935" if v >= avg_demand * 1.1
        else "#43A047" if v <= avg_demand * 0.9
        else "#90CAF9"
        for v in monthly["trip_count"]
    ]

    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=monthly["month_name"], y=monthly["trip_count"].round(1),
        marker_color=bar_colors,
        hovertemplate="%{x}: %{y:.0f} avg trips/hr<extra></extra>"
    ))
    fig_month.add_hline(
        y=avg_demand, line_dash="dot", line_color="gray",
        annotation_text=f"Annual avg: {avg_demand:.0f}",
        annotation_position="top right"
    )
    fig_month.update_layout(
        height=300, xaxis_title="Month", yaxis_title="Avg trips/hour",
        margin=dict(t=10, b=40), showlegend=False
    )
    st.plotly_chart(fig_month, use_container_width=True)

    peak_m   = monthly.nlargest(1, "trip_count").iloc[0]
    trough_m = monthly.nsmallest(1, "trip_count").iloc[0]
    swing    = (peak_m.trip_count / trough_m.trip_count - 1) * 100
    st.markdown(
        f"<div class='insight-box'>"
        f"Demand peaks in <strong>{peak_m.month_name}</strong> "
        f"({peak_m.trip_count:.0f} trips/hr) and dips in "
        f"<strong>{trough_m.month_name}</strong> ({trough_m.trip_count:.0f} trips/hr) — "
        f"a <strong>{swing:.0f}% seasonal swing</strong>. "
        f"Use this to time driver recruitment and incentive campaigns."
        f"</div>",
        unsafe_allow_html=True
    )

# ================================================================
#  TAB B — LOCATION HOTSPOTS
# ================================================================

elif tab_choice == "Location Hotspots":

    st.title("Location Demand Hotspots")
    st.markdown(
        "<div class='data-note'>"
        "High-demand pickup zones are structurally stable — airports, business districts, "
        "and entertainment areas hold their ranking year over year. "
        "This drives fleet positioning, driver incentive zones, and surge pricing strategy."
        "</div>",
        unsafe_allow_html=True
    )

    boro_colors = {
        "Manhattan":    "#2196F3",
        "Brooklyn":     "#4CAF50",
        "Queens":       "#FF9800",
        "Bronx":        "#9C27B0",
        "Staten Island":"#F44336",
        "EWR":          "#607D8B",
    }

    # ── B1: Top zones bar chart ──────────────────────────────────
    st.subheader("Top 20 pickup zones by average hourly demand")

    zone_totals = (
        demand_series
        .groupby(["pu_location_id", "zone", "borough"])["trip_count"]
        .mean().reset_index()
        .rename(columns={"trip_count": "avg_hourly_demand"})
        .sort_values("avg_hourly_demand", ascending=False)
        .head(20)
    )

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Bar(
        x=zone_totals["avg_hourly_demand"].round(1),
        y=zone_totals["zone"],
        orientation="h",
        marker_color=[boro_colors.get(b, "#90A4AE") for b in zone_totals["borough"]],
        hovertemplate="<b>%{y}</b><br>%{x:.0f} avg trips/hr<extra></extra>"
    ))
    for boro, color in boro_colors.items():
        if boro in zone_totals["borough"].values:
            fig_zones.add_trace(go.Bar(
                x=[None], y=[None], marker_color=color,
                name=boro, showlegend=True
            ))

    fig_zones.update_layout(
        height=560, xaxis_title="Avg trips per hour",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(t=10, l=200, b=40), barmode="overlay"
    )
    st.plotly_chart(fig_zones, use_container_width=True)

    st.markdown("---")

    # ── B2: Heatmap for selected zone ───────────────────────────
    st.subheader(f"Demand heatmap — {selected_zone_name}")
    st.markdown(
        "<div class='section-label'>Average trips by hour of day and day of week</div>",
        unsafe_allow_html=True
    )

    zone_df = demand_series[demand_series["pu_location_id"] == selected_zone_id].copy()
    zone_df["dow_name"] = pd.to_datetime(zone_df["trip_date"]).dt.strftime("%a")
    dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    heatmap_data = (
        zone_df.groupby(["pickup_hour", "dow_name"])["trip_count"]
        .mean().reset_index()
        .pivot(index="pickup_hour", columns="dow_name", values="trip_count")
        .reindex(columns=dow_order)
    )

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data.values.round(1),
        x=heatmap_data.columns.tolist(),
        y=[f"{h:02d}:00" for h in heatmap_data.index],
        colorscale="Blues",
        hovertemplate="<b>%{x} %{y}</b><br>%{z:.0f} avg trips<extra></extra>",
        colorbar=dict(title="Trips/hr", thickness=12)
    ))
    fig_heat.update_layout(
        height=480, xaxis_title="Day of week",
        yaxis=dict(title="Hour", autorange="reversed"),
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── B3: Borough comparison ───────────────────────────────────
    st.subheader("Borough demand comparison")

    borough_df = (
        demand_series.groupby("borough")["trip_count"]
        .mean().reset_index()
        .rename(columns={"trip_count": "avg_hourly"})
        .sort_values("avg_hourly", ascending=False)
    )

    fig_boro = px.bar(
        borough_df, x="borough", y="avg_hourly",
        color="borough", color_discrete_map=boro_colors,
        labels={"avg_hourly": "Avg trips/hour", "borough": "Borough"}
    )
    fig_boro.update_layout(
        height=300, showlegend=False, margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_boro, use_container_width=True)

    top_boro = borough_df.iloc[0]
    st.markdown(
        f"<div class='insight-box'>"
        f"<strong>{top_boro['borough']}</strong> leads with "
        f"<strong>{top_boro['avg_hourly']:.0f} avg trips/hour</strong>. "
        f"Use the heatmap above to identify the specific hours in "
        f"<strong>{selected_zone_name}</strong> that are consistently under- or over-served."
        f"</div>",
        unsafe_allow_html=True
    )


# ================================================================
#  TAB C — CAPACITY PLANNING
# ================================================================

elif tab_choice == "Capacity Planning":

    st.title("Long-term Capacity Planning")
    st.markdown(
        "<div class='data-note'>"
        "A 3-month data lag has no impact on capacity planning — driver supply, "
        "infrastructure, and pricing decisions operate on monthly or quarterly horizons. "
        "The model projects 90 days forward using learned seasonal structure."
        "</div>",
        unsafe_allow_html=True
    )

    zone_df = (
        demand_series[demand_series["pu_location_id"] == selected_zone_id]
        .sort_values("datetime").copy()
    )

    # ── C1: KPIs ────────────────────────────────────────────────
    st.subheader("Zone performance summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg demand",    f"{zone_df['trip_count'].mean():.0f} trips/hr")
    c2.metric("P95 peak",      f"{zone_df['trip_count'].quantile(0.95):.0f} trips/hr",
              help="95th percentile — plan capacity for this, not the average")
    c3.metric("Avg fare",      f"${zone_df['avg_fare'].mean():.2f}")
    c4.metric("Avg distance",  f"{zone_df['avg_distance'].mean():.1f} mi")

    st.markdown("---")

    # ── C2: Year-over-year monthly trend ────────────────────────
    st.subheader("Year-over-year monthly demand")

    yoy = (
        zone_df
        .assign(
            year=zone_df["datetime"].dt.year,
            month_num=zone_df["datetime"].dt.month,
            month_name=zone_df["datetime"].dt.strftime("%b")
        )
        .groupby(["year", "month_num", "month_name"])["trip_count"]
        .mean().reset_index()
        .sort_values(["year", "month_num"])
    )

    year_colors = {2023: "#90CAF9", 2024: "#2196F3", 2025: "#0D47A1"}
    fig_yoy = go.Figure()

    for yr, grp in yoy.groupby("year"):
        fig_yoy.add_trace(go.Scatter(
            x=grp["month_name"], y=grp["trip_count"].round(1),
            mode="lines+markers", name=str(yr),
            line=dict(color=year_colors.get(yr, "#888"), width=2),
            hovertemplate=f"{yr} %{{x}}: %{{y:.0f}} trips/hr<extra></extra>"
        ))

    fig_yoy.update_layout(
        height=320, hovermode="x unified",
        xaxis_title="Month", yaxis_title="Avg trips/hour",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

    st.markdown("---")

    # ── C3: 90-day projection ────────────────────────────────────
    st.subheader("Model-projected demand — next 90 days")
    st.markdown(
        "<div class='section-label'>"
        "Weekly averages shown — suitable for driver allocation and infrastructure planning"
        "</div>",
        unsafe_allow_html=True
    )

    seed_df   = zone_df.tail(500).copy()
    proj_start = DATA_THROUGH + pd.Timedelta(hours=1)
    proj_dts   = [proj_start + pd.Timedelta(hours=h) for h in range(90 * 24)]

    with st.spinner("Running 90-day projection..."):
        proj_vals = predict_demand(selected_zone_id, proj_dts, seed_df)

    proj_df = pd.DataFrame({"datetime": proj_dts, "trip_count": proj_vals})
    proj_df["week"]  = proj_df["datetime"].dt.to_period("W").apply(lambda r: r.start_time)
    proj_df["month"] = proj_df["datetime"].dt.to_period("M").apply(lambda r: r.start_time)

    weekly_proj = proj_df.groupby("week")["trip_count"].mean().reset_index()

    hist_monthly = (
        zone_df
        .assign(month=zone_df["datetime"].dt.to_period("M").apply(lambda r: r.start_time))
        .groupby("month")["trip_count"].mean().reset_index()
        .tail(6)
    )

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Bar(
        x=hist_monthly["month"], y=hist_monthly["trip_count"].round(1),
        name="Historical monthly avg (last 6 mo)",
        marker_color="#90CAF9",
        hovertemplate="%{x|%b %Y}: %{y:.0f} trips/hr<extra>Historical</extra>"
    ))
    fig_proj.add_trace(go.Scatter(
        x=weekly_proj["week"], y=weekly_proj["trip_count"].round(1),
        mode="lines", name="Projected weekly avg",
        line=dict(color="#E53935", width=2.5),
        hovertemplate="%{x|%d %b %Y}: %{y:.0f} trips/hr<extra>Projected</extra>"
    ))

    vline_x2 = proj_start.strftime("%Y-%m-%d %H:%M:%S")
    fig_proj.add_shape(
        type="line", x0=vline_x2, x1=vline_x2, y0=0, y1=1,
        xref="x", yref="paper", line=dict(color="gray", dash="dot", width=1.2)
    )
    fig_proj.add_annotation(
        x=vline_x2, y=0.97, xref="x", yref="paper",
        text="Projection start", showarrow=False,
        xanchor="left", font=dict(size=10, color="gray")
    )
    fig_proj.update_layout(
        height=360, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Avg trips/hour",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown("---")

    # ── C4: Driver requirement estimator ────────────────────────
    st.subheader("Driver requirement estimator")

    TRIPS_PER_DRIVER_HOUR = st.slider(
        "Trips completed per driver per hour", 1.0, 4.0, 2.0, 0.5,
        help="Adjust based on your observed average trip duration and deadhead time"
    )

    monthly_proj = (
        proj_df.groupby("month")["trip_count"].mean().reset_index()
        .rename(columns={"trip_count": "avg_hourly_demand"})
    )
    monthly_proj["avg_drivers"]  = (monthly_proj["avg_hourly_demand"] / TRIPS_PER_DRIVER_HOUR).round(0).astype(int)
    monthly_proj["peak_drivers"] = ((monthly_proj["avg_hourly_demand"] * 1.5) / TRIPS_PER_DRIVER_HOUR).round(0).astype(int)
    monthly_proj["month_label"]  = monthly_proj["month"].dt.strftime("%b %Y")

    fig_drv = go.Figure()
    fig_drv.add_trace(go.Bar(
        x=monthly_proj["month_label"], y=monthly_proj["avg_drivers"],
        name="Avg demand coverage",
        marker_color="#42A5F5",
        hovertemplate="%{x}: %{y} drivers<extra>Avg</extra>"
    ))
    fig_drv.add_trace(go.Bar(
        x=monthly_proj["month_label"], y=monthly_proj["peak_drivers"],
        name="Peak coverage (1.5× avg)",
        marker_color="#EF5350",
        hovertemplate="%{x}: %{y} drivers<extra>Peak</extra>"
    ))
    fig_drv.update_layout(
        height=300, barmode="group",
        xaxis_title="Month", yaxis_title="Concurrent drivers needed",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(t=10, b=40)
    )
    st.plotly_chart(fig_drv, use_container_width=True)

    st.markdown(
        f"<div class='insight-box'>"
        f"At <strong>{TRIPS_PER_DRIVER_HOUR:.1f} trips/driver/hour</strong>, "
        f"<strong>{selected_zone_name}</strong> needs approximately "
        f"<strong>{int(monthly_proj['avg_drivers'].mean())} concurrent drivers</strong> on average "
        f"and up to <strong>{int(monthly_proj['peak_drivers'].max())} during peak periods</strong>. "
        f"Adjust the slider to match your observed utilisation rate."
        f"</div>",
        unsafe_allow_html=True
    )