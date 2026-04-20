import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="NYC Taxi Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }

    /* ── sidebar ── */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.3px;
        margin-bottom: 2px;
    }
    .sidebar-subtitle {
        font-size: 12px;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    .sidebar-section {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9ca3af;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #e5e7eb;
    }

    /* ── year toggle pills ── */
    /* Hide the native Streamlit checkbox entirely */
    div[data-testid="stCheckbox"] { display: none !important; }

    .year-pill {
        display: block;
        width: 100%;
        padding: 7px 0;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        text-align: center;
        cursor: pointer;
        margin-bottom: 6px;
        transition: background 0.15s;
        border: none;
        outline: none;
    }
    .year-pill-on  { background: #2196F3; color: #fff; }
    .year-pill-off { background: #f1f5f9; color: #64748b; }

    /* ── content ── */
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
        margin-bottom: 0.5rem;
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
# SPARK + DATA
# --------------------------------------------------

@st.cache_resource
def get_spark():
    builder = (
        SparkSession.builder.appName("HistoricalDashboard")
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


import os

@st.cache_data
def load_gold():
    spark = get_spark()
    
    # 1. Get the absolute path of the directory this specific script is sitting in
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Navigate up two levels ("..", "..") and into the gold folder
    # This creates a rock-solid absolute path no matter where the orchestrator is run
    gold = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "03_storage_gold"))

    def read(path):
        return spark.read.format("delta").load(path).toPandas()

    # 3. Use os.path.join instead of hardcoded slashes for bulletproof cross-platform paths
    return {
        "fact_year":     read(os.path.join(gold, "facts", "fact_trip_yearly_summary")),
        "fact_payment":  read(os.path.join(gold, "facts", "fact_payment_summary")),
        "fact_distance": read(os.path.join(gold, "facts", "fact_distance_summary")),
        "fact_vendor":   read(os.path.join(gold, "facts", "fact_vendor_summary")),
        "fact_time":     read(os.path.join(gold, "facts", "fact_trip_time_summary")),
        "fact_location": read(os.path.join(gold, "facts", "fact_location_summary")),
        "dim_location":  read(os.path.join(gold, "dimensions", "dim_location")),
        "dim_payment":   read(os.path.join(gold, "dimensions", "dim_payment_type")),
        "dim_vendor":    read(os.path.join(gold, "dimensions", "dim_vendor")),
    }

gold = load_gold()

fact_payment  = gold["fact_payment"].merge(gold["dim_payment"],  on="payment_type", how="left")
fact_vendor   = gold["fact_vendor"].merge(gold["dim_vendor"],    on="vendor_id",    how="left")
fact_location = gold["fact_location"].merge(
    gold["dim_location"], left_on="pu_location_id", right_on="location_id", how="left"
)

BORO_COLORS = {
    "Manhattan":    "#2196F3",
    "Brooklyn":     "#4CAF50",
    "Queens":       "#FF9800",
    "Bronx":        "#9C27B0",
    "Staten Island":"#F44336",
    "EWR":          "#607D8B",
}

all_years = sorted(gold["fact_year"]["trip_year"].unique())


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("<div class='sidebar-title'>NYC Taxi</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle'>Historical Analytics</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Data source</div>", unsafe_allow_html=True)
    st.caption("NYC TLC Trip Record Data\nGold layer aggregations")

    # ── Year filter — single clickable pill per year ──────────────
    # Uses st.checkbox hidden via CSS; pill button is a label for it.
    # Clicking the pill visually toggles via JS class swap, but
    # the actual state is tracked through Streamlit session_state.

    st.markdown("<div class='sidebar-section'>Year filter</div>", unsafe_allow_html=True)

    if "selected_years" not in st.session_state:
        st.session_state.selected_years = list(all_years)

    for yr in all_years:
        is_on = yr in st.session_state.selected_years
        pill_class = "year-pill year-pill-on" if is_on else "year-pill year-pill-off"
        label = f"{yr}  ✓" if is_on else str(yr)

        if st.button(label, key=f"pill_{yr}", use_container_width=True):
            if is_on:
                # Don't allow deselecting all years
                if len(st.session_state.selected_years) > 1:
                    st.session_state.selected_years.remove(yr)
            else:
                st.session_state.selected_years.append(yr)
                st.session_state.selected_years.sort()
            st.rerun()

        # Inject pill styling over the button via markdown
        st.markdown(
            f"<style>"
            f"div[data-testid='stButton'][key='pill_{yr}'] > button {{"
            f"  background: {'#2196F3' if is_on else '#f1f5f9'} !important;"
            f"  color: {'white' if is_on else '#64748b'} !important;"
            f"  border: none !important;"
            f"  border-radius: 8px !important;"
            f"  font-weight: 600 !important;"
            f"  font-size: 14px !important;"
            f"}}"
            f"</style>",
            unsafe_allow_html=True
        )

    selected_years = st.session_state.selected_years

    st.markdown("<div class='sidebar-section'>Sections</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#64748b; line-height:2.2;">
    📈 &nbsp;Revenue trend<br>
    💳 &nbsp;Payment behaviour<br>
    🚕 &nbsp;Vendor performance<br>
    📏 &nbsp;Distance patterns<br>
    🕐 &nbsp;Peak demand hours<br>
    📍 &nbsp;Location intelligence
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# YEAR FILTER HELPER
# --------------------------------------------------

def yf(df, col="trip_year"):
    return df[df[col].isin(selected_years)]


fy = yf(gold["fact_year"])
fp = yf(fact_payment)
fv = yf(fact_vendor)
ft = yf(gold["fact_time"])
fd = yf(gold["fact_distance"]).copy()
fl = yf(fact_location)


# --------------------------------------------------
# KPI HELPER  (reused at top of every section)
# --------------------------------------------------

def render_kpis():
    total_trips   = int(fy["total_trips"].sum())
    total_revenue = fy["total_revenue"].sum()
    avg_value     = fy["avg_trip_value"].mean()
    avg_dist      = fy["avg_distance"].mean()
    avg_fpm       = fy["avg_fare_per_mile"].mean()

    # Row 1 — totals
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Total trips",   f"{total_trips:,}")
    r1c2.metric("Total revenue", f"${total_revenue:,.0f}")
    r1c3.metric("Years selected", ", ".join(str(y) for y in sorted(selected_years)))

    # Row 2 — averages
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Avg trip value",  f"${avg_value:.2f}")
    r2c2.metric("Avg distance",    f"{avg_dist:.2f} mi")
    r2c3.metric("Avg fare / mile", f"${avg_fpm:.2f}")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("NYC Taxi — Historical Analytics")
st.markdown(
    "<div class='data-note'>"
    "Business intelligence from the Gold layer — aggregated facts across all trips. "
    "Use the year filter in the sidebar to compare periods."
    "</div>", unsafe_allow_html=True
)

render_kpis()

st.markdown("---")


# --------------------------------------------------
# 1. REVENUE TREND
# --------------------------------------------------

st.subheader("📈 Revenue trend")

fy_sorted = fy.sort_values("trip_year")

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.markdown("<div class='section-label'>Total revenue by year</div>", unsafe_allow_html=True)
    fig_rev = go.Figure(go.Bar(
        x=fy_sorted["trip_year"].astype(str),
        y=fy_sorted["total_revenue"].round(0),
        marker_color="#2196F3",
        text=["$" + f"{v/1e6:.1f}M" for v in fy_sorted["total_revenue"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    fig_rev.update_layout(
        height=300,
        yaxis=dict(title="Total revenue ($)", tickformat="$,.0f"),
        xaxis_title="Year",
        margin=dict(t=30, b=40), showlegend=False
    )
    st.plotly_chart(fig_rev, use_container_width=True)

with col_r2:
    st.markdown("<div class='section-label'>Total trips by year</div>", unsafe_allow_html=True)
    fig_trips = go.Figure(go.Bar(
        x=fy_sorted["trip_year"].astype(str),
        y=fy_sorted["total_trips"],
        marker_color="#FF9800",
        text=[f"{v/1e6:.1f}M" for v in fy_sorted["total_trips"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} trips<extra></extra>"
    ))
    fig_trips.update_layout(
        height=300,
        yaxis_title="Total trips",
        xaxis_title="Year",
        margin=dict(t=30, b=40), showlegend=False
    )
    st.plotly_chart(fig_trips, use_container_width=True)

if len(fy_sorted) >= 2:
    rev_delta  = (fy_sorted["total_revenue"].iloc[-1] / fy_sorted["total_revenue"].iloc[-2] - 1) * 100
    trip_delta = (fy_sorted["total_trips"].iloc[-1]   / fy_sorted["total_trips"].iloc[-2]   - 1) * 100
    st.markdown(
        f"<div class='insight-box'>"
        f"Revenue is <strong>{'up' if rev_delta>=0 else 'down'} {abs(rev_delta):.1f}%</strong> and "
        f"trips are <strong>{'up' if trip_delta>=0 else 'down'} {abs(trip_delta):.1f}%</strong> "
        f"from {int(fy_sorted['trip_year'].iloc[-2])} to {int(fy_sorted['trip_year'].iloc[-1])}. "
        f"When both move together, growth is demand-driven. "
        f"A gap between them signals a fare or distance shift."
        f"</div>", unsafe_allow_html=True
    )

st.markdown("---")


# --------------------------------------------------
# 2. PAYMENT BEHAVIOUR
# --------------------------------------------------

st.subheader("💳 Payment behaviour")

payment_mix = (
    fp.groupby("payment_name")["total_trips"]
    .sum().reset_index().sort_values("total_trips", ascending=False)
)
payment_val = (
    fp.groupby("payment_name")["avg_trip_value"]
    .mean().reset_index().sort_values("avg_trip_value", ascending=False)
)

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.markdown("<div class='section-label'>Trip share by payment type</div>", unsafe_allow_html=True)
    fig_pmix = go.Figure(go.Bar(
        x=payment_mix["payment_name"],
        y=payment_mix["total_trips"],
        marker_color="#2196F3",
        text=[f"{v/payment_mix['total_trips'].sum()*100:.1f}%" for v in payment_mix["total_trips"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} trips<extra></extra>"
    ))
    fig_pmix.update_layout(
        height=300, yaxis_title="Total trips",
        xaxis_title="Payment type",
        margin=dict(t=30, b=60), showlegend=False
    )
    st.plotly_chart(fig_pmix, use_container_width=True)

with col_p2:
    st.markdown("<div class='section-label'>Avg trip value by payment type</div>", unsafe_allow_html=True)
    fig_pval = go.Figure(go.Bar(
        x=payment_val["payment_name"],
        y=payment_val["avg_trip_value"].round(2),
        marker_color="#FF9800",
        text=["$" + f"{v:.2f}" for v in payment_val["avg_trip_value"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.2f}<extra></extra>"
    ))
    fig_pval.update_layout(
        height=300, yaxis_title="Avg trip value ($)",
        xaxis_title="Payment type",
        margin=dict(t=30, b=60), showlegend=False
    )
    st.plotly_chart(fig_pval, use_container_width=True)

top_pay   = payment_mix.iloc[0]
top_share = top_pay.total_trips / payment_mix["total_trips"].sum() * 100
st.markdown(
    f"<div class='insight-box'>"
    f"<strong>{top_pay.payment_name}</strong> accounts for "
    f"<strong>{top_share:.0f}%</strong> of all trips. "
    f"Credit card trips carry a higher avg value than cash — driven by "
    f"longer distances and digital tipping behaviour."
    f"</div>", unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------------------------
# 3. VENDOR PERFORMANCE
# --------------------------------------------------

st.subheader("🚕 Vendor performance")

vendor_summary = (
    fv.groupby("vendor_name")
    .agg(
        total_trips      =("total_trips",       "sum"),
        total_revenue    =("total_revenue",     "sum"),
        avg_fare_per_mile=("avg_fare_per_mile", "mean"),
        avg_passengers   =("avg_passengers",    "mean"),
    )
    .reset_index()
    .sort_values("total_revenue", ascending=False)
)

col_v1, col_v2, col_v3 = st.columns(3)

with col_v1:
    st.markdown("<div class='section-label'>Total revenue</div>", unsafe_allow_html=True)
    fig_vrev = go.Figure(go.Bar(
        x=vendor_summary["vendor_name"],
        y=vendor_summary["total_revenue"].round(0),
        marker_color="#2196F3",
        text=["$" + f"{v/1e6:.1f}M" for v in vendor_summary["total_revenue"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    fig_vrev.update_layout(
        height=280, yaxis_title="Revenue ($)",
        margin=dict(t=30, b=50), showlegend=False
    )
    st.plotly_chart(fig_vrev, use_container_width=True)

with col_v2:
    st.markdown("<div class='section-label'>Total trips</div>", unsafe_allow_html=True)
    fig_vtrips = go.Figure(go.Bar(
        x=vendor_summary["vendor_name"],
        y=vendor_summary["total_trips"],
        marker_color="#FF9800",
        text=[f"{v/1e6:.1f}M" for v in vendor_summary["total_trips"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} trips<extra></extra>"
    ))
    fig_vtrips.update_layout(
        height=280, yaxis_title="Trips",
        margin=dict(t=30, b=50), showlegend=False
    )
    st.plotly_chart(fig_vtrips, use_container_width=True)

with col_v3:
    st.markdown("<div class='section-label'>Avg fare per mile</div>", unsafe_allow_html=True)
    fig_vfpm = go.Figure(go.Bar(
        x=vendor_summary["vendor_name"],
        y=vendor_summary["avg_fare_per_mile"].round(2),
        marker_color="#4CAF50",
        text=["$" + f"{v:.2f}" for v in vendor_summary["avg_fare_per_mile"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.2f}/mi<extra></extra>"
    ))
    fig_vfpm.update_layout(
        height=280, yaxis_title="Avg fare / mile ($)",
        margin=dict(t=30, b=50), showlegend=False
    )
    st.plotly_chart(fig_vfpm, use_container_width=True)

top_vend = vendor_summary.iloc[0]
st.markdown(
    f"<div class='insight-box'>"
    f"<strong>{top_vend.vendor_name}</strong> leads on both revenue and trips. "
    f"Fare-per-mile differences between vendors reflect route mix "
    f"(airport vs urban) rather than pricing differences."
    f"</div>", unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------------------------
# 4. DISTANCE PATTERNS
# --------------------------------------------------

st.subheader("📏 Trip distance patterns")

st.markdown(
    "<div class='data-note'>"
    "<strong>Distance buckets: </strong>"
    "<strong>Short</strong> = under 2 miles &nbsp;|&nbsp; "
    "<strong>Medium</strong> = 2–10 miles &nbsp;|&nbsp; "
    "<strong>Long</strong> = over 10 miles"
    "</div>", unsafe_allow_html=True
)

bucket_order  = {"short": 0, "medium": 1, "long": 2}
bucket_colors = {"short": "#42A5F5", "medium": "#FF9800", "long": "#4CAF50"}

fd["order"] = fd["distance_bucket"].map(bucket_order)

dist_mix = (
    fd.groupby(["distance_bucket","order"])["total_trips"]
    .sum().reset_index().sort_values("order")
)
dist_val = (
    fd.groupby(["distance_bucket","order"])
    .agg(avg_trip_value=("avg_trip_value","mean"), avg_fpm=("avg_fare_per_mile","mean"))
    .reset_index().sort_values("order")
)

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("<div class='section-label'>Trip volume</div>", unsafe_allow_html=True)
    fig_dvol = go.Figure(go.Bar(
        x=dist_mix["distance_bucket"],
        y=dist_mix["total_trips"],
        marker_color=[bucket_colors[b] for b in dist_mix["distance_bucket"]],
        text=[f"{v/1e6:.1f}M" for v in dist_mix["total_trips"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} trips<extra></extra>"
    ))
    fig_dvol.update_layout(
        height=280, yaxis_title="Total trips",
        xaxis_title="Distance bucket",
        margin=dict(t=30, b=50), showlegend=False
    )
    st.plotly_chart(fig_dvol, use_container_width=True)

with col_d2:
    st.markdown("<div class='section-label'>Avg trip value</div>", unsafe_allow_html=True)
    fig_dval = go.Figure(go.Bar(
        x=dist_val["distance_bucket"],
        y=dist_val["avg_trip_value"].round(2),
        marker_color=[bucket_colors[b] for b in dist_val["distance_bucket"]],
        text=["$" + f"{v:.2f}" for v in dist_val["avg_trip_value"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.2f}<extra></extra>"
    ))
    fig_dval.update_layout(
        height=280, yaxis_title="Avg trip value ($)",
        xaxis_title="Distance bucket",
        margin=dict(t=30, b=50), showlegend=False
    )
    st.plotly_chart(fig_dval, use_container_width=True)

with col_d3:
    st.markdown("<div class='section-label'>Avg fare per mile</div>", unsafe_allow_html=True)
    fig_dfpm = go.Figure(go.Bar(
        x=dist_val["distance_bucket"],
        y=dist_val["avg_fpm"].round(2),
        marker_color=[bucket_colors[b] for b in dist_val["distance_bucket"]],
        text=["$" + f"{v:.2f}" for v in dist_val["avg_fpm"]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.2f}/mi<extra></extra>"
    ))
    fig_dfpm.update_layout(
        height=280, yaxis_title="Avg fare / mile ($)",
        xaxis_title="Distance bucket",
        margin=dict(t=30, b=50), showlegend=False
    )
    st.plotly_chart(fig_dfpm, use_container_width=True)

top_bucket = dist_mix.sort_values("total_trips", ascending=False).iloc[0]
top_fpm    = dist_val.sort_values("avg_fpm", ascending=False).iloc[0]
st.markdown(
    f"<div class='insight-box'>"
    f"<strong>{top_bucket.distance_bucket.capitalize()}</strong> trips dominate volume. "
    f"<strong>{top_fpm.distance_bucket.capitalize()}</strong> trips yield the highest "
    f"fare-per-mile — short trips are more lucrative per mile, "
    f"informing driver routing and incentive design."
    f"</div>", unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------------------------
# 5. PEAK DEMAND HOURS
# --------------------------------------------------

st.subheader("🕐 Peak demand hours")

hourly_trips = (
    ft.groupby("pickup_hour")["total_trips"]
    .sum().reset_index().sort_values("pickup_hour")
)
peak_hour = hourly_trips.nlargest(1, "total_trips").iloc[0]

fig_hourly = go.Figure(go.Bar(
    x=hourly_trips["pickup_hour"],
    y=hourly_trips["total_trips"],
    marker_color=[
        "#E53935" if h == int(peak_hour.pickup_hour) else "#90CAF9"
        for h in hourly_trips["pickup_hour"]
    ],
    hovertemplate="Hour %{x}:00 — %{y:,} trips<extra></extra>"
))

for x0, x1, label in [(7, 10, "Morning rush"), (16, 19, "Evening rush")]:
    fig_hourly.add_vrect(
        x0=x0, x1=x1,
        fillcolor="rgba(255,152,0,0.08)",
        layer="below", line_width=0,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#FF9800"
    )

fig_hourly.update_layout(
    height=320,
    xaxis=dict(title="Hour of day", tickmode="linear", tick0=0, dtick=1),
    yaxis_title="Total trips",
    margin=dict(t=10, b=40),
    showlegend=False
)
st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown(
    f"<div class='insight-box'>"
    f"Peak hour is <strong>{int(peak_hour.pickup_hour)}:00</strong> "
    f"with <strong>{int(peak_hour.total_trips):,}</strong> trips (red bar). "
    f"Orange bands show typical rush hour windows (07:00–10:00 and 16:00–19:00)."
    f"</div>", unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------------------------
# 6. LOCATION INTELLIGENCE
# --------------------------------------------------

st.subheader("📍 Location intelligence")

top_zones_rev = (
    fl.groupby(["zone","borough"])[["total_revenue","total_trips"]]
    .sum().reset_index()
    .sort_values("total_revenue", ascending=False)
    .head(10)
)
top_zones_val = (
    fl.groupby("zone")["avg_trip_value"]
    .mean().reset_index()
    .sort_values("avg_trip_value", ascending=False)
    .head(10)
)
borough_trips = (
    fl.groupby("borough")["total_trips"]
    .sum().reset_index()
    .sort_values("total_trips", ascending=False)
)

# ── Top 10 zones by revenue — horizontal bar, coloured by borough ──
st.markdown("<div class='section-label'>Top 10 zones by total revenue</div>", unsafe_allow_html=True)

# Build one trace per borough so legend toggles work correctly
# (one trace = one legend item = correct show/hide behaviour)
fig_zones = go.Figure()

for boro, color in BORO_COLORS.items():
    subset = top_zones_rev[top_zones_rev["borough"] == boro]
    if subset.empty:
        continue
    fig_zones.add_trace(go.Bar(
        name=boro,
        x=subset["total_revenue"].round(0),
        y=subset["zone"],
        orientation="h",
        marker_color=color,
        text=["$" + f"{v/1e6:.1f}M" for v in subset["total_revenue"]],
        textposition="outside",
        customdata=subset["total_trips"],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Revenue: $%{x:,.0f}<br>"
            "Trips: %{customdata:,}<br>"
            f"Borough: {boro}<extra></extra>"
        )
    ))

fig_zones.update_layout(
    height=420, barmode="overlay",
    xaxis=dict(title="Total revenue ($)", tickformat="$,.0f"),
    yaxis=dict(
        autorange="reversed",
        categoryorder="total ascending"
    ),
    legend=dict(orientation="h", y=1.05, x=0),
    margin=dict(t=10, l=190, b=50)
)
st.plotly_chart(fig_zones, use_container_width=True)

# ── Borough trip share — same horizontal bar style ─────────────────
st.markdown("<div class='section-label'>Trip share by borough</div>", unsafe_allow_html=True)

fig_boro = go.Figure()

for _, row in borough_trips.iterrows():
    boro  = row["borough"]
    color = BORO_COLORS.get(boro, "#90A4AE")
    pct   = row["total_trips"] / borough_trips["total_trips"].sum() * 100
    fig_boro.add_trace(go.Bar(
        name=boro,
        x=[row["total_trips"]],
        y=[boro],
        orientation="h",
        marker_color=color,
        text=[f"{pct:.1f}%"],
        textposition="outside",
        hovertemplate=f"<b>{boro}</b><br>Trips: {row['total_trips']:,}<br>{pct:.1f}%<extra></extra>"
    ))

fig_boro.update_layout(
    height=300, barmode="overlay",
    xaxis_title="Total trips",
    yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(t=10, l=120, b=50)
)
st.plotly_chart(fig_boro, use_container_width=True)

# ── Premium zones ──────────────────────────────────────────────────
st.markdown("<div class='section-label'>Top 10 premium zones by avg trip value</div>", unsafe_allow_html=True)

fig_premium = go.Figure(go.Bar(
    x=top_zones_val["zone"],
    y=top_zones_val["avg_trip_value"].round(2),
    marker_color="#FF9800",
    text=["$" + f"{v:.2f}" for v in top_zones_val["avg_trip_value"]],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Avg trip value: $%{y:.2f}<extra></extra>"
))
fig_premium.update_layout(
    height=300,
    xaxis_title="Zone", yaxis_title="Avg trip value ($)",
    xaxis_tickangle=-30,
    margin=dict(t=30, b=100), showlegend=False
)
st.plotly_chart(fig_premium, use_container_width=True)

top_boro     = borough_trips.iloc[0]
top_zone     = top_zones_rev.iloc[0]
premium_zone = top_zones_val.iloc[0]
st.markdown(
    f"<div class='insight-box'>"
    f"<strong>{top_boro.borough}</strong> dominates trip volume "
    f"({top_boro.total_trips:,} trips). "
    f"<strong>{top_zone.zone}</strong> is the highest revenue zone. "
    f"<strong>{premium_zone.zone}</strong> has the highest avg trip value "
    f"(${premium_zone.avg_trip_value:.2f}) — indicating long-distance or airport "
    f"routes distinct from high-volume urban clusters."
    f"</div>", unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    "<div style='text-align:right; font-size:12px; color:#aaa;'>"
    "Data source: NYC TLC Trip Record Data &nbsp;|&nbsp; Gold layer aggregations"
    "</div>",
    unsafe_allow_html=True
)