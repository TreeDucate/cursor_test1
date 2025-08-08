import os
import numpy as np
import pandas as pd
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

# Plotly defaults aligned with VW theme
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#001E50", "#00A3E0", "#6C757D", "#B0BEC5", "#263238"]
px.defaults.width = None
px.defaults.height = 400
# Note: Plotly figures have their own font; set a common font family
DEFAULT_FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"


# -----------------------------
# Synthetic data generation (Aftersales)
# -----------------------------
np.random.seed(42)

brands = ["Volkswagen", "Audi", "SEAT", "Škoda", "Porsche"]
regions = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]
channels = ["Dealer", "Independent", "Online"]
product_groups = ["Parts", "Accessories", "Service"]

# Generate monthly date range for last 24 months
end_date = pd.Period(datetime.today().strftime("%Y-%m"), freq="M").to_timestamp()
start_date = (end_date - pd.DateOffset(months=23)).replace(day=1)
months = pd.date_range(start=start_date, end=end_date, freq="MS")

rows = []
for month in months:
    for region in regions:
        for brand in brands:
            for channel in channels:
                for pg in product_groups:
                    base_units = np.random.poisson(lam=120)
                    seasonality = 1.0 + 0.2 * np.sin(2 * np.pi * (month.month / 12))
                    brand_factor = 1.3 if brand == "Volkswagen" else 1.1 if brand == "Audi" else 0.9
                    channel_factor = 1.15 if channel == "Dealer" else 0.95 if channel == "Independent" else 1.05
                    pg_factor = 1.2 if pg == "Service" else 1.0

                    units = max(0, int(base_units * seasonality * brand_factor * channel_factor * pg_factor * np.random.uniform(0.85, 1.15)))
                    revenue = units * np.random.uniform(45, 480)
                    margin_pct = np.random.uniform(0.18, 0.42)
                    margin = revenue * margin_pct
                    warranty_rate = np.random.uniform(0.01, 0.05)

                    rows.append(
                        {
                            "month": month,
                            "region": region,
                            "brand": brand,
                            "channel": channel,
                            "product_group": pg,
                            "units": units,
                            "revenue": revenue,
                            "margin": margin,
                            "warranty_rate": warranty_rate,
                        }
                    )

df = pd.DataFrame(rows)

# -----------------------------
# Synthetic data generation (VW models by German postal code)
# -----------------------------
models = ["Golf", "Tiguan", "Passat", "Polo", "T-Roc", "ID.3", "ID.4", "Arteon", "Touran", "Up!"]
# Generate ~400 random postal codes with coordinates within Germany bbox
num_plz = 400
np.random.seed(7)
plz_codes = [f"{np.random.randint(1, 99):02d}{np.random.randint(0, 99):02d}{np.random.randint(0, 99):02d}" for _ in range(num_plz)]
# Germany approx bbox: lat 47.2..55.1, lon 5.9..15.0
lats = np.random.uniform(47.2, 55.1, size=num_plz)
lons = np.random.uniform(5.9, 15.0, size=num_plz)
base_df = pd.DataFrame({"postal_code": plz_codes, "lat": lats, "lon": lons})

rows_models = []
for _, r in base_df.iterrows():
    for m in models:
        units = max(0, int(np.random.lognormal(mean=2.5, sigma=0.6)))
        rows_models.append({
            "postal_code": r["postal_code"],
            "lat": r["lat"],
            "lon": r["lon"],
            "model": m,
            "units": units,
        })

vw_plz_df = pd.DataFrame(rows_models)

# Helpful options
model_options = [{"label": m, "value": m} for m in models]
plz_options = [{"label": pc, "value": pc} for pc in sorted(vw_plz_df["postal_code"].unique())]


# -----------------------------
# App init
# -----------------------------
external_scripts = []
app = Dash(__name__, assets_folder="assets", external_scripts=external_scripts, suppress_callback_exceptions=True)
app.title = "VW Aftersales Germany"
server = app.server


# -----------------------------
# Components
# -----------------------------
brand_options = [{"label": b, "value": b} for b in brands]
channel_options = [{"label": c, "value": c} for c in channels]
pgroup_options = [{"label": p, "value": p} for p in product_groups]
region_options = [{"label": r, "value": r} for r in regions]


def kpi_card(title: str, value: str, delta: str = None, id_suffix: str = ""):
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(className="kpi-title", children=title),
            html.Div(className="kpi-value", id=f"kpi-value-{id_suffix}", children=value),
            html.Div(className="kpi-delta", id=f"kpi-delta-{id_suffix}", children=delta or ""),
        ],
    )


# -----------------------------
# Page layouts
# -----------------------------

def layout_header(title: str, subtitle: str = ""):
    return html.Div(
        className="header",
        children=[
            html.Div(className="brand-mark", children=[html.Span("VW")]),
            html.Div(className="titles", children=[html.H1(title), html.Div(subtitle, className="subtitle")]),
        ],
    )


def main_page_layout():
    return html.Div(
        className="container",
        children=[
            layout_header("Aftersales Dashboard", "Germany • Volkswagen Group"),
            html.Div(
                className="controls",
                children=[
                    html.Div(className="control", children=[html.Label("Brand"), dcc.Dropdown(id="brand-dd", options=brand_options, value=["Volkswagen"], multi=True, clearable=False)]),
                    html.Div(className="control", children=[html.Label("Channel"), dcc.Dropdown(id="channel-dd", options=channel_options, value=[c["value"] for c in channel_options], multi=True)]),
                    html.Div(className="control", children=[html.Label("Product Group"), dcc.Dropdown(id="pg-dd", options=pgroup_options, value=[p["value"] for p in pgroup_options], multi=True)]),
                    html.Div(className="control", children=[html.Label("Region"), dcc.Dropdown(id="region-dd", options=region_options, value=[r["value"] for r in region_options], multi=True)]),
                    html.Div(className="control", children=[
                        html.Label("Date Range"),
                        dcc.RangeSlider(
                            id="month-slider",
                            min=0,
                            max=len(months) - 1,
                            value=[max(0, len(months) - 6), len(months) - 1],
                            allowCross=False,
                            marks={i: m.strftime("%b '%y") for i, m in enumerate(months)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ]),
                ],
            ),
            html.Div(className="kpi-row", children=[
                kpi_card("Revenue", "€0", id_suffix="revenue"),
                kpi_card("Units", "0", id_suffix="units"),
                kpi_card("Gross Margin", "0%", id_suffix="margin"),
                kpi_card("Warranty Rate", "0%", id_suffix="warranty"),
            ]),
            html.Div(className="charts-grid", children=[
                html.Div([dcc.Graph(id="rev-trend")], className="chart"),
                html.Div([dcc.Graph(id="units-by-region")], className="chart"),
                html.Div([dcc.Graph(id="mix-by-brand")], className="chart"),
                html.Div([dcc.Graph(id="pg-sunburst")], className="chart"),
            ]),
            html.Div(className="table-wrap", children=[
                dash_table.DataTable(
                    id="detail-table",
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"padding": "10px", "textAlign": "left"},
                    sort_action="native",
                    filter_action="native",
                )
            ]),
            html.Div(className="footer", children=[html.Div("Synthetic data for demo purposes. Not affiliated with Volkswagen AG.")]),
        ],
    )


def research_page_layout():
    return html.Div(
        className="container",
        children=[
            layout_header("VW Model Sales Map", "Germany • Postal Code Research"),
            html.Div(
                className="controls",
                children=[
                    html.Div(className="control", children=[
                        html.Label("Query Mode"),
                        dcc.RadioItems(
                            id="query-mode",
                            options=[
                                {"label": "Model across postal codes", "value": "by-model"},
                                {"label": "Models within a postal code", "value": "by-plz"},
                            ],
                            value="by-model",
                            labelStyle={"display": "inline-block", "marginRight": "16px"},
                        ),
                    ]),
                    html.Div(className="control", children=[
                        html.Label("Model"),
                        dcc.Dropdown(id="model-filter", options=model_options, value="Golf", multi=False, clearable=False),
                    ], id="model-control"),
                    html.Div(className="control", children=[
                        html.Label("Postal Code"),
                        dcc.Dropdown(id="plz-filter", options=plz_options[:500], value=plz_options[0]["value"] if plz_options else None, multi=False),
                    ], id="plz-control"),
                ],
            ),
            html.Div(className="charts-grid", children=[
                html.Div([dcc.Graph(id="de-map")], className="chart"),
                html.Div([dash_table.DataTable(id="plz-table", page_size=10)], className="chart"),
            ]),
            html.Div(className="footer", children=[html.Div("Map uses OpenStreetMap tiles. Synthetic positions within Germany bbox.")]),
        ],
    )


# -----------------------------
# App shell with sidebar
# -----------------------------
app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="sidebar",
            children=[
                html.Div(className="brand-mark large", children=[html.Span("VW")]),
                html.Nav(children=[
                    dcc.Link("Dashboard", href="/", className="nav-link", id="nav-dashboard"),
                    dcc.Link("Research Map", href="/research", className="nav-link", id="nav-research"),
                ]),
            ],
        ),
        html.Div(
            className="content",
            children=[
                dcc.Location(id="url"),
                html.Div(id="page-content"),
            ],
        ),
    ],
)


# -----------------------------
# Callbacks - main page (aftersales)
# -----------------------------

def filter_df(brand_values, channel_values, pg_values, region_values, slider_range):
    start_idx, end_idx = slider_range
    selected_months = months[start_idx : end_idx + 1]
    mask = (
        df["brand"].isin(brand_values)
        & df["channel"].isin(channel_values)
        & df["product_group"].isin(pg_values)
        & df["region"].isin(region_values)
        & df["month"].isin(selected_months)
    )
    return df.loc[mask].copy()


@app.callback(
    Output("rev-trend", "figure"),
    Output("units-by-region", "figure"),
    Output("mix-by-brand", "figure"),
    Output("pg-sunburst", "figure"),
    Output("detail-table", "data"),
    Output("detail-table", "columns"),
    Output("kpi-value-revenue", "children"),
    Output("kpi-value-units", "children"),
    Output("kpi-value-margin", "children"),
    Output("kpi-value-warranty", "children"),
    Input("brand-dd", "value"),
    Input("channel-dd", "value"),
    Input("pg-dd", "value"),
    Input("region-dd", "value"),
    Input("month-slider", "value"),
)

def update_dashboard(brand_values, channel_values, pg_values, region_values, slider_range):
    dff = filter_df(brand_values, channel_values, pg_values, region_values, slider_range)

    total_revenue = dff["revenue"].sum()
    total_units = dff["units"].sum()
    total_margin = dff["margin"].sum()
    margin_pct = (total_margin / total_revenue) if total_revenue > 0 else 0
    avg_warranty = dff["warranty_rate"].mean() if len(dff) else 0

    kpi_rev = f"€{total_revenue:,.0f}"
    kpi_units = f"{total_units:,.0f}"
    kpi_margin = f"{margin_pct*100:,.1f}%"
    kpi_warranty = f"{avg_warranty*100:,.2f}%"

    rev_trend = dff.groupby("month", as_index=False)["revenue"].sum().sort_values("month")
    fig_trend = px.area(rev_trend, x="month", y="revenue", title="Revenue Trend", labels={"month": "Month", "revenue": "Revenue (€)"}, markers=True)
    fig_trend.update_traces(line_color="#001E50")
    fig_trend.update_layout(margin=dict(l=10, r=10, t=50, b=10), font=dict(family=DEFAULT_FONT))

    units_region = dff.groupby("region", as_index=False)["units"].sum().sort_values("units", ascending=False).head(10)
    fig_region = px.bar(units_region, x="units", y="region", orientation="h", title="Units by Region (Top 10)", labels={"units": "Units", "region": "Region"}, color_discrete_sequence=["#00A3E0"])
    fig_region.update_layout(margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(categoryorder="total ascending"), font=dict(family=DEFAULT_FONT))

    brand_mix = dff.groupby("brand", as_index=False)["revenue"].sum()
    fig_mix = px.pie(brand_mix, names="brand", values="revenue", title="Revenue Mix by Brand", hole=0.45, color_discrete_sequence=["#001E50", "#00A3E0", "#6C757D", "#B0BEC5", "#263238"])
    fig_mix.update_layout(font=dict(family=DEFAULT_FONT))

    pg_hier = dff.groupby(["product_group", "channel", "region"], as_index=False)["revenue"].sum()
    fig_sun = px.sunburst(pg_hier, path=["product_group", "channel", "region"], values="revenue", title="Revenue Hierarchy", color_discrete_sequence=["#001E50", "#00A3E0", "#6C757D", "#B0BEC5", "#263238"], maxdepth=2)
    fig_sun.update_layout(font=dict(family=DEFAULT_FONT))

    table_df = (
        dff.groupby(["month", "brand", "region", "channel", "product_group"], as_index=False)
        .agg(units=("units", "sum"), revenue=("revenue", "sum"), margin=("margin", "sum"))
        .sort_values(["month", "revenue"], ascending=[True, False])
    )
    table_df["month"] = table_df["month"].dt.strftime("%Y-%m")
    columns = [{"name": col.replace("_", " ").title(), "id": col, "type": "numeric" if col in ["units", "revenue", "margin"] else "text"} for col in table_df.columns]

    return (fig_trend, fig_region, fig_mix, fig_sun, table_df.to_dict("records"), columns, kpi_rev, kpi_units, kpi_margin, kpi_warranty)


# -----------------------------
# Callbacks - router and research page
# -----------------------------
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/research":
        return research_page_layout()
    return main_page_layout()


@app.callback(
    Output("model-control", "style"),
    Output("plz-control", "style"),
    Input("query-mode", "value"),
)
def toggle_controls(mode):
    if mode == "by-plz":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("de-map", "figure"),
    Output("plz-table", "data"),
    Output("plz-table", "columns"),
    Input("query-mode", "value"),
    Input("model-filter", "value"),
    Input("plz-filter", "value"),
)

def update_research(mode, model_value, plz_value):
    if mode == "by-plz" and plz_value:
        dff = vw_plz_df[vw_plz_df["postal_code"] == plz_value]
        # Single marker on the map
        loc = dff.iloc[0]
        total_units = int(dff["units"].sum())
        map_df = pd.DataFrame({"lat": [loc["lat"]], "lon": [loc["lon"]], "label": [f"PLZ {plz_value}"], "units": [total_units]})
        fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", size="units", hover_name="label", hover_data={"units": True}, zoom=7, height=420)
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=40, b=10))
        # Table by model for the postal code
        table = dff.groupby("model", as_index=False)["units"].sum().sort_values("units", ascending=False)
        columns = [{"name": "Model", "id": "model"}, {"name": "Units", "id": "units", "type": "numeric"}]
        return fig, table.to_dict("records"), columns

    # Default: by model across postal codes
    if not model_value:
        model_value = models[0]
    dff = vw_plz_df[vw_plz_df["model"] == model_value]
    agg = dff.groupby(["postal_code", "lat", "lon"], as_index=False)["units"].sum()
    fig = px.scatter_mapbox(agg, lat="lat", lon="lon", size="units", color_discrete_sequence=["#001E50"], hover_name="postal_code", hover_data={"units": True}, zoom=5, height=420)
    fig.update_layout(mapbox_style="open-street-map", mapbox_center={"lat": 51.2, "lon": 10.4}, margin=dict(l=10, r=10, t=40, b=10))
    # Table of top postal codes for the model
    table = agg.sort_values("units", ascending=False).head(50)[["postal_code", "units"]]
    columns = [{"name": "Postal Code", "id": "postal_code"}, {"name": "Units", "id": "units", "type": "numeric"}]
    return fig, table.to_dict("records"), columns


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)