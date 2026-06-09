"""
Portfolio Analytics Dashboard
==============================
Dash + Groq AI per-chart commentary + News + Full risk analytics.
python app.py → http://localhost:8050
"""

import dash, os
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

RISK_FREE = 0.045
TRADING_DAYS = 252
GOLD = "#D4A843"
COLORS = [GOLD, "#5C8AE6", "#4CAF50", "#EF5350", "#9B7FD4", "#E8C872", "#E87461", "#61C4E8"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Portfolio Analytics"
server = app.server

CARD = {"backgroundColor": "#132038", "border": "1px solid rgba(212,168,67,0.15)",
        "borderRadius": "6px", "padding": "20px", "marginBottom": "16px"}

AI_NOTE = {"color": "#7A8BA0", "fontSize": "0.84rem", "lineHeight": "1.6",
           "marginTop": "10px", "borderTop": "1px solid #1A2A44", "paddingTop": "10px"}


def mkpi(label, value, color="#E0E6ED"):
    return html.Div(style={"backgroundColor": "#132038", "border": "1px solid rgba(212,168,67,0.15)",
                           "borderRadius": "4px", "padding": "14px", "textAlign": "center"}, children=[
        html.P(label, style={"color": "#7A8BA0", "fontSize": "0.65rem", "textTransform": "uppercase",
                             "letterSpacing": "1.2px", "margin": "0"}),
        html.P(value, style={"color": color, "fontSize": "1.3rem", "fontWeight": "700",
                             "margin": "4px 0 0 0"})])


def sfig(fig):
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7A8BA0", size=11),
                      xaxis=dict(gridcolor="#1A2A44", zeroline=False),
                      yaxis=dict(gridcolor="#1A2A44", zeroline=False),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7A8BA0")),
                      margin=dict(l=40, r=20, t=45, b=30))
    return fig


def fetch_news(tickers):
    all_news = {}
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
            parsed = []
            for a in items[:4]:
                if not isinstance(a, dict): continue
                ct = a.get("content", {}) if isinstance(a.get("content"), dict) else {}
                title = a.get("title") or ct.get("title", "")
                link = a.get("link") or (ct.get("canonicalUrl", {}).get("url", "")
                       if isinstance(ct.get("canonicalUrl"), dict) else "")
                pub = a.get("publisher") or (ct.get("provider", {}).get("displayName", "")
                      if isinstance(ct.get("provider"), dict) else "")
                if title:
                    parsed.append({"title": title, "link": link, "publisher": pub})
            if parsed:
                all_news[t] = parsed
        except Exception:
            pass
    return all_news


def fetch_snippet(url):
    try:
        r = requests.get(url, timeout=4, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p")[:4])
        if text and len(text) > 30 and "not found" not in text.lower() and "oops" not in text.lower() and "cache" not in text.lower():
            return (text[:250].rsplit(" ", 1)[0] + "...") if len(text) > 250 else text
    except Exception:
        pass
    return ""


def generate_commentary(tickers, weights, metrics, holdings_perf, news):
    if not os.getenv("GROQ_API_KEY"):
        return {}
    pstr = ", ".join(f"{t} {w:.0%}" for t, w in zip(tickers, weights))
    hstr = "\n".join(f"- {h['Ticker']}: {h['Return']} return, {h['Volatility']} vol" for h in holdings_perf)
    nstr = "\n".join(f"- {t}: {'; '.join(a['title'] for a in arts[:2])}" for t, arts in news.items()) or "None"

    prompt = f"""You are a Goldman Sachs senior wealth analyst. Write brief chart explanations for a client portfolio review.

Portfolio: {pstr} | Period: {metrics['period']}
Total Return: {metrics['total_return']} | Benchmark ({metrics['benchmark']}): {metrics['bench_return']}
Alpha: {metrics['alpha']} | Sharpe: {metrics['sharpe']} | Beta: {metrics['beta']}
Max Drawdown: {metrics['max_dd']} | VaR 95%: {metrics['var95']} | Volatility: {metrics['volatility']}
Holdings: {hstr}
Recent News: {nstr}

Write EXACTLY this format (2-3 sentences each, use specific numbers, connect to news/current events):

PERFORMANCE: [explain portfolio vs benchmark returns, why it outperformed or underperformed]
HOLDINGS: [which stocks drove returns up or down, connect to recent news events]
DRAWDOWN: [interpret the worst decline, what likely caused it, recovery context]
CORRELATION: [which assets move together vs independently, diversification assessment]
MONTECARLO: [interpret simulation — median outcome, probability of loss, range of scenarios]
DISTRIBUTION: [what the return shape reveals about risk, interpret the VaR number practically]
OVERVIEW: [3-sentence executive summary with forward outlook based on current news]"""

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=1500)
        resp = llm.invoke(prompt)
        sections = {}
        current = None
        for line in resp.content.split("\n"):
            line = line.strip()
            for key in ["PERFORMANCE", "HOLDINGS", "DRAWDOWN", "CORRELATION", "MONTECARLO", "DISTRIBUTION", "OVERVIEW"]:
                if line.upper().startswith(key + ":"):
                    current = key.lower()
                    sections[current] = line.split(":", 1)[1].strip()
                    break
            else:
                if current and line:
                    sections[current] = sections.get(current, "") + " " + line
        return sections
    except Exception as e:
        return {"overview": f"AI commentary unavailable: {e}"}


def ai_note(text):
    if not text:
        return None
    return html.P(text, style=AI_NOTE)


# ---- Layout ----
app.layout = dbc.Container(fluid=True, style={"backgroundColor": "#0B1426", "minHeight": "100vh",
                                                "padding": "20px 30px"}, children=[
    dbc.Row(className="mb-3", children=[dbc.Col(width=12, children=[
        html.Div(style={"background": "linear-gradient(135deg, #0B1426, #132038, #1A2A44)",
                        "borderRadius": "6px", "padding": "20px 28px",
                        "borderBottom": f"3px solid {GOLD}"}, children=[
            html.H1("Portfolio Analytics", style={"color": GOLD, "fontSize": "1.8rem",
                     "fontWeight": "700", "margin": "0"}),
            html.P("Risk metrics • Monte Carlo • AI-powered market intelligence • Groq LLM",
                   style={"color": "#4E5D73", "fontSize": "0.85rem", "margin": "4px 0 0 0"}),
        ])])]),

    dbc.Row(className="mb-3", children=[
        dbc.Col(md=4, children=[html.Div(style=CARD, children=[
            html.Label("Portfolio (TICKER WEIGHT%)", style={"color": GOLD, "fontWeight": "600",
                       "fontSize": "0.8rem", "marginBottom": "8px"}),
            dcc.Textarea(id="port-in", value="AAPL 25\nMSFT 20\nGOOGL 15\nAMZN 10\nBND 15\nVTI 15",
                         style={"width": "100%", "height": "130px", "backgroundColor": "#0B1426",
                                "color": "#E0E6ED", "border": "1px solid #1A2A44",
                                "borderRadius": "4px", "padding": "10px", "fontFamily": "monospace"})])]),
        dbc.Col(md=5, children=[html.Div(style=CARD, children=[
            dbc.Row(className="mb-3", children=[
                dbc.Col([html.Label("Benchmark", style={"color": "#7A8BA0", "fontSize": "0.75rem"}),
                         dbc.Input(id="bench-in", value="SPY", style={"backgroundColor": "#0B1426",
                                   "color": "#E0E6ED", "border": "1px solid #1A2A44"})]),
                dbc.Col([html.Label("Period", style={"color": "#7A8BA0", "fontSize": "0.75rem"}),
                         dbc.Select(id="period-in", value="1 Year",
                                    options=[{"label": p, "value": p} for p in
                                             ["6 Months", "1 Year", "2 Years", "5 Years"]],
                                    style={"backgroundColor": "#0B1426", "color": "#E0E6ED",
                                           "border": "1px solid #1A2A44"})])]),
            dbc.Row(children=[
                dbc.Col([html.Label("MC Runs", style={"color": "#7A8BA0", "fontSize": "0.75rem"}),
                         dbc.Input(id="mc-in", value="1000", type="number",
                                   style={"backgroundColor": "#0B1426", "color": "#E0E6ED",
                                          "border": "1px solid #1A2A44"})]),
                dbc.Col(className="d-flex align-items-end", children=[
                    dbc.Button("Analyze Portfolio", id="go-btn", className="w-100",
                               style={"fontWeight": "600", "backgroundColor": GOLD,
                                      "border": "none", "color": "#0B1426"})])])])]),
        dbc.Col(md=3, children=[
            html.Div(id="status", style={**CARD, "textAlign": "center", "display": "flex",
                     "alignItems": "center", "justifyContent": "center", "height": "100%"},
                     children=[html.P("Enter portfolio and click Analyze",
                                      style={"color": "#7A8BA0", "margin": "0"})])])]),

    dcc.Loading(type="dot", color=GOLD, children=[
        html.Div(id="out-kpi"),
        html.Div(id="out-overview"),
        html.Div(id="out-charts"),
        html.Div(id="out-news"),
    ]),

    html.Hr(style={"borderColor": "#1A2A44"}),
    html.P(f"Portfolio Analytics • Yahoo Finance • Groq AI • Risk-free: {RISK_FREE:.1%}",
           style={"textAlign": "center", "color": "#4E5D73", "fontSize": "0.8rem"}),
])


@app.callback(
    [Output("out-kpi", "children"), Output("out-overview", "children"),
     Output("out-charts", "children"), Output("out-news", "children"),
     Output("status", "children")],
    Input("go-btn", "n_clicks"),
    [State("port-in", "value"), State("bench-in", "value"),
     State("period-in", "value"), State("mc-in", "value")],
    prevent_initial_call=True)
def update(n, ptxt, bench, period, mc_n):
    if not ptxt or not ptxt.strip():
        return [None]*4 + [html.P("Enter portfolio", style={"color": "#EF5350"})]

    holdings = {}
    for line in ptxt.strip().split("\n"):
        p = line.strip().split()
        if len(p) >= 2:
            try: holdings[p[0].upper()] = float(p[1])
            except: pass
    if not holdings:
        return [None]*4 + [html.P("Invalid format", style={"color": "#EF5350"})]

    tickers = list(holdings.keys())
    rw = np.array(list(holdings.values()))
    weights = rw / rw.sum()
    mc_n = int(mc_n or 1000)
    lb = {"6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825}
    end_d, start_d = datetime.today(), datetime.today() - timedelta(days=lb.get(period, 365))

    try:
        data = yf.download(tickers + [bench], start=start_d, end=end_d, progress=False)
        prices = data.get("Adj Close", data.get("Close", data))
        if isinstance(prices, pd.Series): prices = prices.to_frame(tickers[0])
        prices = prices.dropna()
    except Exception as e:
        return [None]*4 + [html.P(f"Error: {e}", style={"color": "#EF5350"})]

    miss = [t for t in tickers if t not in prices.columns]
    if miss or bench not in prices.columns:
        return [None]*4 + [html.P(f"Not found: {', '.join(miss or [bench])}", style={"color": "#EF5350"})]

    # ---- Calculations ----
    rets = prices[tickers].pct_change().dropna()
    br = prices[bench].pct_change().dropna()
    ci = rets.index.intersection(br.index)
    rets, br = rets.loc[ci], br.loc[ci]
    pr = (rets * weights).sum(axis=1)
    pc, bc = (1 + pr).cumprod(), (1 + br).cumprod()

    ar = pr.mean() * TRADING_DAYS
    av = pr.std() * np.sqrt(TRADING_DAYS)
    sh = (ar - RISK_FREE) / av if av > 0 else 0
    ba = br.mean() * TRADING_DAYS
    cv = np.cov(pr, br) if len(pr) > 1 else [[0, 0], [0, 1]]
    beta = cv[0][1] / cv[1][1] if cv[1][1] != 0 else 0
    alpha = ar - (RISK_FREE + beta * (ba - RISK_FREE))
    rm = pc.cummax(); dd = pc / rm - 1; mdd = dd.min()
    var95 = np.percentile(pr, 5)
    tr = pc.iloc[-1] - 1; btr = bc.iloc[-1] - 1
    vc = lambda v: "#4CAF50" if v > 0 else "#EF5350"

    h_data = []
    for tn in tickers:
        ti = rets[tn]
        h_data.append({"Ticker": tn, "Weight": f"{weights[tickers.index(tn)]:.0%}",
                        "Return": f"{((1+ti).prod()-1):.1%}",
                        "Volatility": f"{ti.std()*np.sqrt(TRADING_DAYS):.1%}",
                        "Sharpe": f"{(ti.mean()*TRADING_DAYS-RISK_FREE)/(ti.std()*np.sqrt(TRADING_DAYS)):.2f}" if ti.std()>0 else "–"})

    news = fetch_news(tickers)

    met = {"total_return": f"{tr:.1%}", "bench_return": f"{btr:.1%}", "alpha": f"{alpha:.1%}",
           "sharpe": f"{sh:.2f}", "beta": f"{beta:.2f}", "max_dd": f"{mdd:.1%}",
           "var95": f"{var95:.2%}", "volatility": f"{av:.1%}", "benchmark": bench, "period": period}
    cmt = generate_commentary(tickers, weights, met, h_data, news)

    # ---- KPIs ----
    kpis = dbc.Row(className="mb-3 g-2", children=[
        dbc.Col(mkpi("Total Return", f"{tr:.1%}", vc(tr)), width=True),
        dbc.Col(mkpi("Bench Return", f"{btr:.1%}", vc(btr)), width=True),
        dbc.Col(mkpi("Alpha", f"{alpha:.1%}", vc(alpha)), width=True),
        dbc.Col(mkpi("Sharpe", f"{sh:.2f}", vc(sh)), width=True),
        dbc.Col(mkpi("Beta", f"{beta:.2f}"), width=True),
        dbc.Col(mkpi("Max Drawdown", f"{mdd:.1%}", "#EF5350"), width=True),
        dbc.Col(mkpi("Volatility", f"{av:.1%}"), width=True),
        dbc.Col(mkpi("VaR 95%", f"{var95:.2%}", "#EF5350"), width=True)])

    # ---- Overview ----
    overview = dbc.Row(className="mb-3", children=[dbc.Col(width=12, children=[
        html.Div(style={**CARD, "borderLeft": f"4px solid {GOLD}"}, children=[
            html.H5("Executive Summary", style={"color": GOLD, "fontWeight": "700", "marginBottom": "10px"}),
            html.P(cmt.get("overview", "Enable GROQ_API_KEY for AI commentary."),
                   style={"color": "#B8C4D4", "fontSize": "0.92rem", "lineHeight": "1.7", "margin": "0"}),
        ])])])

    # ---- Charts ----
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=pc.index, y=pc.values, name="Portfolio", line=dict(color=GOLD, width=2.5)))
    fig1.add_trace(go.Scatter(x=bc.index, y=bc.values, name=bench, line=dict(color="#7A8BA0", width=1.5, dash="dash")))
    fig1.update_layout(title="Cumulative Returns", yaxis_title="Growth of $1", hovermode="x unified")
    sfig(fig1)

    figs = go.Figure()
    for i, tn in enumerate(tickers):
        sc = (1 + rets[tn]).cumprod()
        figs.add_trace(go.Scatter(x=sc.index, y=sc.values, name=tn, line=dict(width=1.8, color=COLORS[i % len(COLORS)])))
    figs.update_layout(title="Individual Holdings", yaxis_title="Growth of $1", hovermode="x unified")
    sfig(figs)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd.values*100, fill="tozeroy",
                              fillcolor="rgba(239,83,80,0.12)", line=dict(color="#EF5350", width=1.5)))
    fig2.update_layout(title="Drawdown from Peak", yaxis_title="%")
    sfig(fig2)

    corr = rets.corr()
    fig3 = px.imshow(corr, x=corr.columns, y=corr.columns,
                     color_continuous_scale=["#EF5350", "#0B1426", "#4CAF50"],
                     color_continuous_midpoint=0, aspect="auto", labels=dict(color="Corr"))
    fig3.update_layout(title="Correlation Matrix")
    sfig(fig3)

    mu, sig = pr.mean(), pr.std()
    np.random.seed(42)
    sim = np.zeros((TRADING_DAYS, mc_n))
    for i in range(mc_n): sim[:, i] = np.cumprod(1 + np.random.normal(mu, sig, TRADING_DAYS))
    p5, p25, p50, p75, p95 = [np.percentile(sim, p, axis=1) for p in [5, 25, 50, 75, 95]]
    dr = list(range(1, TRADING_DAYS + 1))

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=dr+dr[::-1], y=list(p95)+list(p5)[::-1], fill="toself",
                              fillcolor="rgba(212,168,67,0.06)", line=dict(width=0), name="5th–95th"))
    fig4.add_trace(go.Scatter(x=dr+dr[::-1], y=list(p75)+list(p25)[::-1], fill="toself",
                              fillcolor="rgba(212,168,67,0.12)", line=dict(width=0), name="25th–75th"))
    fig4.add_trace(go.Scatter(x=dr, y=p50, line=dict(color=GOLD, width=2), name="Median"))
    fig4.add_hline(y=1.0, line_dash="dot", line_color="#4E5D73")
    fig4.update_layout(title=f"Monte Carlo — {mc_n:,} Simulations", xaxis_title="Days", yaxis_title="$1 Growth")
    sfig(fig4)

    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(x=pr*100, nbinsx=50, marker_color=GOLD, opacity=0.7))
    fig5.add_vline(x=var95*100, line_dash="dash", line_color="#EF5350",
                   annotation_text=f"VaR: {var95:.2%}", annotation_font_color="#EF5350")
    fig5.update_layout(title="Return Distribution", xaxis_title="Daily Return %")
    sfig(fig5)

    fig6 = go.Figure(data=[go.Pie(labels=tickers, values=weights*100,
                    marker=dict(colors=COLORS[:len(tickers)]),
                    hole=0.45, textinfo="label+percent", textfont=dict(color="white"))])
    fig6.update_layout(title="Allocation", showlegend=False)
    sfig(fig6)

    finals = sim[-1, :]
    mc_stats = dbc.Row(className="g-2 mt-2", children=[
        dbc.Col(mkpi("Median", f"${p50[-1]:.2f}"), width=3),
        dbc.Col(mkpi("Best 95th", f"${p95[-1]:.2f}", "#4CAF50"), width=3),
        dbc.Col(mkpi("Worst 5th", f"${p5[-1]:.2f}", "#EF5350"), width=3),
        dbc.Col(mkpi("P(Loss)", f"{(finals<1).mean():.0%}",
                     "#EF5350" if (finals<1).mean()>0.3 else "#4CAF50"), width=3)])

    charts = html.Div([
        dbc.Row(className="mb-3 g-3", children=[
            dbc.Col(md=7, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=fig1, config={"displayModeBar": False}),
                ai_note(cmt.get("performance"))])]),
            dbc.Col(md=5, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=figs, config={"displayModeBar": False}),
                ai_note(cmt.get("holdings"))])])]),
        dbc.Row(className="mb-3 g-3", children=[
            dbc.Col(md=7, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=fig4, config={"displayModeBar": False}),
                mc_stats,
                ai_note(cmt.get("montecarlo"))])]),
            dbc.Col(md=5, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=fig5, config={"displayModeBar": False}),
                ai_note(cmt.get("distribution"))])])]),
        dbc.Row(className="mb-3 g-3", children=[
            dbc.Col(md=4, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=fig3, config={"displayModeBar": False}),
                ai_note(cmt.get("correlation"))])]),
            dbc.Col(md=3, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=fig6, config={"displayModeBar": False})])]),
            dbc.Col(md=5, children=[html.Div(style=CARD, children=[
                dcc.Graph(figure=fig2, config={"displayModeBar": False}),
                ai_note(cmt.get("drawdown"))])])]),
        dbc.Row(className="mb-3", children=[dbc.Col(width=12, children=[
            html.Div(style=CARD, children=[
                html.H6("Holdings Breakdown", style={"color": GOLD, "marginBottom": "12px"}),
                dash_table.DataTable(data=h_data,
                    columns=[{"name": c, "id": c} for c in h_data[0].keys()],
                    style_header={"backgroundColor": "#1A2A44", "color": "#7A8BA0",
                                  "fontWeight": "600", "fontSize": "0.75rem", "border": "none"},
                    style_cell={"backgroundColor": "#132038", "color": "#E0E6ED",
                                "border": "1px solid #1A2A44", "fontSize": "0.85rem", "padding": "10px"},
                    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#0F1B2E"}])])])])])

    # ---- News ----
    news_children = []
    for t, articles in news.items():
        news_children.append(html.H6(t, style={"color": GOLD, "marginBottom": "10px", "marginTop": "12px"}))
        for a in articles[:3]:
            snippet = fetch_snippet(a["link"]) if a.get("link") else ""
            children = [
                html.P(a["title"], style={"color": "#E0E6ED", "fontWeight": "600", "fontSize": "0.92rem", "margin": "0"}),
                html.P(a.get("publisher", ""), style={"color": "#4E5D73", "fontSize": "0.75rem", "margin": "4px 0 6px 0"})]
            if snippet:
                children.append(html.P(snippet, style={"color": "#7A8BA0", "fontSize": "0.84rem",
                                       "lineHeight": "1.5", "margin": "0 0 8px 0"}))
            if a.get("link"):
                children.append(html.A("Read full article →", href=a["link"], target="_blank",
                                       style={"color": GOLD, "fontSize": "0.8rem", "textDecoration": "none", "fontWeight": "600"}))
            news_children.append(html.Div(style={**CARD, "borderLeft": f"3px solid {GOLD}", "padding": "14px 18px"}, children=children))

    news_section = dbc.Row(className="mb-3", children=[dbc.Col(width=12, children=[
        html.Div(style=CARD, children=[
            html.H5("Market News", style={"color": GOLD, "fontWeight": "700", "marginBottom": "14px"}),
            html.Div(news_children if news_children else [html.P("No recent news.", style={"color": "#7A8BA0"})])]
        )])]) if news else None

    status = html.Div([
        html.P(f"{len(tickers)} assets", style={"color": GOLD, "fontSize": "1.1rem", "fontWeight": "700", "margin": "0"}),
        html.P(f"{len(pr)} days", style={"color": "#7A8BA0", "fontSize": "0.8rem"}),
        html.P("AI: ON" if cmt.get("overview") else "AI: OFF",
               style={"color": "#4CAF50" if cmt.get("overview") else "#EF5350", "fontSize": "0.75rem", "margin": "0"})])

    return kpis, overview, charts, news_section, status


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)