# Portfolio Analytics & Risk Dashboard

AI-powered portfolio analysis platform built with Dash and Groq. Enter any stock/ETF portfolio and get comprehensive risk metrics, Monte Carlo simulation, per-chart AI commentary connecting performance to current market news, and a full market intelligence feed.

![Portfolio Dashboard](screenshots/dashboard.png)
<!-- Replace with your actual screenshot -->

## Live Demo

[**View the live dashboard**](YOUR_ECS_URL_HERE)

## Features

| Feature | Description |
|---------|-------------|
| **KPI Metrics** | Total return, alpha, Sharpe ratio, beta, max drawdown, VaR, volatility |
| **Cumulative Returns** | Portfolio vs benchmark (SPY) with interactive chart |
| **Individual Holdings** | Per-stock performance comparison |
| **Monte Carlo Simulation** | 1,000+ scenario projections with confidence bands |
| **Return Distribution** | Histogram with 95% VaR line |
| **Correlation Matrix** | Asset correlation heatmap for diversification analysis |
| **Drawdown Analysis** | Peak-to-trough decline visualization |
| **AI Commentary** | Per-chart explanations powered by Groq connecting data to current events |
| **Market News** | Live news feed with article previews for each holding |
| **Holdings Table** | Weight, return, volatility, and Sharpe per asset |

## AI-Powered Analysis

Each chart includes AI-generated commentary from Groq (Llama 3.3 70B) that explains what the data means in the context of current market news — like having a Goldman Sachs analyst write you a personal portfolio briefing.

## Tech Stack

**Framework:** Dash (Flask + React + Plotly) — NOT Streamlit. Grid-based layout used by hedge funds and trading desks.

**Analytics:** NumPy, pandas — Sharpe ratio, beta/alpha (CAPM), Value at Risk, Monte Carlo simulation

**AI:** Groq (Llama 3.3 70B) — per-chart market commentary

**Data:** Yahoo Finance (free, real-time market data)

**Deployment:** Docker, Amazon ECR, Amazon ECS Fargate

## Quick Start

```bash
git clone https://github.com/aneeketdaswani1/portfolio-dashboard.git
cd portfolio-dashboard
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env (free at console.groq.com)
python app.py
# Open http://localhost:8050
```

## Deploy to AWS

```bash
docker buildx build --platform linux/amd64 --load -t portfolio-dashboard .
docker tag portfolio-dashboard:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/portfolio-dashboard:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/portfolio-dashboard:latest
# Create ECS Express Mode service with port 8050
```

## Project Structure
```
├── app.py              # Dash dashboard + Groq AI + all analytics
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container config (port 8050)
├── .env.example        # API key template
├── .gitignore          # Excludes .env from repo
└── README.md
```

## License

MIT
