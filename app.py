import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="월별 매출 대시보드",
    layout="wide",
)

# ---------------------------
# 유틸 함수
# ---------------------------
def parse_month_to_period(s):
    # "YYYY-MM" -> pandas Period('YYYY-MM')
    s = str(s).strip()
    try:
        return pd.Period(s, freq="M")
    except Exception:
        return pd.NaT

def moving_avg(series, k=3):
    return series.rolling(window=k, min_periods=k).mean()

def krw(n):
    try:
        return f"{int(n):,} 원"
    except Exception:
        return "-"

# ---------------------------
# 사이드바 - 입력
# ---------------------------
st.sidebar.title("설정")
uploaded = st.sidebar.file_uploader("CSV 업로드 (월, 매출액, 전년동월, 증감률)", type=["csv"])
target = st.sidebar.number_input("연간 매출 목표(원)", min_value=0, value=0, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("※ 헤더명은 반드시 `월, 매출액, 전년동월, 증감률`을 사용하세요.")

st.title("📊 월별 매출 대시보드")

if uploaded is None:
    st.info("좌측 사이드바에서 CSV 파일을 업로드해주세요.")
    st.stop()

# ---------------------------
# 데이터 로드 & 전처리
# ---------------------------
df = pd.read_csv(uploaded)

required_cols = ["월", "매출액", "전년동월", "증감률"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"다음 컬럼이 없습니다: {missing}")
    st.stop()

# 숫자 정제 (콤마/공백 제거)
for c in ["매출액", "전년동월"]:
    df[c] = (
        df[c]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan})
        .astype(float)
    )

# 증감률은 % 기호 제거 후 float
df["증감률"] = (
    df["증감률"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
    .replace({"": np.nan})
    .astype(float)
)

# 월을 Period로 변환 및 정렬
df["월_period"] = df["월"].apply(parse_month_to_period)
if df["월_period"].isna().any():
    st.warning("일부 '월' 값이 YYYY-MM 형식이 아닙니다. 해당 행은 뒤로 밀릴 수 있습니다.")
df = df.sort_values("월_period").reset_index(drop=True)

# 파생 컬럼
df["MA3"] = moving_avg(df["매출액"], 3)
df["MoM"] = df["매출액"].diff()            # 전월 대비 변화액
df["누적매출"] = df["매출액"].cumsum()

# ---------------------------
# KPI 요약
# ---------------------------
ytd_sales = df["누적매출"].iloc[-1] if len(df) else 0
ytd_prev = df["전년동월"].sum(skipna=True)
yoy = ((ytd_sales - ytd_prev) / ytd_prev * 100) if ytd_prev else np.nan

max_idx = df["매출액"].idxmax()
min_idx = df["매출액"].idxmin()
max_label = f"{df.loc[max_idx, '월']} · {krw(df.loc[max_idx, '매출액'])}" if len(df) else "-"
min_label = f"{df.loc[min_idx, '월']} · {krw(df.loc[min_idx, '매출액'])}" if len(df) else "-"

col1, col2, col3, col4 = st.columns(4)
col1.metric("YTD 누적 매출", krw(ytd_sales))
col2.metric("YTD 전년대비 증감률", f"{yoy:.1f} %" if pd.notna(yoy) else "-")
col3.metric("최고 매출 월", max_label)
col4.metric("최저 매출 월", min_label)

st.divider()

# ---------------------------
# ① 매출 추세 & 3M 이동평균
# ---------------------------
fig_trend = go.Figure()
fig_trend.add_trace(
    go.Scatter(
        x=df["월"], y=df["매출액"],
        mode="lines+markers", name="매출액",
        line=dict(width=3)
    )
)
fig_trend.add_trace(
    go.Scatter(
        x=df["월"], y=df["MA3"],
        mode="lines", name="3M 이동평균",
        line=dict(dash="dash")
    )
)
fig_trend.update_layout(
    height=360, margin=dict(t=20, r=20, b=40, l=50),
    xaxis_title="월", yaxis_title="원",
)
st.subheader("① 매출 추세 & 3M 이동평균")
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------
# ② 전년동월 대비 (이중축)
# ---------------------------
fig_yoy = make_subplots(specs=[[{"secondary_y": True}]])
fig_yoy.add_trace(
    go.Scatter(x=df["월"], y=df["매출액"], name="매출액", mode="lines+markers", line=dict(width=3)),
    secondary_y=False
)
fig_yoy.add_trace(
    go.Scatter(x=df["월"], y=df["전년동월"], name="전년동월", mode="lines+markers", line=dict(dash="dot")),
    secondary_y=True
)
fig_yoy.update_yaxes(title_text="매출액(원)", secondary_y=False)
fig_yoy.update_yaxes(title_text="전년동월(원)", secondary_y=True)
fig_yoy.update_layout(height=360, margin=dict(t=20, r=20, b=40, l=50), xaxis_title="월")
st.subheader("② 전년동월 대비")
st.plotly_chart(fig_yoy, use_container_width=True)

# ---------------------------
# ③ 증감률(%) 막대
# ---------------------------
rate_colors = np.where(df["증감률"] >= 0, "#2ca02c", "#d62728")
fig_rate = go.Figure(
    data=[
        go.Bar(
            x=df["월"], y=df["증감률"],
            marker=dict(color=rate_colors),
            hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>",
            name="증감률(%)",
        )
    ]
)
fig_rate.update_layout(height=360, margin=dict(t=20, r=20, b=40, l=50), xaxis_title="월", yaxis_title="증감률(%)")
st.subheader("③ 증감률(%)")
st.plotly_chart(fig_rate, use_container_width=True)

# ---------------------------
# ④ 목표 달성도(게이지)
# ---------------------------
cum_last = df["누적매출"].iloc[-1] if len(df) else 0
pct = min(100, cum_last / target * 100) if target and target > 0 else None

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=pct if pct is not None else 0,
        number={"suffix": "%"},
        title={"text": f"누적/목표 ({krw(cum_last)} / {krw(target)})" if target else "목표 입력 필요"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#6ea8fe"},
            "steps": [
                {"range": [0, 50], "color": "rgba(110,168,254,0.15)"},
                {"range": [50, 80], "color": "rgba(110,168,254,0.25)"},
                {"range": [80, 100], "color": "rgba(110,168,254,0.35)"},
            ],
        },
    )
)
fig_gauge.update_layout(height=360, margin=dict(t=40, r=20, b=20, l=20))
st.subheader("④ 목표 달성도(누적)")
st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------
# ⑤ 월별 기여도(워터폴)
# ---------------------------
if len(df) > 0:
    base = df["매출액"].iloc[0]
    measures = ["absolute"] + ["relative"] * (len(df) - 1)
    values = [base] + df["MoM"].iloc[1:].tolist()

    fig_wf = go.Figure(
        go.Waterfall(
            x=[f"{df['월'].iloc[0]} (기준)"] + df["월"].iloc[1:].tolist(),
            measure=measures,
            y=values,
            decreasing={"marker": {"color": "#d62728"}},
            increasing={"marker": {"color": "#2ca02c"}},
            totals={"marker": {"color": "#6ea8fe"}},
            connector={"line": {"color": "#42507a"}},
        )
    )
    fig_wf.update_layout(height=360, margin=dict(t=20, r=20, b=40, l=50), xaxis_title="월", yaxis_title="원")
else:
    fig_wf = go.Figure()

st.subheader("⑤ 월별 기여도(워터폴)")
st.plotly_chart(fig_wf, use_container_width=True)

# ---------------------------
# 원본 데이터 미리보기
# ---------------------------
with st.expander("원본 데이터 보기"):
    st.dataframe(df[["월", "매출액", "전년동월", "증감률", "MA3", "MoM", "누적매출"]])
