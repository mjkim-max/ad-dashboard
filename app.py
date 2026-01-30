import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 관리 BI", page_icon=None, layout="wide")

# [주소 설정]
META_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1jEB4zTYPb2mrxZGXriju6RymHo1nEMC8QIVzqgiHwdg/edit?gid=141038195#gid=141038195"

# [세션 상태 초기화]
if 'chart_target_creative' not in st.session_state:
    st.session_state['chart_target_creative'] = None

# -----------------------------------------------------------------------------
# 1. 데이터 로드
# -----------------------------------------------------------------------------
def convert_google_sheet_url(url):
    try:
        if "/edit" in url:
            base_url = url.split("/edit")[0]
            if "gid=" in url:
                gid = url.split("gid=")[1].split("#")[0]
                return f"{base_url}/export?format=csv&gid={gid}"
        return url
    except:
        return url

@st.cache_data(ttl=600)
def load_data():
    dfs = []
    rename_map = {
        '일': 'Date', '날짜': 'Date',
        '캠페인 이름': 'Campaign', '캠페인': 'Campaign',
        '광고 세트 이름': 'AdGroup', '광고 그룹 이름': 'AdGroup', '광고 그룹': 'AdGroup',
        '광고 이름': 'Creative_ID', '소재 이름': 'Creative_ID', '소재': 'Creative_ID',
        '지출 금액 (KRW)': 'Cost', '비용': 'Cost', '지출': 'Cost',
        '노출': 'Impressions', '노출수': 'Impressions',
        '링크 클릭': 'Clicks', '클릭': 'Clicks', '클릭수': 'Clicks',
        '구매': 'Conversions', '전환': 'Conversions', '전환수': 'Conversions',
        '구매 전환값': 'Conversion_Value', '전환 가치': 'Conversion_Value', '전환값': 'Conversion_Value',
        '상태': 'Status', '소재 상태': 'Status', '광고 상태': 'Status'
    }

    try:
        df_meta = pd.read_csv(convert_google_sheet_url(META_SHEET_URL)).rename(columns=rename_map)
        df_meta['Platform'] = 'Meta'
        if 'Status' not in df_meta.columns: df_meta['Status'] = 'On'
        dfs.append(df_meta)
    except: pass

    try:
        df_google = pd.read_csv(convert_google_sheet_url(GOOGLE_SHEET_URL)).rename(columns=rename_map)
        df_google['Platform'] = 'Google'
        if 'Status' not in df_google.columns: df_google['Status'] = 'On'
        dfs.append(df_google)
    except: pass
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    num_cols = ['Cost', 'Impressions', 'Clicks', 'Conversions', 'Conversion_Value']
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').replace('nan', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# -----------------------------------------------------------------------------
# 2. 진단 로직 (절대평가 Only)
# -----------------------------------------------------------------------------
def get_stats_for_period(df, days):
    max_date = df['Date'].max()
    start_date = max_date - timedelta(days=days-1)
    filtered = df[df['Date'] >= start_date]
    stats = filtered.groupby(['Campaign', 'AdGroup', 'Creative_ID']).agg({
        'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum', 'Clicks': 'sum'
    }).reset_index()
    stats['CPA'] = np.where(stats['Conversions']>0, stats['Cost']/stats['Conversions'], np.inf)
    return stats

def run_diagnosis(df, target_cpa):
    if df.empty: return pd.DataFrame()
    s3, s7, s14 = get_stats_for_period(df, 3), get_stats_for_period(df, 7), get_stats_for_period(df, 14)
    s_all = get_stats_for_period(df, 9999) 

    m = s3.merge(s7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3', '_7'), how='left')
    m = m.merge(s14, on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.rename(columns={'CPA': 'CPA_14', 'Cost': 'Cost_14', 'Conversions': 'Conversions_14'})
    m = m.merge(s_all[['Campaign','AdGroup','Creative_ID']], on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.fillna(0)
    for col in ['CPA_3', 'CPA_7', 'CPA_14']: m[col] = m[col].replace(0, np.inf)

    results = []

    for _, row in m.iterrows():
        if row['Cost_3'] < 3000: continue
        
        cpa3, cpa7, cpa14 = row['CPA_3'], row['CPA_7'], row['CPA_14']
        status, title, detail = "White", "", ""

        # [Logic: 목표 CPA 기준 절대평가]
        if (cpa14 <= target_cpa) and (cpa7 <= target_cpa) and (cpa3 <= target_cpa):
            status = "Blue"; title = "성과 우수 (Best)"; detail = "14일/7일/3일 모두 목표 달성."
        elif (cpa14 > target_cpa) and (cpa7 > target_cpa) and (cpa3 > target_cpa):
            status = "Red"; title = "종료 추천 (지속 부진)"; detail = "14일/7일/3일 모두 목표 미달성."
        else:
            status = "Yellow"
            if cpa3 <= target_cpa: title = "성장 가능성 (반등)"; detail = "과거엔 목표 초과했으나, 최근 3일은 목표 달성."
            else: title = "관망 필요 (최근 저하)"; detail = "과거엔 좋았으나, 최근 3일은 목표 초과."

        row['Status_Color'] = status; row['Diag_Title'] = title; row['Diag_Detail'] = detail
        results.append(row)
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 사이드바
# -----------------------------------------------------------------------------
df_raw = load_data()

st.sidebar.header("목표 설정")
target_cpa_warning = st.sidebar.number_input("목표 CPA", value=100000, step=1000)
target_cpa_opportunity = st.sidebar.number_input("증액추천 CPA", value=50000, step=1000)
st.sidebar.markdown("---")

st.sidebar.header("기간 설정")
preset = st.sidebar.selectbox("기간선택", ["오늘", "어제", "최근 3일", "최근 7일", "최근 14일", "최근 30일", "이번 달", "지난 달", "최근 90일"])
today = datetime.now().date()
if preset == "오늘": s, e = today, today
elif preset == "어제": s = today - timedelta(days=1); e = s
elif preset == "최근 3일": s = today - timedelta(days=2); e = today
elif preset == "최근 7일": s = today - timedelta(days=6); e = today
elif preset == "최근 14일": s = today - timedelta(days=13); e = today
elif preset == "최근 30일": s = today - timedelta(days=29); e = today
elif preset == "최근 90일": s = today - timedelta(days=89); e = today
elif preset == "이번 달": s = date(today.year, today.month, 1); e = today
elif preset == "지난 달": 
    first = date(today.year, today.month, 1); e = first - timedelta(days=1); s = date(e.year, e.month, 1)
date_range = st.sidebar.date_input("날짜범위", [s, e])
st.sidebar.markdown("---")

st.sidebar.header("필터 설정")
st.sidebar.write("매체선택")
c_m, c_g = st.sidebar.columns(2)
sel_pl = []
if c_m.checkbox("Meta", True): sel_pl.append("Meta")
if c_g.checkbox("Google", True): sel_pl.append("Google")
if 'Platform' in df_raw.columns: df_raw = df_raw[df_raw['Platform'].isin(sel_pl)]

df_filtered = df_raw.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['Date'].dt.date >= date_range[0]) & (df_filtered['Date'].dt.date <= date_range[1])]

camps = ['전체'] + sorted(df_filtered['Campaign'].unique().tolist())
sel_camp = st.sidebar.selectbox("캠페인필터", camps)

grps = ['전체']
if sel_camp != '전체': grps = ['전체'] + sorted(df_filtered[df_filtered['Campaign'] == sel_camp]['AdGroup'].unique().tolist())
sel_grp = st.sidebar.selectbox("광고그룹필터", grps)

crvs = []
if sel_grp != '전체': crvs = sorted(df_filtered[df_filtered['AdGroup'] == sel_grp]['Creative_ID'].unique().tolist())
sel_crv = st.sidebar.multiselect("광고소재필터", crvs)

status_opt = st.sidebar.radio("게재상태", ["전체", "게재중 (On)", "비게재 (Off)"], index=1)
if 'Status' in df_filtered.columns:
    if status_opt == "게재중 (On)": df_filtered = df_filtered[df_filtered['Status'] == 'On']
    elif status_opt == "비게재 (Off)": df_filtered = df_filtered[df_filtered['Status'] == 'Off']

target_df = df_filtered.copy()
if sel_camp != '전체': target_df = target_df[target_df['Campaign'] == sel_camp]
if sel_grp != '전체': target_df = target_df[target_df['AdGroup'] == sel_grp]
if sel_crv: target_df = target_df[target_df['Creative_ID'].isin(sel_crv)]

# -----------------------------------------------------------------------------
# 4. 메인 화면: 진단 리포트
# -----------------------------------------------------------------------------
st.title("광고 성과 관리 대시보드")
st.subheader("1. 캠페인 성과 진단")

diag_base = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - timedelta(days=14))]
diag_res = run_diagnosis(diag_base, target_cpa_warning)

def get_color_box(color):
    if color == "Red": return st.error("종료 추천", icon=None)
    elif color == "Yellow": return st.warning("판별 필요", icon=None)
    elif color == "Blue": return st.info("성과 우수", icon=None)
    else: return st.container(border=True)

if not diag_res.empty:
    camp_grps = diag_res.groupby('Campaign')
    sorted_camps = []
    
    for c_name, grp in camp_grps:
        has_red = 'Red' in grp['Status_Color'].values
        has_yellow = 'Yellow' in grp['Status_Color'].values
        
        if has_red: prio = 1; h_col = ":red"
        elif has_yellow: prio = 2; h_col = ":orange"
        else: prio = 3; h_col = ":blue"
        
        c3 = grp['Cost_3'].sum(); cv3 = grp['Conversions_3'].sum()
        cpa3 = c3 / cv3 if cv3 > 0 else 0
        c7 = grp['Cost_7'].sum(); cv7 = grp['Conversions_7'].sum()
        cpa7 = c7 / cv7 if cv7 > 0 else 0
        c14 = grp['Cost_14'].sum(); cv14 = grp['Conversions_14'].sum()
        cpa14 = c14 / cv14 if cv14 > 0 else 0

        sorted_camps.append({
            'name': c_name, 'data': grp, 'prio': prio, 'header': c_name, 'color': h_col,
            'stats_3': (cpa3, c3, cv3),
            'stats_7': (cpa7, c7, cv7),
            'stats_14': (cpa14, c14, cv14)
        })
    
    sorted_camps.sort(key=lambda x: x['prio'])

    for item in sorted_camps:
        if sel_camp != '전체' and item['name'] != sel_camp: continue
        
        with st.expander(f"{item['color']}[{item['header']}]", expanded=False):
            
            # 캠페인 요약
            st.markdown("##### 📊 캠페인 기간별 성과 요약")
            c_3d, c_7d, c_14d = st.columns(3)
            with c_3d:
                st.markdown("**📅 최근 3일**")
                cpa, cost, conv = item['stats_3']
                st.metric("CPA", f"{cpa:,.0f}원")
                st.caption(f"비용: {cost/10000:,.1f}만 / 전환: {conv:,.0f}")
            with c_7d:
                st.markdown("**📅 최근 7일**")
                cpa, cost, conv = item['stats_7']
                st.metric("CPA", f"{cpa:,.0f}원")
                st.caption(f"비용: {cost/10000:,.1f}만 / 전환: {conv:,.0f}")
            with c_14d:
                st.markdown("**📅 최근 14일**")
                cpa, cost, conv = item['stats_14']
                st.metric("CPA", f"{cpa:,.0f}원")
                st.caption(f"비용: {cost/10000:,.1f}만 / 전환: {conv:,.0f}")
            
            st.divider()

            # 소재별 진단
            st.markdown("##### 📂 소재별 진단")
            for _, r in item['data'].iterrows():
                with get_color_box(r['Status_Color']):
                    # 레이아웃: [데이터(2.5) | 진단(1.0) | 버튼(0.5)]
                    c1, c2, c3 = st.columns([2.5, 1.0, 0.5])
                    
                    with c1:
                        st.markdown(f"**{r['Creative_ID']}**")
                        # 3일/7일/14일 데이터를 모두 동일한 폰트 사이즈(markdown)로 표시
                        # 데이터 형식: 3일: CPA [X] / 비용 X / 전환 X
                        
                        def fmt_line(label, cpa, cost, conv):
                            cpa_val = "∞" if cpa == np.inf else f"{cpa:,.0f}"
                            return f"**{label}:** CPA [{cpa_val}원] / 비용 {cost:,.0f}원 / 전환 {conv:,.0f}"

                        st.markdown(fmt_line("3일", r['CPA_3'], r['Cost_3'], r['Conversions_3']))
                        st.markdown(fmt_line("7일", r['CPA_7'], r['Cost_7'], r['Conversions_7']))
                        st.markdown(fmt_line("14일", r['CPA_14'], r['Cost_14'], r['Conversions_14']))
                        
                    with c2:
                        t_col = "red" if r['Status_Color']=="Red" else "blue" if r['Status_Color']=="Blue" else "orange"
                        st.markdown(f":{t_col}[**{r['Diag_Title']}**]")
                        st.caption(r['Diag_Detail'])
                    
                    with c3:
                        unique_key = f"btn_{item['name']}_{r['AdGroup']}_{r['Creative_ID']}"
                        if st.button("분석하기", key=unique_key):
                            st.session_state['chart_target_creative'] = r['Creative_ID']
                            st.rerun()
else:
    st.info("진단 데이터 부족")

# -----------------------------------------------------------------------------
# 5. 추세 그래프 & 상세 표
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. 지표별 추세 및 상세 분석")

target_creative = st.session_state['chart_target_creative']
chart_data = target_df.copy()

if target_creative:
    st.info(f"🔎 현재 **'{target_creative}'** 소재를 집중 분석 중입니다. (설정된 기간: {date_range[0]} ~ {date_range[1]})")
    chart_data = df_filtered[df_filtered['Creative_ID'] == target_creative]
    if st.button("전체 목록으로 차트 초기화"):
        st.session_state['chart_target_creative'] = None
        st.rerun()

c_freq, c_opts, c_norm = st.columns([1, 2, 1])

freq_option = c_freq.radio("집계 기준", ["1일", "3일", "7일"], horizontal=True)
freq_map = {"1일": "D", "3일": "3D", "7일": "W"}

metrics = c_opts.multiselect(
    "지표 선택", 
    ['Impressions', 'Clicks', 'CTR', 'CPM', 'CPC', 'CPA', 'Cost', 'Conversions', 'CVR', 'ROAS'], 
    default=['Impressions', 'CTR', 'CPM']
)
use_norm = c_norm.checkbox("데이터 정규화 (0-100%)", value=True)

if not chart_data.empty and metrics:
    agg_df = chart_data.set_index('Date').groupby(pd.Grouper(freq=freq_map[freq_option])).agg({
        'Cost': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
    }).reset_index().sort_values('Date', ascending=False)

    agg_df['CPA'] = np.where(agg_df['Conversions']>0, agg_df['Cost']/agg_df['Conversions'], 0)
    agg_df['CPM'] = np.where(agg_df['Impressions']>0, agg_df['Cost']/agg_df['Impressions']*1000, 0)
    agg_df['CTR'] = np.where(agg_df['Impressions']>0, agg_df['Clicks']/agg_df['Impressions']*100, 0)
    agg_df['CPC'] = np.where(agg_df['Clicks']>0, agg_df['Cost']/agg_df['Clicks'], 0)
    agg_df['CVR'] = np.where(agg_df['Clicks']>0, agg_df['Conversions']/agg_df['Clicks']*100, 0)
    agg_df['ROAS'] = np.where(agg_df['Cost']>0, agg_df['Conversion_Value']/agg_df['Cost']*100, 0)

    plot_df = agg_df.sort_values('Date', ascending=True)
    fig = go.Figure()
    
    for m in metrics:
        y_data = plot_df[m]
        if use_norm and y_data.max() > 0:
            y_plot = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100
            hover_temp = f"{m}: %{{customdata:,.2f}}"
        else:
            y_plot = y_data
            hover_temp = f"{m}: %{{y:,.2f}}"

        fig.add_trace(go.Scatter(
            x=plot_df['Date'], y=y_plot, mode='lines+markers', name=m,
            customdata=y_data, hovertemplate=hover_temp
        ))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', tickformat="%m-%d")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_layout(height=450, hovermode='x unified', title=f"추세 분석 ({freq_option} 기준)", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 상세 데이터")
    display_cols = ['Date', 'CPA', 'Cost', 'Impressions', 'CPM', 'Clicks', 'Conversions', 'CTR', 'CPC', 'CVR', 'ROAS']
    table_df = agg_df[display_cols].copy()
    table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')
    table_df.columns = ['날짜', 'CPA', '비용', '노출', 'CPM', '클릭', '전환', '클릭률', 'CPC', '전환율', 'ROAS']

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "날짜": st.column_config.TextColumn("날짜"),
            "CPA": st.column_config.NumberColumn("CPA", format="%d원"),
            "비용": st.column_config.NumberColumn("비용", format="%d원"),
            "노출": st.column_config.NumberColumn("노출", format="%d"),
            "CPM": st.column_config.NumberColumn("CPM", format="%d원"),
            "클릭": st.column_config.NumberColumn("클릭", format="%d"),
            "전환": st.column_config.NumberColumn("전환", format="%d"),
            "클릭률": st.column_config.NumberColumn("클릭률", format="%.2f%%"),
            "CPC": st.column_config.NumberColumn("CPC", format="%d원"),
            "전환율": st.column_config.NumberColumn("전환율", format="%.2f%%"),
            "ROAS": st.column_config.NumberColumn("ROAS", format="%.0f%%"),
        }
    )
else:
    st.warning("설정된 기간 내에 데이터가 없습니다.")