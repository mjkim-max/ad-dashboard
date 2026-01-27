import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 관리 BI", page_icon="📈", layout="wide")

# [핵심] Secrets 설정 없이 주소를 직접 입력 (이 방식이 가장 확실합니다)
# 구글 시트 주소 (edit -> export 변환 로직 적용됨)
META_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1jEB4zTYPb2mrxZGXriju6RymHo1nEMC8QIVzqgiHwdg/edit?gid=141038195#gid=141038195"

# [COLOR] 색상 팔레트
METRIC_COLORS = {
    'Cost': '#D32F2F', 'ROAS': '#6200EA', 'CPM': '#FF6D00', 'CTR': '#00C853',
    'CPA': '#C2185B', 'Conversions': '#009688', 'Clicks': '#2962FF',
    'Impressions': '#FFD600', 'Conversion_Value': '#304FFE',
    'CPC': '#795548'
}

DISTINCT_PALETTE = [
    '#2962FF', '#D50000', '#00C853', '#FFD600', '#AA00FF', 
    '#00E5FF', '#FF6D00', '#304FFE', '#C2185B', '#64DD17',
    '#3D5AFE', '#FFAB00', '#00BFA5', '#D500F9', '#FF1744'
]

# -----------------------------------------------------------------------------
# 0. Session State & Callbacks
# -----------------------------------------------------------------------------
if 'selected_campaign' not in st.session_state:
    st.session_state['selected_campaign'] = '전체'
if 'selected_ad_group' not in st.session_state:
    st.session_state['selected_ad_group'] = '전체'
if 'selected_creatives' not in st.session_state:
    st.session_state['selected_creatives'] = []

def update_filters(campaign, adgroup, creative):
    st.session_state['selected_campaign'] = campaign
    st.session_state['selected_ad_group'] = adgroup
    st.session_state['selected_creatives'] = [creative]

# -----------------------------------------------------------------------------
# 1. 함수 정의 (수정됨: CSV 직접 로드 방식)
# -----------------------------------------------------------------------------
def convert_google_sheet_url(url):
    """구글 시트 URL을 CSV 다운로드 링크로 변환"""
    try:
        # /edit 부분을 /export?format=csv로 변경
        if "/edit" in url:
            base_url = url.split("/edit")[0]
            # gid 파싱
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

    # [핵심 수정] st.connection 대신 pd.read_csv 사용 (Secrets 불필요)
    try:
        csv_url = convert_google_sheet_url(META_SHEET_URL)
        df_meta = pd.read_csv(csv_url)
        df_meta = df_meta.rename(columns=rename_map)
        df_meta['Platform'] = 'Meta'
        if 'Status' not in df_meta.columns: df_meta['Status'] = 'On'
        dfs.append(df_meta)
    except Exception as e:
        # 에러 발생 시 진짜 원인을 화면에 표시 (디버깅용)
        st.error(f"메타 데이터 로드 실패: {e}")

    try:
        csv_url = convert_google_sheet_url(GOOGLE_SHEET_URL)
        df_google = pd.read_csv(csv_url)
        df_google = df_google.rename(columns=rename_map)
        df_google['Platform'] = 'Google'
        if 'Status' not in df_google.columns: df_google['Status'] = 'On'
        dfs.append(df_google)
    except Exception as e:
        st.error(f"구글 데이터 로드 실패: {e}")
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    
    expected_cols = ['Date', 'Platform', 'Campaign', 'AdGroup', 'Creative_ID', 'Cost', 'Impressions', 'Clicks', 'Conversions', 'Conversion_Value', 'Status']
    existing_cols = [c for c in expected_cols if c in df.columns]
    df = df[existing_cols]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    num_cols = ['Cost', 'Impressions', 'Clicks', 'Conversions', 'Conversion_Value']
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').replace('nan', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

def filter_by_recent_days(df, days):
    if df.empty: return df
    max_date = df['Date'].max()
    start_date = max_date - timedelta(days=days-1)
    return df[df['Date'] >= start_date]

def analyze_cpa_performance(df, warning_limit, opportunity_limit):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    creatives = df.groupby(['Campaign', 'AdGroup', 'Creative_ID']).agg({
        'Cost': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
    }).reset_index()
    
    creatives = creatives[creatives['Cost'] > 0]
    creatives['CPA'] = np.where(creatives['Conversions'] > 0, creatives['Cost'] / creatives['Conversions'], np.inf)
    
    warnings = creatives[creatives['CPA'] > warning_limit].sort_values('CPA', ascending=False)
    opportunities = creatives[(creatives['Conversions'] > 0) & (creatives['CPA'] <= opportunity_limit)].sort_values('CPA', ascending=True)
    
    return warnings, opportunities

# -----------------------------------------------------------------------------
# 2. 메인 앱 실행
# -----------------------------------------------------------------------------
df = load_data()
st.title("📊 광고 성과 관리 BI 대시보드")

if df.empty:
    st.warning("데이터를 불러오지 못했습니다. 위의 에러 메시지를 확인해주세요.")
    st.stop()

# =============================================================================
# [SIDEBAR] 필터링
# =============================================================================
st.sidebar.header("🎯 목표 설정")
target_cpa_warning = st.sidebar.number_input("🔴 긴급 점검 기준 CPA (원)", value=100000, step=1000)
target_cpa_opportunity = st.sidebar.number_input("🔵 증액 추천 기준 CPA (원)", value=50000, step=1000)

st.sidebar.divider()
st.sidebar.header("🎛️ 기본 필터")

# 1. 상태 필터
status_options = ["전체"]
if 'Status' in df.columns:
    if 'On' in df['Status'].unique(): status_options.append("게재중 (On)")
    if 'Off' in df['Status'].unique(): status_options.append("비게재 (Off)")

default_idx = 1 if "게재중 (On)" in status_options else 0
status_option = st.sidebar.radio("게재 상태 (Status)", status_options, index=default_idx, horizontal=True)

base_df = df.copy()
if status_option == "게재중 (On)": base_df = base_df[base_df['Status'] == 'On']
elif status_option == "비게재 (Off)": base_df = base_df[base_df['Status'] == 'Off']

if base_df.empty or pd.isna(base_df['Date'].max()):
    st.warning("선택한 상태에 맞는 데이터가 없습니다.")
    st.stop()

# 2. 기간 필터 (빠른 설정)
min_data_date = base_df['Date'].min()
max_data_date = base_df['Date'].max()

preset_options = ["전체 기간", "최근 3일", "최근 7일", "최근 14일", "최근 30일", "이번 달", "지난 달", "최근 90일"]
selected_preset = st.sidebar.selectbox("📅 기간 빠른 설정", preset_options)

if selected_preset == "최근 3일": start_val = max_data_date - timedelta(days=2)
elif selected_preset == "최근 7일": start_val = max_data_date - timedelta(days=6)
elif selected_preset == "최근 14일": start_val = max_data_date - timedelta(days=13)
elif selected_preset == "최근 30일": start_val = max_data_date - timedelta(days=29)
elif selected_preset == "최근 90일": start_val = max_data_date - timedelta(days=89)
elif selected_preset == "이번 달": start_val = max_data_date.replace(day=1)
elif selected_preset == "지난 달": 
    start_val = (max_data_date.replace(day=1) - timedelta(days=1)).replace(day=1)
    max_data_date = max_data_date.replace(day=1) - timedelta(days=1)
else: start_val = min_data_date

if start_val < min_data_date: start_val = min_data_date
end_val = max_data_date

date_range = st.sidebar.date_input("직접 날짜 선택", value=(start_val, end_val), min_value=min_data_date, max_value=max_data_date)

st.sidebar.divider()
st.sidebar.header("🔍 상세 필터")

# 3. 매체 필터 (체크박스)
st.sidebar.markdown("**매체 선택**")
available_platforms = sorted(base_df['Platform'].dropna().unique().tolist())
selected_platforms = []
for platform in available_platforms:
    if st.sidebar.checkbox(platform, value=True, key=f"check_{platform}"):
        selected_platforms.append(platform)

if selected_platforms:
    base_df = base_df[base_df['Platform'].isin(selected_platforms)]
else:
    base_df = base_df[0:0] 

# 4. 캠페인/그룹/소재 필터
chart_df = base_df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    chart_df = chart_df[(chart_df['Date'] >= pd.Timestamp(start_date)) & (chart_df['Date'] <= pd.Timestamp(end_date))]

campaigns = ['전체'] + sorted(chart_df['Campaign'].unique().tolist())
if st.session_state['selected_campaign'] not in campaigns: st.session_state['selected_campaign'] = '전체'
selected_campaign = st.sidebar.selectbox("1단계: 캠페인", campaigns, key='selected_campaign')

ad_groups = ['전체']
if selected_campaign != '전체':
    ad_groups = ['전체'] + sorted(chart_df[chart_df['Campaign'] == selected_campaign]['AdGroup'].unique().tolist())
if st.session_state['selected_ad_group'] not in ad_groups: st.session_state['selected_ad_group'] = '전체'
selected_ad_group = st.sidebar.selectbox("2단계: 광고그룹", ad_groups, disabled=(selected_campaign == '전체'), key='selected_ad_group')

creatives_list = []
if selected_ad_group != '전체':
    creatives_list = sorted(chart_df[chart_df['AdGroup'] == selected_ad_group]['Creative_ID'].unique().tolist())
valid_creatives = [c for c in st.session_state['selected_creatives'] if c in creatives_list]
if len(valid_creatives) != len(st.session_state['selected_creatives']): st.session_state['selected_creatives'] = valid_creatives
selected_creatives = st.sidebar.multiselect("3단계: 광고소재", creatives_list, disabled=(selected_ad_group == '전체'), key='selected_creatives')

target_df = chart_df
if selected_campaign != '전체': target_df = target_df[target_df['Campaign'] == selected_campaign]
if selected_ad_group != '전체': target_df = target_df[target_df['AdGroup'] == selected_ad_group]


# =============================================================================
# [MAIN] 1. 스마트 알림 시스템
# =============================================================================
st.header("🚨 Smart Alert System")

df_3d = filter_by_recent_days(base_df, 3)
df_7d = filter_by_recent_days(base_df, 7)
df_14d = filter_by_recent_days(base_df, 14)

warn_3, opp_3 = analyze_cpa_performance(df_3d, target_cpa_warning, target_cpa_opportunity)
warn_7, opp_7 = analyze_cpa_performance(df_7d, target_cpa_warning, target_cpa_opportunity)
warn_14, opp_14 = analyze_cpa_performance(df_14d, target_cpa_warning, target_cpa_opportunity)

bad_ids_3 = set(warn_3['Creative_ID']) if not warn_3.empty else set()
bad_ids_7 = set(warn_7['Creative_ID']) if not warn_7.empty else set()
bad_ids_14 = set(warn_14['Creative_ID']) if not warn_14.empty else set()

def display_alert_box(data, type='warning', unique_key_prefix='alert'):
    if data.empty:
        msg = "✅ 기준을 초과하는 소재가 없습니다." if type == 'warning' else "⚠️ 기준을 만족하는 소재가 없습니다."
        if type == 'warning': st.success(msg)
        else: st.warning(msg)
        return

    for idx, row in data.iterrows():
        cid = row['Creative_ID']
        cpa_val = "전환 없음" if row['CPA'] == np.inf else f"{row['CPA']:,.0f}원"
        tags = []
        if cid in bad_ids_3: tags.append("(3일)")
        if cid in bad_ids_7: tags.append("(7일)")
        if cid in bad_ids_14: tags.append("(14일)")
        tag_str = " ".join(tags)
        
        with st.container(border=True):
            col_text, col_btn = st.columns([5, 1], gap="small")
            with col_text:
                if type == 'warning': st.markdown(f"**:red[{row['Creative_ID']}]**")
                else: st.markdown(f"**:blue[{row['Creative_ID']}]**")
                st.caption(f"{row['Campaign']} > {row['AdGroup']}")
                if type == 'warning': st.markdown(f"💸 CPA: **{cpa_val}** (비용: {row['Cost']:,.0f}원)")
                else: st.markdown(f"💰 CPA: **{row['CPA']:,.0f}원** (전환: {row['Conversions']:,.0f}건)")
                if tag_str: st.markdown(f"**{tag_str}**")
            with col_btn:
                st.write("") 
                st.write("")
                st.button("🔍 분석", key=f"{unique_key_prefix}_{idx}_{cid}", on_click=update_filters, args=(row['Campaign'], row['AdGroup'], cid), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"🔴 긴급 점검 (> {target_cpa_warning:,}원)")
    tab3, tab7, tab14 = st.tabs(["최근 3일", "최근 7일", "최근 14일"])
    with tab3: st.caption("📉 최근 3일간 성과"); display_alert_box(warn_3, 'warning', 'w3')
    with tab7: st.caption("📉 최근 7일간 성과"); display_alert_box(warn_7, 'warning', 'w7')
    with tab14: st.caption("📉 최근 14일간 성과"); display_alert_box(warn_14, 'warning', 'w14')

with col2:
    st.subheader(f"🔵 증액 추천 (<= {target_cpa_opportunity:,}원)")
    tab3_opp, tab7_opp, tab14_opp = st.tabs(["최근 3일", "최근 7일", "최근 14일"])
    with tab3_opp: st.caption("📈 최근 3일간 성과"); display_alert_box(opp_3, 'opportunity', 'opp3')
    with tab7_opp: st.caption("📈 최근 7일간 성과"); display_alert_box(opp_7, 'opportunity', 'opp7')
    with tab14_opp: st.caption("📈 최근 14일간 성과"); display_alert_box(opp_14, 'opportunity', 'opp14')

st.divider()

# =============================================================================
# [MAIN] 2. 상세 테이블 (인덱스 제거 hide_index=True)
# =============================================================================
if selected_campaign == '전체': lv, gc, ds = "캠페인", "Campaign", target_df
elif selected_ad_group == '전체': lv, gc, ds = "광고그룹", "AdGroup", target_df[target_df['Campaign']==selected_campaign]
elif not selected_creatives: lv, gc, ds = "광고소재", "Creative_ID", target_df[target_df['AdGroup']==selected_ad_group]
else: lv, gc, ds = "선택 소재", "Creative_ID", target_df[target_df['Creative_ID'].isin(selected_creatives)]

st.header(f"📋 {lv}별 상세 성과")

# 집계 기준 (공유)
resample_option = st.radio("집계 기준", ["일별", "3일", "주별", "월별"], horizontal=True)
resample_map = {"일별": "D", "3일": "3D", "주별": "W", "월별": "ME"}

# 집계 및 정렬 (기본: Cost 내림차순)
summary_df = ds.groupby(gc).agg({'Cost': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum', 'Clicks': 'sum', 'Impressions': 'sum'})
sorted_items = summary_df.sort_values('Cost', ascending=False).index.tolist()

for idx, item in enumerate(sorted_items):
    with st.expander(f"📄 {item}", expanded=(idx==0)):
        res = ds[ds[gc]==item].set_index('Date').groupby(pd.Grouper(freq=resample_map[resample_option])).agg({
            'Cost': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
        }).reset_index().sort_values('Date', ascending=False)
        
        while not res.empty and res.iloc[0][['Cost', 'Impressions']].sum() == 0:
             res = res.iloc[1:]

        total = pd.DataFrame([res.sum(numeric_only=True)])
        total['Date'] = '📊 기간 합계'
        final = pd.concat([total, res], ignore_index=True)
        final['ROAS'] = (final['Conversion_Value']/final['Cost']*100).fillna(0)
        final['CPA'] = (final['Cost']/final['Conversions']).fillna(0)
        final['CTR'] = (final['Clicks']/final['Impressions']*100).fillna(0)
        final['CPM'] = (final['Cost']/final['Impressions']*1000).fillna(0)
        final['CPC'] = (final['Cost']/final['Clicks']).replace([np.inf, -np.inf], 0).fillna(0)
        final['Date'] = final['Date'].astype(str).str[:10]
        
        cols_order = ['Date', 'CPA', 'Cost', 'Impressions', 'Clicks', 'Conversions', 'Conversion_Value', 'CPM', 'CPC', 'CTR', 'ROAS']
        cols_order = [c for c in cols_order if c in final.columns]
        final = final[cols_order]

        # [핵심] hide_index=True 추가
        st.dataframe(final.style.format({
            'CPA':'{:,.0f}', 'Cost':'{:,.0f}', 'Impressions':'{:,.0f}', 'Clicks':'{:,.0f}',
            'Conversions':'{:,.0f}', 'Conversion_Value':'{:,.0f}', 'CPM':'{:,.0f}', 'CPC':'{:,.0f}',
            'CTR':'{:.2f}%', 'ROAS':'{:.1f}%'
        }), use_container_width=True, hide_index=True)

st.divider()

# =============================================================================
# [MAIN] 3. 성과 추이 그래프
# =============================================================================
st.header("📈 성과 추이 그래프")

c1, c2, c3 = st.columns([2, 1, 1])
metrics_to_show = c1.multiselect("Y축 지표", ["ROAS", "CPM", "CPC", "CTR", "CPA", "Cost", "Conversions", "Clicks", "Impressions"], default=["CPM", "CTR", "Cost"])
show_values = c2.checkbox("☑️ 데이터 값 표시 (k단위/%)", value=False)
chart_style = c3.radio("스타일", ["선", "영역"], horizontal=True)

if not target_df.empty:
    fig = go.Figure()
    plot_items = []
    
    if selected_creatives:
        color_idx = 0
        for creative in selected_creatives:
            c_df = target_df[target_df['Creative_ID'] == creative].copy().set_index('Date')
            resampled = c_df.groupby(pd.Grouper(freq=resample_map[resample_option])).agg({
                'Cost': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
            }).reset_index()
            
            while not resampled.empty:
                last_row = resampled.iloc[-1]
                if last_row[['Cost', 'Impressions']].sum() == 0:
                    resampled = resampled.iloc[:-1]
                else:
                    break

            resampled['ROAS'] = (resampled['Conversion_Value'] / resampled['Cost'] * 100).fillna(0)
            resampled['CPA'] = (resampled['Cost'] / resampled['Conversions']).fillna(0)
            resampled['CTR'] = (resampled['Clicks'] / resampled['Impressions'] * 100).fillna(0)
            resampled['CPM'] = (resampled['Cost'] / resampled['Impressions'] * 1000).fillna(0)
            resampled['CPC'] = (resampled['Cost'] / resampled['Clicks']).replace([np.inf, -np.inf], 0).fillna(0)
            
            for metric in metrics_to_show:
                plot_items.append({
                    'x': resampled['Date'], 'y': resampled[metric], 
                    'name': f"{metric} - {creative}", 'metric_type': metric,
                    'color': DISTINCT_PALETTE[color_idx % len(DISTINCT_PALETTE)]
                })
                color_idx += 1
    else:
        agg_df = target_df.copy().set_index('Date')
        resampled = agg_df.groupby(pd.Grouper(freq=resample_map[resample_option])).agg({
            'Cost': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
        }).reset_index()
        
        while not resampled.empty:
            last_row = resampled.iloc[-1]
            if last_row[['Cost', 'Impressions']].sum() == 0:
                resampled = resampled.iloc[:-1]
            else:
                break

        resampled['ROAS'] = (resampled['Conversion_Value'] / resampled['Cost'] * 100).fillna(0)
        resampled['CPA'] = (resampled['Cost'] / resampled['Conversions']).fillna(0)
        resampled['CTR'] = (resampled['Clicks'] / resampled['Impressions'] * 100).fillna(0)
        resampled['CPM'] = (resampled['Cost'] / resampled['Impressions'] * 1000).fillna(0)
        resampled['CPC'] = (resampled['Cost'] / resampled['Clicks']).replace([np.inf, -np.inf], 0).fillna(0)
        
        for metric in metrics_to_show:
            plot_items.append({
                'x': resampled['Date'], 'y': resampled[metric], 
                'name': metric, 'metric_type': metric,
                'color': METRIC_COLORS.get(metric, '#000')
            })

    for item in plot_items:
        real_y = item['y']
        norm_y = real_y.copy()
        
        if real_y.max() > 0:
            norm_y = (real_y - real_y.min()) / (real_y.max() - real_y.min()) * 100
        
        text_vals = []
        if show_values:
            for val in real_y:
                if val == 0: text_vals.append("")
                elif item['metric_type'] in ['Cost', 'Conversion_Value']:
                    text_vals.append(f"{val/1000:,.0f}k")
                elif item['metric_type'] == 'CTR':
                    text_vals.append(f"{val:.2f}%")
                elif item['metric_type'] == 'ROAS':
                    text_vals.append(f"{val:.0f}%")
                else:
                    text_vals.append(f"{val:,.0f}")

        if item['metric_type'] in ['Cost', 'CPA', 'CPM', 'CPC', 'Conversion_Value', 'Impressions', 'Clicks', 'Conversions']:
            hover_str = "%{customdata:,.0f}"
        elif item['metric_type'] == 'CTR':
            hover_str = "%{customdata:.2f}%"
        else:
            hover_str = "%{customdata:.2f}"

        fig.add_trace(go.Scatter(
            x=item['x'], y=norm_y, customdata=real_y,
            mode='lines+markers+text' if show_values else 'lines+markers',
            text=text_vals if show_values else None,
            textposition="top center",
            textfont=dict(color=item['color'], size=11, family="Arial Black"),
            name=item['name'], 
            line=dict(width=3, color=item['color']), 
            marker=dict(size=6 if show_values else 8), 
            fill='tozeroy' if chart_style=="영역" else 'none',
            hovertemplate=hover_str
        ))
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F0F0F0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F0F0F0', showticklabels=False)
    
    fig.update_layout(
        height=500, 
        hovermode='x unified',
        plot_bgcolor='white',
        font=dict(size=14, color="#333"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=13))
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("조건에 맞는 데이터가 없습니다.")