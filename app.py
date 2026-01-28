import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 관리 BI", page_icon=None, layout="wide")

# [주소 설정]
META_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1jEB4zTYPb2mrxZGXriju6RymHo1nEMC8QIVzqgiHwdg/edit?gid=141038195#gid=141038195"

# [색상 팔레트]
METRIC_COLORS = {
    'Cost': '#D32F2F', 'ROAS': '#6200EA', 'CPM': '#FF6D00', 'CTR': '#00C853',
    'CPA': '#C2185B', 'Conversions': '#009688', 'Clicks': '#2962FF',
    'Impressions': '#FFD600', 'Conversion_Value': '#304FFE',
    'CPC': '#795548'
}
DISTINCT_PALETTE = ['#2962FF', '#D50000', '#00C853', '#FFD600', '#AA00FF', '#00E5FF', '#FF6D00', '#304FFE']

# -----------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리
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
        csv_url = convert_google_sheet_url(META_SHEET_URL)
        df_meta = pd.read_csv(csv_url)
        df_meta = df_meta.rename(columns=rename_map)
        df_meta['Platform'] = 'Meta'
        if 'Status' not in df_meta.columns: df_meta['Status'] = 'On'
        dfs.append(df_meta)
    except: pass

    try:
        csv_url = convert_google_sheet_url(GOOGLE_SHEET_URL)
        df_google = pd.read_csv(csv_url)
        df_google = df_google.rename(columns=rename_map)
        df_google['Platform'] = 'Google'
        if 'Status' not in df_google.columns: df_google['Status'] = 'On'
        dfs.append(df_google)
    except: pass
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    num_cols = ['Cost', 'Impressions', 'Clicks', 'Conversions', 'Conversion_Value']
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').replace('nan', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df

# -----------------------------------------------------------------------------
# 2. 진단 로직 함수 (14-7-3일 분석)
# -----------------------------------------------------------------------------
def get_stats_for_period(df, days):
    max_date = df['Date'].max()
    start_date = max_date - timedelta(days=days-1)
    filtered = df[df['Date'] >= start_date]
    stats = filtered.groupby(['Campaign', 'AdGroup', 'Creative_ID']).agg({
        'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum', 'Clicks': 'sum'
    }).reset_index()
    stats['CPA'] = np.where(stats['Conversions']>0, stats['Cost']/stats['Conversions'], np.inf)
    stats['CPM'] = np.where(stats['Impressions']>0, stats['Cost']/stats['Impressions']*1000, 0)
    stats['CTR'] = np.where(stats['Impressions']>0, stats['Clicks']/stats['Impressions']*100, 0)
    return stats

def run_diagnosis(df, target_cpa):
    if df.empty: return pd.DataFrame()
    
    # 기간별 통계
    s3 = get_stats_for_period(df, 3)
    s7 = get_stats_for_period(df, 7)
    s14 = get_stats_for_period(df, 14)
    s_all = get_stats_for_period(df, 9999) # 전체

    # 병합
    m = s3.merge(s7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3', '_7'), how='left')
    m = m.merge(s14, on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.rename(columns={'CPA': 'CPA_14', 'Cost': 'Cost_14', 'Conversions': 'Conversions_14'})
    m = m.merge(s_all[['Campaign','AdGroup','Creative_ID']], on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.fillna(0)

    # Infinity 처리
    for col in ['CPA_3', 'CPA_7', 'CPA_14']:
        m[col] = m[col].replace(0, np.inf)

    results = []
    # 캠페인 Best CPA (상대평가용)
    camp_best = m[m['Conversions_14'] > 0].groupby('Campaign')['CPA_14'].min().to_dict()

    for _, row in m.iterrows():
        # 비용 3000원 미만 제외
        if row['Cost_3'] < 3000: continue

        cpa3, cpa7, cpa14 = row['CPA_3'], row['CPA_7'], row['CPA_14']
        best = camp_best.get(row['Campaign'], 99999999)
        
        status, title, detail = "White", "", ""

        # 로직 적용
        # 1. 상대평가 (에이스 독주)
        if (cpa3 > target_cpa) and (best <= target_cpa * 0.9):
            status = "Red"
            title = "[종료 추천] 상대적 열위"
            detail = f"캠페인 Best(CPA {best:,.0f}원) 대비 저조."
        
        # 2. 타겟 확장 신호 (보류)
        elif (cpa7 <= target_cpa * 1.2) and (cpa3 > target_cpa) and (row['CPM_3'] < row['CPM_7']*0.9) and (row['CTR_3'] < row['CTR_7']*0.9):
            status = "Yellow"
            title = "[보류] 타겟 탐색 신호"
            detail = "CPM/CTR 동반 하락. 저가 입찰 탐색 중."
            
        # 3. 절대평가 (지속 부진)
        elif (cpa14 > target_cpa) and (cpa7 > target_cpa) and (cpa3 > target_cpa):
            status = "Red"
            title = "[효율 저조] 지속 부진"
            detail = "2주간 목표 미달성."

        # 4. 성과 개선
        elif (cpa7 > target_cpa) and (cpa3 <= target_cpa):
            status = "Green" # 시각적으론 Green, 정렬순위 고려
            title = "[성과 개선] 반등 중"
            detail = "최근 효율 개선됨."

        # 5. 성과 우수
        elif (cpa3 <= target_cpa) and (cpa7 <= target_cpa):
            status = "Blue"
            title = "[성과 우수] Best"
            detail = "목표 달성 중. 증액 검토."

        # 6. 단순 흔들림
        elif (cpa7 <= target_cpa) and (cpa3 > target_cpa):
            status = "Yellow"
            title = "[주의] 최근 흔들림"
            detail = "일시적 저하인지 확인."

        row['Status_Color'] = status
        row['Diag_Title'] = title
        row['Diag_Detail'] = detail
        results.append(row)

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 메인 앱 실행
# -----------------------------------------------------------------------------
df_raw = load_data()

# [SIDEBAR] 필터 복구
st.sidebar.header("설정 및 필터")

# 1. CPA 목표 설정
target_cpa_warning = st.sidebar.number_input("🔴 목표 CPA (점검 기준)", value=100000, step=1000)
target_cpa_opportunity = st.sidebar.number_input("🔵 증액 추천 CPA 기준", value=50000, step=1000)

st.sidebar.divider()

# 2. 매체 및 상태 필터
status_filter = st.sidebar.radio("게재 상태", ["전체", "게재중 (On)", "비게재 (Off)"], index=1)
if 'Status' in df_raw.columns:
    if status_filter == "게재중 (On)": df_raw = df_raw[df_raw['Status'] == 'On']
    elif status_filter == "비게재 (Off)": df_raw = df_raw[df_raw['Status'] == 'Off']

platforms = sorted(df_raw['Platform'].unique())
selected_platforms = st.sidebar.multiselect("매체 선택", platforms, default=platforms)
if selected_platforms:
    df_raw = df_raw[df_raw['Platform'].isin(selected_platforms)]

# 3. 날짜 선택
min_date = df_raw['Date'].min()
max_date = df_raw['Date'].max()
date_range = st.sidebar.date_input("날짜 범위", [min_date, max_date], min_value=min_date, max_value=max_date)

# 4. 캠페인/그룹/소재 필터
df_filtered = df_raw.copy()
if len(date_range) == 2:
    s_date, e_date = date_range
    df_filtered = df_filtered[(df_filtered['Date'] >= pd.Timestamp(s_date)) & (df_filtered['Date'] <= pd.Timestamp(e_date))]

campaigns = ['전체'] + sorted(df_filtered['Campaign'].unique().tolist())
sel_camp = st.sidebar.selectbox("캠페인", campaigns)

adgroups = ['전체']
if sel_camp != '전체':
    adgroups = ['전체'] + sorted(df_filtered[df_filtered['Campaign'] == sel_camp]['AdGroup'].unique().tolist())
sel_adgroup = st.sidebar.selectbox("광고그룹", adgroups)

creatives = []
if sel_adgroup != '전체':
    creatives = sorted(df_filtered[df_filtered['AdGroup'] == sel_adgroup]['Creative_ID'].unique().tolist())
sel_creative = st.sidebar.multiselect("소재", creatives)

# 최종 필터링 데이터
target_df = df_filtered.copy()
if sel_camp != '전체': target_df = target_df[target_df['Campaign'] == sel_camp]
if sel_adgroup != '전체': target_df = target_df[target_df['AdGroup'] == sel_adgroup]
if sel_creative: target_df = target_df[target_df['Creative_ID'].isin(sel_creative)]


# =============================================================================
# [SECTION 1] 진단 리포트 (상단 배치)
# =============================================================================
st.subheader("1. 캠페인 성과 진단 (최근 14일/7일/3일 분석)")
st.markdown("---")

# 진단은 '필터링 전 원본(Active 상태)'을 기준으로 돌리는 게 정확함
diag_base = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - timedelta(days=14))]
diag_res = run_diagnosis(diag_base, target_cpa_warning)

if not diag_res.empty:
    # 정렬: Red -> Blue -> Yellow -> White
    camp_grps = diag_res.groupby('Campaign')
    sorted_camps = []
    
    for c_name, grp in camp_grps:
        has_red = 'Red' in grp['Status_Color'].values
        has_blue = 'Blue' in grp['Status_Color'].values
        
        prio = 3
        header_color = ":grey"
        if has_red: prio = 1; header_color = ":red"
        elif has_blue: prio = 2; header_color = ":blue"
        
        sorted_camps.append({'name': c_name, 'data': grp, 'prio': prio, 'color': header_color})
    
    sorted_camps.sort(key=lambda x: x['prio'])

    for item in sorted_camps:
        # 필터 선택된 캠페인이 있으면 그것만 보여주기
        if sel_camp != '전체' and item['name'] != sel_camp: continue

        with st.expander(f"{item['color']}[{item['name']}]", expanded=(item['prio']==1)):
            for _, r in item['data'].iterrows():
                # 색상 박스 대신 st.container와 색상 텍스트 사용 (이모지 제거 요청 반영)
                with st.container(border=True):
                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                        st.markdown(f"**{r['Creative_ID']}**")
                        cols = st.columns(3)
                        cols[0].metric("3일 CPA", f"{r['CPA_3']/10000:.1f}만")
                        cols[1].caption(f"7일: {r['CPA_7']/10000:.1f}만")
                        cols[2].caption(f"14일: {r['CPA_14']/10000:.1f}만")
                    with c2:
                        # 진단 결과 (색상 텍스트)
                        t_col = "red" if r['Status_Color']=="Red" else "blue" if r['Status_Color']=="Blue" else "orange" if r['Status_Color']=="Yellow" else "green"
                        st.markdown(f":{t_col}[**{r['Diag_Title']}**]")
                        st.caption(r['Diag_Detail'])
else:
    st.info("진단할 데이터가 충분하지 않습니다.")

# =============================================================================
# [SECTION 2] 상세 데이터 테이블 (복구됨)
# =============================================================================
st.write("")
st.subheader("2. 상세 성과 데이터")
st.markdown("---")

if not target_df.empty:
    group_col = 'Campaign'
    if sel_camp != '전체': group_col = 'AdGroup'
    if sel_adgroup != '전체': group_col = 'Creative_ID'

    summary = target_df.groupby(group_col).agg({
        'Cost': 'sum', 'Conversions': 'sum', 'Clicks': 'sum', 'Impressions': 'sum', 'Conversion_Value': 'sum'
    }).reset_index()
    
    summary['CPA'] = (summary['Cost'] / summary['Conversions']).fillna(0)
    summary['ROAS'] = (summary['Conversion_Value'] / summary['Cost'] * 100).fillna(0)
    summary['CTR'] = (summary['Clicks'] / summary['Impressions'] * 100).fillna(0)
    
    st.dataframe(summary.style.format({
        'Cost': '{:,.0f}', 'Conversions': '{:,.0f}', 'CPA': '{:,.0f}', 
        'ROAS': '{:.1f}%', 'CTR': '{:.2f}%', 'Impressions': '{:,.0f}'
    }), use_container_width=True)
else:
    st.warning("선택한 기간/조건에 맞는 데이터가 없습니다.")

# =============================================================================
# [SECTION 3] 성과 추이 그래프 (복구됨)
# =============================================================================
st.write("")
st.subheader("3. 지표별 추세 그래프")
st.markdown("---")

c1, c2 = st.columns([3, 1])
metric_y = c1.selectbox("Y축 지표", ['CPA', 'ROAS', 'Cost', 'Conversions', 'CPM', 'CTR'], index=0)
chart_freq = c2.radio("집계 기준", ['일별', '주별'], horizontal=True)

if not target_df.empty:
    freq_map = {'일별': 'D', '주별': 'W'}
    
    fig = go.Figure()
    
    # 선택된 캠페인/그룹에 따라 그래프 라인 나누기
    lines_group = 'Campaign'
    if sel_camp != '전체': lines_group = 'AdGroup'
    if sel_adgroup != '전체': lines_group = 'Creative_ID'
    
    for name, grp in target_df.groupby(lines_group):
        res = grp.set_index('Date').resample(freq_map[chart_freq]).agg({
            'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversion_Value': 'sum'
        }).reset_index()
        
        # 지표 계산
        if metric_y == 'CPA': y_val = np.where(res['Conversions']>0, res['Cost']/res['Conversions'], 0)
        elif metric_y == 'ROAS': y_val = np.where(res['Cost']>0, res['Conversion_Value']/res['Cost']*100, 0)
        elif metric_y == 'CPM': y_val = np.where(res['Impressions']>0, res['Cost']/res['Impressions']*1000, 0)
        elif metric_y == 'CTR': y_val = np.where(res['Impressions']>0, res['Clicks']/res['Impressions']*100, 0)
        else: y_val = res[metric_y]
        
        fig.add_trace(go.Scatter(
            x=res['Date'], y=y_val, mode='lines+markers', name=name
        ))
    
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)