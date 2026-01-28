import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 관리 BI", page_icon="📊", layout="wide")

# [주소 설정]
META_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1jEB4zTYPb2mrxZGXriju6RymHo1nEMC8QIVzqgiHwdg/edit?gid=141038195#gid=141038195"

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
    
    s3 = get_stats_for_period(df, 3)
    s7 = get_stats_for_period(df, 7)
    s14 = get_stats_for_period(df, 14)
    s_all = get_stats_for_period(df, 9999) 

    m = s3.merge(s7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3', '_7'), how='left')
    m = m.merge(s14, on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.rename(columns={'CPA': 'CPA_14', 'Cost': 'Cost_14', 'Conversions': 'Conversions_14'})
    m = m.merge(s_all[['Campaign','AdGroup','Creative_ID']], on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.fillna(0)

    for col in ['CPA_3', 'CPA_7', 'CPA_14']:
        m[col] = m[col].replace(0, np.inf)

    results = []
    camp_best = m[m['Conversions_14'] > 0].groupby('Campaign')['CPA_14'].min().to_dict()

    for _, row in m.iterrows():
        if row['Cost_3'] < 3000: continue

        cpa3, cpa7, cpa14 = row['CPA_3'], row['CPA_7'], row['CPA_14']
        best = camp_best.get(row['Campaign'], 99999999)
        status, title, detail = "White", "", ""

        if (cpa3 > target_cpa) and (best <= target_cpa * 0.9):
            status = "Red"; title = "[종료 추천] 상대적 열위"; detail = f"Best({best:,.0f}원) 대비 저조."
        elif (cpa7 <= target_cpa * 1.2) and (cpa3 > target_cpa) and (row['CPM_3'] < row['CPM_7']*0.9) and (row['CTR_3'] < row['CTR_7']*0.9):
            status = "Yellow"; title = "[보류] 타겟 탐색 신호"; detail = "CPM/CTR 동반 하락. 탐색 중."
        elif (cpa14 > target_cpa) and (cpa7 > target_cpa) and (cpa3 > target_cpa):
            status = "Red"; title = "[효율 저조] 지속 부진"; detail = "2주간 목표 미달성."
        elif (cpa7 > target_cpa) and (cpa3 <= target_cpa):
            status = "Green"; title = "[성과 개선] 반등 중"; detail = "효율 개선됨 (골든크로스)."
        elif (cpa3 <= target_cpa) and (cpa7 <= target_cpa):
            status = "Blue"; title = "[성과 우수] Best"; detail = "목표 달성 중. 증액 검토."
        elif (cpa7 <= target_cpa) and (cpa3 > target_cpa):
            status = "Yellow"; title = "[주의] 최근 흔들림"; detail = "일시적 저하인지 확인."

        row['Status_Color'] = status
        row['Diag_Title'] = title
        row['Diag_Detail'] = detail
        results.append(row)

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 메인 앱 실행
# -----------------------------------------------------------------------------
df_raw = load_data()

# [SIDEBAR] 1. 날짜 설정 (프리셋 복구)
st.sidebar.header("📅 날짜 및 매체 설정")

preset = st.sidebar.selectbox(
    "기간 선택", 
    ["오늘", "어제", "최근 3일", "최근 7일", "최근 14일", "최근 30일", "이번 달", "지난 달", "최근 90일", "직접 선택"]
)

today = datetime.now().date()
if preset == "오늘": s_date = today; e_date = today
elif preset == "어제": s_date = today - timedelta(days=1); e_date = s_date
elif preset == "최근 3일": s_date = today - timedelta(days=2); e_date = today
elif preset == "최근 7일": s_date = today - timedelta(days=6); e_date = today
elif preset == "최근 14일": s_date = today - timedelta(days=13); e_date = today
elif preset == "최근 30일": s_date = today - timedelta(days=29); e_date = today
elif preset == "최근 90일": s_date = today - timedelta(days=89); e_date = today
elif preset == "이번 달": s_date = date(today.year, today.month, 1); e_date = today
elif preset == "지난 달": 
    first_day_this_month = date(today.year, today.month, 1)
    e_date = first_day_this_month - timedelta(days=1)
    s_date = date(e_date.year, e_date.month, 1)
else:
    s_date = df_raw['Date'].min().date() if not df_raw.empty else today
    e_date = df_raw['Date'].max().date() if not df_raw.empty else today

date_range = st.sidebar.date_input("날짜 범위", [s_date, e_date])

# [SIDEBAR] 2. 매체 선택 (체크박스 좌우 배치)
st.sidebar.write("매체 선택")
c_m, c_g = st.sidebar.columns(2)
show_meta = c_m.checkbox("Meta", value=True)
show_google = c_g.checkbox("Google", value=True)

selected_platforms = []
if show_meta: selected_platforms.append("Meta")
if show_google: selected_platforms.append("Google")

if 'Platform' in df_raw.columns:
    df_raw = df_raw[df_raw['Platform'].isin(selected_platforms)]

# [SIDEBAR] 3. 필터 및 목표
st.sidebar.divider()
target_cpa_warning = st.sidebar.number_input("🔴 목표 CPA (점검)", value=100000, step=1000)
target_cpa_opportunity = st.sidebar.number_input("🔵 증액 추천 CPA", value=50000, step=1000)

status_filter = st.sidebar.radio("게재 상태", ["전체", "게재중 (On)", "비게재 (Off)"], index=1)
if 'Status' in df_raw.columns:
    if status_filter == "게재중 (On)": df_raw = df_raw[df_raw['Status'] == 'On']
    elif status_filter == "비게재 (Off)": df_raw = df_raw[df_raw['Status'] == 'Off']

# 캠페인/그룹 필터 적용
df_filtered = df_raw.copy()
if len(date_range) == 2:
    start_dt, end_dt = date_range
    df_filtered = df_filtered[(df_filtered['Date'].dt.date >= start_dt) & (df_filtered['Date'].dt.date <= end_dt)]

campaigns = ['전체'] + sorted(df_filtered['Campaign'].unique().tolist())
sel_camp = st.sidebar.selectbox("캠페인 필터", campaigns)

adgroups = ['전체']
if sel_camp != '전체':
    adgroups = ['전체'] + sorted(df_filtered[df_filtered['Campaign'] == sel_camp]['AdGroup'].unique().tolist())
sel_adgroup = st.sidebar.selectbox("광고그룹 필터", adgroups)

target_df = df_filtered.copy()
if sel_camp != '전체': target_df = target_df[target_df['Campaign'] == sel_camp]
if sel_adgroup != '전체': target_df = target_df[target_df['AdGroup'] == sel_adgroup]

# =============================================================================
# [SECTION 1] 진단 리포트
# =============================================================================
st.title("📊 광고 성과 대시보드")
st.subheader("1. 캠페인 성과 진단 (최근 3/7/14일)")
st.markdown("---")

diag_base = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - timedelta(days=14))]
diag_res = run_diagnosis(diag_base, target_cpa_warning)

# 상태 박스 렌더링 함수
def render_status_box(status_color):
    if status_color == "Red": return st.error("🚨 점검 필요", icon="🚨")
    elif status_color == "Yellow": return st.warning("✋ 보류 / 관망", icon="✋")
    elif status_color == "Blue": return st.info("💎 성과 우수", icon="💎")
    elif status_color == "Green": return st.success("📈 성과 개선", icon="📈")
    else: return st.container(border=True)

if not diag_res.empty:
    camp_grps = diag_res.groupby('Campaign')
    sorted_camps = []
    
    for c_name, grp in camp_grps:
        has_red = 'Red' in grp['Status_Color'].values
        has_blue = 'Blue' in grp['Status_Color'].values
        
        prio = 3
        # 헤더에 표시할 캠페인 요약 정보 계산 (최근 3일 기준)
        camp_cost_3 = grp['Cost_3'].sum()
        camp_conv_3 = grp['Conversions_3'].sum()
        camp_cpa_3 = camp_cost_3 / camp_conv_3 if camp_conv_3 > 0 else 0
        
        # 헤더 텍스트 구성
        header_text = f"📂 {c_name} (💸3일 CPA: {camp_cpa_3:,.0f}원 | 💰비용: {camp_cost_3/10000:,.0f}만)"
        header_color = ":grey"

        if has_red: 
            prio = 1; header_color = ":red"
            header_text = f"🚨 {c_name} (💸3일 CPA: {camp_cpa_3:,.0f}원 | 점검 필요)"
        elif has_blue: 
            prio = 2; header_color = ":blue"
            header_text = f"✨ {c_name} (💸3일 CPA: {camp_cpa_3:,.0f}원 | 우수)"
        
        sorted_camps.append({'name': c_name, 'data': grp, 'prio': prio, 'header': header_text, 'color': header_color})
    
    sorted_camps.sort(key=lambda x: x['prio'])

    for item in sorted_camps:
        if sel_camp != '전체' and item['name'] != sel_camp: continue
        
        # 기본적으로 닫혀있게 설정 (expanded=False)
        with st.expander(f"{item['color']}[{item['header']}]", expanded=False):
            for _, r in item['data'].iterrows():
                status_box = render_status_box(r['Status_Color'])
                with status_box:
                    c1, c2 = st.columns([1.5, 1])
                    with c1:
                        st.markdown(f"**{r['Creative_ID']}**")
                        # [수정] 3일 CPA가 너무 크지 않게 Markdown으로 통일
                        cols = st.columns(3)
                        
                        val_3 = "∞" if r['CPA_3'] == np.inf else f"{r['CPA_3']/10000:.1f}만"
                        val_7 = "∞" if r['CPA_7'] == np.inf else f"{r['CPA_7']/10000:.1f}만"
                        val_14 = "∞" if r['CPA_14'] == np.inf else f"{r['CPA_14']/10000:.1f}만"
                        
                        cols[0].markdown(f"**3일:** {val_3}")
                        cols[1].markdown(f"**7일:** {val_7}")
                        cols[2].markdown(f"**14일:** {val_14}")
                        
                    with c2:
                        t_col = "red" if r['Status_Color']=="Red" else "blue" if r['Status_Color']=="Blue" else "orange" if r['Status_Color']=="Yellow" else "green"
                        st.markdown(f":{t_col}[**{r['Diag_Title']}**]")
                        st.caption(r['Diag_Detail'])
                        if r['CPM_3'] > 0:
                            arr_cpm = "⬇️" if r['CPM_3'] < r['CPM_7'] else "⬆️"
                            arr_ctr = "⬇️" if r['CTR_3'] < r['CTR_7'] else "⬆️"
                            st.caption(f"신호: CPM{arr_cpm} CTR{arr_ctr}")
else:
    st.info("진단할 데이터가 충분하지 않습니다.")

# =============================================================================
# [SECTION 2] 상세 데이터 (하단)
# =============================================================================
st.write(""); st.subheader("2. 상세 성과 데이터")
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
    st.warning("데이터가 없습니다.")

# =============================================================================
# [SECTION 3] 그래프
# =============================================================================
st.write(""); st.subheader("3. 추세 그래프")
st.markdown("---")
c1, c2 = st.columns([3, 1])
metric_y = c1.selectbox("Y축 지표", ['CPA', 'ROAS', 'Cost', 'Conversions', 'CPM', 'CTR'])
chart_freq = c2.radio("집계 기준", ['일별', '주별'], horizontal=True)

if not target_df.empty:
    freq_map = {'일별': 'D', '주별': 'W'}
    fig = go.Figure()
    lines_group = 'Campaign'
    if sel_camp != '전체': lines_group = 'AdGroup'
    if sel_adgroup != '전체': lines_group = 'Creative_ID'
    
    for name, grp in target_df.groupby(lines_group):
        res = grp.set_index('Date').resample(freq_map[chart_freq]).agg({
            'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversion_Value': 'sum'
        }).reset_index()
        
        if metric_y == 'CPA': y_val = np.where(res['Conversions']>0, res['Cost']/res['Conversions'], 0)
        elif metric_y == 'ROAS': y_val = np.where(res['Cost']>0, res['Conversion_Value']/res['Cost']*100, 0)
        elif metric_y == 'CPM': y_val = np.where(res['Impressions']>0, res['Cost']/res['Impressions']*1000, 0)
        elif metric_y == 'CTR': y_val = np.where(res['Impressions']>0, res['Clicks']/res['Impressions']*100, 0)
        else: y_val = res[metric_y]
        
        fig.add_trace(go.Scatter(x=res['Date'], y=y_val, mode='lines+markers', name=name))
    st.plotly_chart(fig, use_container_width=True)