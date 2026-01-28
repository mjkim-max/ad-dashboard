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

# [세션 상태 초기화: 그래프 분석용]
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
# 2. 진단 로직 (Logic: CPA 낮을수록 Good, 높을수록 Bad)
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
    s3, s7, s14, s_all = get_stats_for_period(df, 3), get_stats_for_period(df, 7), get_stats_for_period(df, 14), get_stats_for_period(df, 9999)

    m = s3.merge(s7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3', '_7'), how='left')
    m = m.merge(s14, on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.rename(columns={'CPA': 'CPA_14', 'Cost': 'Cost_14', 'Conversions': 'Conversions_14'})
    m = m.merge(s_all[['Campaign','AdGroup','Creative_ID']], on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.fillna(0)
    for col in ['CPA_3', 'CPA_7', 'CPA_14']: m[col] = m[col].replace(0, np.inf)

    results = []
    camp_best = m[m['Conversions_14'] > 0].groupby('Campaign')['CPA_14'].min().to_dict()

    for _, row in m.iterrows():
        if row['Cost_3'] < 3000: continue
        cpa3, cpa7, cpa14 = row['CPA_3'], row['CPA_7'], row['CPA_14']
        best = camp_best.get(row['Campaign'], 99999999)
        status, title, detail = "White", "", ""

        # [Logic Check]
        # CPA > Target : 성과 나쁨 (비쌈) -> Red/Yellow
        # CPA <= Target : 성과 좋음 (저렴) -> Blue/Green

        if (cpa3 > target_cpa) and (best <= target_cpa * 0.9):
            # 내 CPA는 비싼데(>Target), 캠페인 에이스는 쌀 때(<=Target*0.9)
            status = "Red"; title = "종료 추천 (상대적 열위)"; detail = f"Best([{best:,.0f}원]) 대비 고비용."
        
        elif (cpa7 <= target_cpa * 1.2) and (cpa3 > target_cpa) and (row['CPM_3'] < row['CPM_7']*0.9) and (row['CTR_3'] < row['CTR_7']*0.9):
            # 7일은 괜찮았는데 3일만 비싸짐 + 근데 CPM/CTR 떨어짐 -> 탐색중
            status = "Yellow"; title = "보류 (타겟 탐색 신호)"; detail = "CPM/CTR 동반 하락. 탐색 중."
        
        elif (cpa14 > target_cpa) and (cpa7 > target_cpa) and (cpa3 > target_cpa):
            # 14/7/3일 전부 목표보다 비쌈 -> 진짜 못하는 애
            status = "Red"; title = "효율 저조 (지속 부진)"; detail = "2주간 목표 미달성."
        
        elif (cpa7 > target_cpa) and (cpa3 <= target_cpa):
            # 7일은 비쌌는데 3일은 목표 안쪽으로 들어옴(쌈) -> 개선
            status = "Green"; title = "성과 개선 (반등 중)"; detail = "효율 목표 달성."
        
        elif (cpa3 <= target_cpa) and (cpa7 <= target_cpa):
            # 둘 다 목표보다 쌈 -> 아주 잘함
            status = "Blue"; title = "성과 우수 (Best)"; detail = "목표 달성 중. 증액 검토."
        
        elif (cpa7 <= target_cpa) and (cpa3 > target_cpa):
            # 7일은 쌌는데 3일은 비싸짐 -> 흔들림
            status = "Yellow"; title = "최근 흔들림 (주의)"; detail = "일시적 저하인지 확인."

        row['Status_Color'] = status; row['Diag_Title'] = title; row['Diag_Detail'] = detail
        results.append(row)
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 사이드바 (사용자 요청 순서 반영)
# -----------------------------------------------------------------------------
df_raw = load_data()

# 1. 목표 설정
st.sidebar.header("목표 설정")
target_cpa_warning = st.sidebar.number_input("목표 CPA (점검)", value=100000, step=1000)
target_cpa_opportunity = st.sidebar.number_input("증액추천 CPA", value=50000, step=1000)
st.sidebar.markdown("---")

# 2. 기간 설정
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

# 3. 필터 설정
st.sidebar.header("필터 설정")
st.sidebar.write("매체선택")
c_m, c_g = st.sidebar.columns(2)
sel_pl = []
if c_m.checkbox("Meta", True): sel_pl.append("Meta")
if c_g.checkbox("Google", True): sel_pl.append("Google")
if 'Platform' in df_raw.columns: df_raw = df_raw[df_raw['Platform'].isin(sel_pl)]

# 날짜 필터링
df_filtered = df_raw.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['Date'].dt.date >= date_range[0]) & (df_filtered['Date'].dt.date <= date_range[1])]

# 캠페인 > 그룹 > 소재 필터
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
# 4. 메인 화면: 진단 리포트 (분석 버튼 추가)
# -----------------------------------------------------------------------------
st.title("광고 성과 관리 대시보드")
st.subheader("1. 캠페인 성과 진단")

diag_base = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - timedelta(days=14))]
diag_res = run_diagnosis(diag_base, target_cpa_warning)

def get_color_box(color):
    if color == "Red": return st.error("점검 필요", icon=None)
    elif color == "Yellow": return st.warning("보류 / 관망", icon=None)
    elif color == "Blue": return st.info("성과 우수", icon=None)
    elif color == "Green": return st.success("성과 개선", icon=None)
    else: return st.container(border=True)

if not diag_res.empty:
    camp_grps = diag_res.groupby('Campaign')
    sorted_camps = []
    
    for c_name, grp in camp_grps:
        has_red = 'Red' in grp['Status_Color'].values
        has_blue = 'Blue' in grp['Status_Color'].values
        prio = 3
        
        # [수정] 헤더 정보: 비용 삭제 / 3,7,14일 CPA 모두 표시
        c3 = grp['Cost_3'].sum(); cv3 = grp['Conversions_3'].sum()
        cpa3 = c3 / cv3 if cv3 > 0 else 0
        
        c7 = grp['Cost_7'].sum(); cv7 = grp['Conversions_7'].sum()
        cpa7 = c7 / cv7 if cv7 > 0 else 0
        
        c14 = grp['Cost_14'].sum(); cv14 = grp['Conversions_14'].sum()
        cpa14 = c14 / cv14 if cv14 > 0 else 0
        
        h_txt = f"{c_name} (3일:[{cpa3:,.0f}] 7일:[{cpa7:,.0f}] 14일:[{cpa14:,.0f}])"
        h_col = ":grey"
        if has_red: prio = 1; h_col = ":red"
        elif has_blue: prio = 2; h_col = ":blue"
        
        sorted_camps.append({'name': c_name, 'data': grp, 'prio': prio, 'header': h_txt, 'color': h_col})
    
    sorted_camps.sort(key=lambda x: x['prio'])

    for item in sorted_camps:
        if sel_camp != '전체' and item['name'] != sel_camp: continue
        
        with st.expander(f"{item['color']}[{item['header']}]", expanded=False):
            for _, r in item['data'].iterrows():
                with get_color_box(r['Status_Color']):
                    c1, c2, c3 = st.columns([2, 1.5, 0.5])
                    with c1:
                        st.markdown(f"**{r['Creative_ID']}**")
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.markdown(f"3일: [{r['CPA_3']:,.0f}원]")
                        cc2.markdown(f"7일: [{r['CPA_7']:,.0f}원]") # caption -> markdown으로 변경 (가독성)
                        cc3.markdown(f"14일: [{r['CPA_14']:,.0f}원]")
                    with c2:
                        t_col = "red" if r['Status_Color']=="Red" else "blue" if r['Status_Color']=="Blue" else "orange" if r['Status_Color']=="Yellow" else "green"
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
# 5. 추세 그래프 (선택된 소재 분석)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. 지표별 추세 그래프")

# 분석 대상 소재 확인
target_creative = st.session_state['chart_target_creative']
chart_data = target_df.copy()

if target_creative:
    st.info(f"🔎 현재 **'{target_creative}'** 소재를 집중 분석 중입니다.")
    chart_data = df_raw[df_raw['Creative_ID'] == target_creative] # 전체 데이터에서 해당 소재만 가져옴
    if st.button("전체 목록으로 차트 초기화"):
        st.session_state['chart_target_creative'] = None
        st.rerun()

# 그래프 컨트롤
c_opts, c_norm = st.columns([3, 1])
metrics = c_opts.multiselect(
    "지표 선택 (다중 선택 가능)", 
    ['Impressions', 'CTR', 'CPM', 'CPA', 'Cost', 'Conversions', 'ROAS'], 
    default=['Impressions', 'CTR', 'CPM']
)
use_norm = c_norm.checkbox("데이터 정규화 (0-100%)", value=True, help="단위가 다른 지표(예: CTR과 노출수)를 한 눈에 비교하기 위해 0~100 범위로 변환합니다.")

if not chart_data.empty and metrics:
    fig = go.Figure()
    
    # 데이터 집계 (일별)
    daily = chart_data.groupby('Date').agg({
        'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversion_Value': 'sum'
    }).reset_index().sort_values('Date')
    
    # 파생 지표 계산
    daily['CPA'] = np.where(daily['Conversions']>0, daily['Cost']/daily['Conversions'], 0)
    daily['CPM'] = np.where(daily['Impressions']>0, daily['Cost']/daily['Impressions']*1000, 0)
    daily['CTR'] = np.where(daily['Impressions']>0, daily['Clicks']/daily['Impressions']*100, 0)
    daily['ROAS'] = np.where(daily['Cost']>0, daily['Conversion_Value']/daily['Cost']*100, 0)

    for m in metrics:
        y_data = daily[m]
        y_name = m
        
        # 정규화 로직
        if use_norm and y_data.max() > 0:
            y_plot = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100
            hover_temp = f"{m}: %{{customdata:,.2f}}"
        else:
            y_plot = y_data
            hover_temp = f"{m}: %{{y:,.2f}}"

        fig.add_trace(go.Scatter(
            x=daily['Date'], y=y_plot, mode='lines+markers', name=y_name,
            customdata=y_data, hovertemplate=hover_temp
        ))

    fig.update_layout(height=450, hovermode='x unified', title=f"'{target_creative or '선택된 필터'}' 추세 분석")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("데이터가 없거나 지표가 선택되지 않았습니다.")