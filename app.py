import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 관리 BI", page_icon=None, layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    div[data-testid="stExpanderDetails"] {padding-top: 0.5rem; padding-bottom: 0.5rem;}
    p {margin-bottom: 0px !important;} 
    hr {margin: 0.5rem 0 !important;}
</style>
""", unsafe_allow_html=True)

# [주소 설정]
META_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1jEB4zTYPb2mrxZGXriju6RymHo1nEMC8QIVzqgiHwdg/edit?gid=141038195#gid=141038195"
GOOGLE_DEMO_SHEET_URL = "https://docs.google.com/spreadsheets/d/17z8PyqTdVFyF4QuTUKe6b0T_acWw2QbfvUP8DnTo5LM/edit?gid=29934845#gid=29934845"

# [세션 상태 초기화]
if 'chart_target_creative' not in st.session_state:
    st.session_state['chart_target_creative'] = None
if 'chart_target_adgroup' not in st.session_state: # [NEW] 광고그룹 추적용
    st.session_state['chart_target_adgroup'] = None

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
def load_main_data():
    dfs = []
    # 메인 시트 매핑
    rename_map = {
        '일': 'Date', '날짜': 'Date', 'Date': 'Date',
        '캠페인 이름': 'Campaign', '캠페인': 'Campaign', 'Campaign': 'Campaign',
        '광고 세트 이름': 'AdGroup', '광고 그룹 이름': 'AdGroup', 'AdGroup': 'AdGroup',
        '광고 이름': 'Creative_ID', '소재 이름': 'Creative_ID', 'Creative_ID': 'Creative_ID',
        '지출 금액 (KRW)': 'Cost', '비용': 'Cost', 'Cost': 'Cost',
        '노출': 'Impressions', 'Impressions': 'Impressions',
        '링크 클릭': 'Clicks', 'Clicks': 'Clicks',
        '구매': 'Conversions', 'Conversions': 'Conversions',
        '구매 전환값': 'Conversion_Value', 'Conversion_Value': 'Conversion_Value',
        '상태': 'Status', 'Status': 'Status',
        'Gender': 'Gender', 'Age': 'Age'
    }

    try:
        df_meta = pd.read_csv(convert_google_sheet_url(META_SHEET_URL))
        df_meta.columns = df_meta.columns.str.strip() # 공백 제거
        df_meta = df_meta.rename(columns=rename_map)
        df_meta['Platform'] = 'Meta'
        if 'Status' not in df_meta.columns: df_meta['Status'] = 'On'
        dfs.append(df_meta)
    except: pass

    try:
        df_google = pd.read_csv(convert_google_sheet_url(GOOGLE_SHEET_URL))
        df_google.columns = df_google.columns.str.strip() # 공백 제거
        df_google = df_google.rename(columns=rename_map)
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
    
    if 'Gender' not in df.columns: df['Gender'] = 'Unknown'
    if 'Age' not in df.columns: df['Age'] = 'Unknown'
    df['Gender'] = df['Gender'].fillna('Unknown')
    df['Age'] = df['Age'].fillna('Unknown')
    df['Gender'] = df['Gender'].replace({'male': '남성', 'female': '여성', 'Male': '남성', 'Female': '여성'})
            
    return df

@st.cache_data(ttl=600)
def load_google_demo_data():
    try:
        df = pd.read_csv(convert_google_sheet_url(GOOGLE_DEMO_SHEET_URL))
        df.columns = df.columns.str.strip() # 공백 제거
        
        rename_map = {
            'Date': 'Date', 'Campaign': 'Campaign', 'AdGroup': 'AdGroup',
            'Gender': 'Gender', 'Age': 'Age', 'Cost': 'Cost',
            'Impressions': 'Impressions', 'Clicks': 'Clicks',
            'Conversions': 'Conversions', 'Conversion_Value': 'Conversion_Value',
            'Status': 'Status'
        }
        df = df.rename(columns=rename_map)
        
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        for col in ['Cost', 'Conversions', 'Impressions', 'Clicks', 'Conversion_Value']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '').replace('nan', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if 'Gender' not in df.columns: df['Gender'] = 'Unknown'
        if 'Age' not in df.columns: df['Age'] = 'Unknown'
        df['Gender'] = df['Gender'].replace({'male': '남성', 'female': '여성', 'Male': '남성', 'Female': '여성'})
            
        return df
    except:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. 진단 로직
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

        if (cpa14 <= target_cpa) and (cpa7 <= target_cpa) and (cpa3 <= target_cpa):
            status = "Blue"; title = "성과 우수 (Best)"; detail = "14일/7일/3일 모두 목표 달성."
        elif (cpa14 > target_cpa) and (cpa7 > target_cpa) and (cpa3 > target_cpa):
            status = "Red"; title = "종료 추천 (지속 부진)"; detail = "14일/7일/3일 모두 목표 미달성."
        else:
            status = "Yellow"
            if cpa3 <= target_cpa: title = "성장 가능성 (반등)"; detail = "과거엔 목표 초과했으나, 최근 3일은 목표 달성."
            else: title = "관망 필요 (최근 저하)"; detail = "과거엔 좋았으나, 최근 3일은 목표 초과."

        row['Status_Color'] = status; row['Diag_Title'] = title; row['Diag_Detail'] = detail
        row['AdGroup'] = row['AdGroup'] # 광고그룹 정보 보존
        results.append(row)
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 사이드바 & 데이터 준비
# -----------------------------------------------------------------------------
df_raw = load_main_data()
df_google_demo_raw = load_google_demo_data()

st.sidebar.header("목표 설정")
target_cpa_warning = st.sidebar.number_input("목표 CPA", value=100000, step=1000)
st.sidebar.markdown("---")

st.sidebar.header("기간 설정")
preset = st.sidebar.selectbox("기간선택", ["오늘", "어제", "최근 3일", "최근 7일", "최근 14일", "최근 30일", "이번 달", "지난 달", "최근 90일", "전체 기간"], index=4)
today = datetime.now().date()

# [중요] 사용자가 데이터를 2025년과 2026년을 섞어서 넣었으므로, 기본 날짜 계산을 유연하게
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
elif preset == "전체 기간": s = date(2020, 1, 1); e = today # 충분히 넓게

date_range = st.sidebar.date_input("날짜범위", [s, e])
st.sidebar.markdown("---")

st.sidebar.header("필터 설정")
c_m, c_g = st.sidebar.columns(2)
sel_pl = []
if c_m.checkbox("Meta", True): sel_pl.append("Meta")
if c_g.checkbox("Google", True): sel_pl.append("Google")
if 'Platform' in df_raw.columns: df_raw = df_raw[df_raw['Platform'].isin(sel_pl)]

# 1. Main Data 필터링
df_filtered = df_raw.copy()
if len(date_range) == 2:
    df_filtered = df_filtered[(df_filtered['Date'].dt.date >= date_range[0]) & (df_filtered['Date'].dt.date <= date_range[1])]

# 2. Google Demo Data 필터링
df_google_demo_filtered = df_google_demo_raw.copy()
if not df_google_demo_filtered.empty and 'Date' in df_google_demo_filtered.columns and len(date_range) == 2:
    df_google_demo_filtered = df_google_demo_filtered[
        (df_google_demo_filtered['Date'].dt.date >= date_range[0]) & 
        (df_google_demo_filtered['Date'].dt.date <= date_range[1])
    ]

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
        prio = 1 if has_red else 2 if has_yellow else 3
        h_col = ":red" if has_red else ":orange" if has_yellow else ":blue"
        
        c3 = grp['Cost_3'].sum(); cv3 = grp['Conversions_3'].sum()
        cpa3 = c3 / cv3 if cv3 > 0 else 0
        c7 = grp['Cost_7'].sum(); cv7 = grp['Conversions_7'].sum()
        cpa7 = c7 / cv7 if cv7 > 0 else 0
        c14 = grp['Cost_14'].sum(); cv14 = grp['Conversions_14'].sum()
        cpa14 = c14 / cv14 if cv14 > 0 else 0

        sorted_camps.append({
            'name': c_name, 'data': grp, 'prio': prio, 'header': c_name, 'color': h_col,
            'stats_3': (cpa3, c3, cv3), 'stats_7': (cpa7, c7, cv7), 'stats_14': (cpa14, c14, cv14)
        })
    
    sorted_camps.sort(key=lambda x: x['prio'])

    for item in sorted_camps:
        if sel_camp != '전체' and item['name'] != sel_camp: continue
        
        with st.expander(f"{item['color']}[{item['header']}]", expanded=False):
            st.markdown("##### 캠페인 기간별 성과 요약")
            c_3d, c_7d, c_14d = st.columns(3)
            
            def fmt_head(label, cpa, cost, conv):
                return f"""<div style="line-height:1.4;"><strong>{label}</strong><br>CPA <strong>{cpa:,.0f}원</strong><br>비용 {cost:,.0f}원<br>전환 {conv:,.0f}</div>"""
            
            with c_3d: st.markdown(fmt_head("최근 3일", *item['stats_3']), unsafe_allow_html=True)
            with c_7d: st.markdown(fmt_head("최근 7일", *item['stats_7']), unsafe_allow_html=True)
            with c_14d: st.markdown(fmt_head("최근 14일", *item['stats_14']), unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #f0f2f6;'>", unsafe_allow_html=True)
            st.markdown("##### 소재별 진단")
            
            for idx, (_, r) in enumerate(item['data'].iterrows()):
                st.markdown(f"#### {r['Creative_ID']}")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
                
                def format_stat_block(label, cpa, cost, conv):
                    cpa_val = "∞" if cpa == np.inf else f"{cpa:,.0f}"
                    return f"""<div style="line-height:1.6;"><strong>{label}</strong><br>CPA <strong>{cpa_val}원</strong><br>비용 {cost:,.0f}원<br>전환 {conv:,.0f}</div>"""

                with col1: st.markdown(format_stat_block("3일", r['CPA_3'], r['Cost_3'], r['Conversions_3']), unsafe_allow_html=True)
                with col2: st.markdown(format_stat_block("7일", r['CPA_7'], r['Cost_7'], r['Conversions_7']), unsafe_allow_html=True)
                with col3: st.markdown(format_stat_block("14일", r['CPA_14'], r['Cost_14'], r['Conversions_14']), unsafe_allow_html=True)
                with col4:
                    t_col = "red" if r['Status_Color']=="Red" else "blue" if r['Status_Color']=="Blue" else "orange"
                    st.markdown(f":{t_col}[**{r['Diag_Title']}**]")
                    st.caption(r['Diag_Detail'])
                    
                    unique_key = f"btn_{item['name']}_{r['Creative_ID']}_{idx}"
                    # [핵심 수정] 버튼 클릭 시 AdGroup도 함께 저장
                    if st.button("분석하기", key=unique_key):
                        st.session_state['chart_target_creative'] = r['Creative_ID']
                        st.session_state['chart_target_adgroup'] = r['AdGroup'] # AdGroup 저장
                        st.rerun()
                st.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #f0f2f6;'>", unsafe_allow_html=True)
else:
    st.info("진단 데이터 부족")

# -----------------------------------------------------------------------------
# 5. 추세 그래프 & 상세 표 & 성별/연령 분석
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. 지표별 추세 및 상세 분석")

target_creative = st.session_state['chart_target_creative']
target_adgroup = st.session_state['chart_target_adgroup']

# 데이터 준비
trend_df = target_df.copy()
demog_df = pd.DataFrame() 
is_specific = False

if target_creative:
    # 1. Trend Data (선택한 소재만)
    trend_df = target_df[target_df['Creative_ID'] == target_creative]
    
    # 2. Demo Data (AdGroup 기반)
    # 현재 선택된 소재의 행을 찾아서 플랫폼 확인
    sel_row = target_df[target_df['Creative_ID'] == target_creative]
    
    if not sel_row.empty:
        platform = sel_row['Platform'].iloc[0]
        # 만약 session에 저장된 adgroup이 없으면 현재 row에서 가져옴
        current_adgroup = target_adgroup if target_adgroup else sel_row['AdGroup'].iloc[0]
        
        if platform == 'Google':
            # 구글: 기간 필터링된 데모 시트에서 AdGroup으로 조회
            if not df_google_demo_filtered.empty:
                demog_df = df_google_demo_filtered[df_google_demo_filtered['AdGroup'] == current_adgroup]
                
                # [친절한 디버깅 메시지] 데이터가 비었을 경우 원인 추정
                if demog_df.empty:
                    st.warning(f"⚠️ '{current_adgroup}' 광고그룹 데이터가 하단 시트에 없습니다. 날짜범위({date_range[0]}~{date_range[1]})가 맞는지 확인해주세요. (시트 날짜: 2025년 / 현재 선택: 2026년 가능성)")
                else:
                    st.info(f"🔎 **'{target_creative}'** (구글) 분석 중. 인구통계는 **'{current_adgroup}'** 광고그룹 전체 기준입니다.")
            else:
                st.warning("구글 인구통계 데이터가 날짜 필터링에 의해 모두 제외되었습니다. 기간 설정을 확인해주세요.")
        else:
            # 메타: 그냥 해당 소재 데이터
            demog_df = trend_df
            st.info(f"🔎 현재 **'{target_creative}'** 소재를 집중 분석 중입니다.")
            
    is_specific = True
    
    if st.button("전체 목록으로 차트 초기화"):
        st.session_state['chart_target_creative'] = None
        st.session_state['chart_target_adgroup'] = None
        st.rerun()
else:
    # 전체 모드
    demog_df = target_df.copy()
    st.info(f"📊 통합 추세 분석 중 (특정 소재를 보려면 위에서 '분석하기'를 누르세요)")

c_freq, c_opts, c_norm = st.columns([1, 2, 1])
freq_option = c_freq.radio("집계 기준", ["1일", "3일", "7일"], horizontal=True)
freq_map = {"1일": "D", "3일": "3D", "7일": "W"}
metrics = c_opts.multiselect("지표 선택", ['Impressions', 'Clicks', 'CTR', 'CPM', 'CPC', 'CPA', 'Cost', 'Conversions', 'CVR', 'ROAS'], default=['Conversions', 'CPA', 'CTR', 'Impressions'])
use_norm = c_norm.checkbox("데이터 정규화 (0-100%)", value=True)

if not trend_df.empty and metrics:
    agg_df = trend_df.set_index('Date').groupby(pd.Grouper(freq=freq_map[freq_option])).agg({
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
    style_map = {'Conversions': {'color': 'black', 'width': 3}, 'CPA': {'color': 'red', 'width': 3}, 'CTR': {'color': 'blue', 'width': 2}, 'Impressions': {'color': 'green', 'width': 2}}
    
    for m in metrics:
        y_data = plot_df[m]
        y_plot = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100 if use_norm and y_data.max() > 0 else y_data
        style = style_map.get(m, {'color': None, 'width': 2})
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=y_plot, mode='lines+markers', name=m, line=dict(color=style['color'], width=style['width']), customdata=y_data, hovertemplate=f"{m}: %{{customdata:,.2f}}"))

    fig.update_layout(height=450, hovermode='x unified', title=f"추세 분석 ({freq_option} 기준)", plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    table_df = agg_df.copy(); table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(table_df[['Date', 'CPA', 'Cost', 'Impressions', 'CPM', 'Clicks', 'Conversions', 'CTR', 'CPC', 'CVR', 'ROAS']], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("성별/연령 심층 분석")
    
    if demog_df.empty or 'Gender' not in demog_df.columns:
        st.info("데이터가 없습니다. (날짜 범위나 시트 데이터를 확인해주세요)")
    else:
        valid_gender_check = demog_df[~demog_df['Gender'].isin(['Unknown', 'unknown', '알수없음'])]
        if valid_gender_check.empty:
            st.info("성별/연령 정보가 없습니다.")
        else:
            demog_agg = valid_gender_check.groupby(['Age', 'Gender']).agg({'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum'}).reset_index()
            demog_agg['CPA'] = np.where(demog_agg['Conversions']>0, demog_agg['Cost']/demog_agg['Conversions'], 0)
            
            male_data = demog_agg[demog_agg['Gender'].str.contains('남성|Male|male', case=False, na=False)]
            female_data = demog_agg[demog_agg['Gender'].str.contains('여성|Female|female', case=False, na=False)]
            
            title_txt = f"{target_creative} 성별/연령별 전환수" if is_specific else "성별/연령별 전환수 (통합)"
            st.markdown(f"#### {title_txt}")
            
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Bar(x=male_data['Age'], y=male_data['Conversions'], name='남성', marker_color='#9EB9F3'))
            fig_conv.add_trace(go.Bar(x=female_data['Age'], y=female_data['Conversions'], name='여성', marker_color='#F8C8C8'))
            fig_conv.update_layout(barmode='group', height=350, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_conv, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1: st.markdown("**CPA**"); st.dataframe(demog_agg.pivot_table(index='Gender', columns='Age', values='CPA', aggfunc='sum', fill_value=0).style.format("{:,.0f}"), use_container_width=True)
            with c2: st.markdown("**비용**"); st.dataframe(demog_agg.pivot_table(index='Gender', columns='Age', values='Cost', aggfunc='sum', fill_value=0).style.format("{:,.0f}"), use_container_width=True)
else:
    st.warning("설정된 기간 내에 데이터가 없습니다.")
