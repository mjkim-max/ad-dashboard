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

# -----------------------------------------------------------------------------
# [주소 설정]
# 1. 메인 데이터 (소재별/일별 데이터 - Meta는 여기에 성별/연령 포함됨)
MAIN_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"

# 2. 세트 데이터 (구글 전용: 광고그룹별/성별/연령 데이터)
SET_SHEET_URL = "https://docs.google.com/spreadsheets/d/17z8PyqTdVFyF4QuTUKe6b0T_acWw2QbfvUP8DnTo5LM/edit?gid=29934845#gid=29934845"

# -----------------------------------------------------------------------------
# [세션 초기화]
# -----------------------------------------------------------------------------
if 'chart_target_creative' not in st.session_state:
    st.session_state['chart_target_creative'] = None

# -----------------------------------------------------------------------------
# 1. 데이터 로드 함수
# -----------------------------------------------------------------------------
def convert_url(url):
    if "/edit" in url:
        base = url.split("/edit")[0]
        if "gid=" in url:
            gid = url.split("gid=")[1].split("#")[0]
            return f"{base}/export?format=csv&gid={gid}"
    return url

@st.cache_data(ttl=600)
def load_main_sheet():
    try:
        df = pd.read_csv(convert_url(MAIN_SHEET_URL))
        df.columns = df.columns.str.strip()
        
        rename_map = {
            '일': 'Date', '날짜': 'Date', 
            '캠페인 이름': 'Campaign', '캠페인': 'Campaign',
            '광고 세트 이름': 'AdGroup', '광고 그룹 이름': 'AdGroup',
            '광고 이름': 'Creative_ID', '소재 이름': 'Creative_ID',
            '지출 금액 (KRW)': 'Cost', '비용': 'Cost',
            '노출': 'Impressions',
            '링크 클릭': 'Clicks', '클릭': 'Clicks',
            '구매': 'Conversions', '전환': 'Conversions',
            '구매 전환값': 'Conversion_Value', '전환값': 'Conversion_Value',
            '상태': 'Status', 'Platform': 'Platform',
            'Gender': 'Gender', '성별': 'Gender', 'Age': 'Age', '연령': 'Age'
        }
        df = df.rename(columns=rename_map)
        
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for c in ['Cost', 'Conversions', 'Impressions', 'Clicks', 'Conversion_Value']:
            if c in df.columns:
                if df[c].dtype == 'object': df[c] = df[c].astype(str).str.replace(',', '').replace('nan','0')
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        if 'AdGroup' in df.columns: df['AdGroup'] = df['AdGroup'].astype(str).str.strip()
        if 'Creative_ID' in df.columns: df['Creative_ID'] = df['Creative_ID'].astype(str).str.strip()
        if 'Platform' not in df.columns: df['Platform'] = 'Unknown'
        
        # Meta 데이터의 경우 Gender/Age가 있을 수 있으므로 정규화
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].replace({'male': '남성', 'female': '여성', 'Male': '남성', 'Female': '여성'})
        
        return df
    except Exception as e:
        st.error(f"메인 시트 로드 중 에러: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_set_sheet():
    try:
        df = pd.read_csv(convert_url(SET_SHEET_URL))
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'Date': 'Date', 'Campaign': 'Campaign', 'AdGroup': 'AdGroup',
            'Gender': 'Gender', 'Age': 'Age', 'Cost': 'Cost',
            'Impressions': 'Impressions', 'Clicks': 'Clicks',
            'Conversions': 'Conversions', 'Conversion_Value': 'Conversion_Value'
        }
        df = df.rename(columns=rename_map)
        
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for c in ['Cost', 'Conversions', 'Impressions', 'Clicks', 'Conversion_Value']:
            if c in df.columns:
                if df[c].dtype == 'object': df[c] = df[c].astype(str).str.replace(',', '').replace('nan','0')
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                
        if 'AdGroup' in df.columns: df['AdGroup'] = df['AdGroup'].astype(str).str.strip()
        
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].replace({'male': '남성', 'female': '여성', 'Male': '남성', 'Female': '여성'})
            
        return df
    except Exception as e:
        st.error(f"세트 시트 로드 중 에러: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. 로직: 진단
# -----------------------------------------------------------------------------
def get_stats(df, days):
    start = df['Date'].max() - timedelta(days=days-1)
    sub = df[df['Date'] >= start]
    grp = sub.groupby(['Campaign', 'AdGroup', 'Creative_ID']).agg({'Cost':'sum','Conversions':'sum'}).reset_index()
    grp['CPA'] = np.where(grp['Conversions']>0, grp['Cost']/grp['Conversions'], np.inf)
    return grp

def run_diagnosis(df, target_cpa):
    if df.empty: return pd.DataFrame()
    s3 = get_stats(df, 3); s7 = get_stats(df, 7); s14 = get_stats(df, 14)
    
    m = s3.merge(s7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3','_7'), how='left')
    m = m.merge(s14, on=['Campaign','AdGroup','Creative_ID'], how='left')
    m = m.rename(columns={'CPA':'CPA_14', 'Cost':'Cost_14', 'Conversions':'Conversions_14'})
    m = m.fillna(0)
    
    results = []
    for _, r in m.iterrows():
        if r['Cost_3'] < 3000: continue
        c3, c7, c14 = r['CPA_3'], r['CPA_7'], r['CPA_14']
        
        status, title, detail = "White", "대기", ""
        if c14<=target_cpa and c7<=target_cpa and c3<=target_cpa:
            status="Blue"; title="성과 우수"; detail="3/7/14일 모두 목표 달성"
        elif c14>target_cpa and c7>target_cpa and c3>target_cpa:
            status="Red"; title="종료 추천"; detail="3/7/14일 모두 목표 초과"
        else:
            status="Yellow"; title="판별 필요"; detail="추세 변동 있음"
            
        r['Status_Color'] = status; r['Diag_Title'] = title; r['Diag_Detail'] = detail
        results.append(r)
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. UI: 사이드바
# -----------------------------------------------------------------------------
df_main = load_main_sheet()
df_set = load_set_sheet()

st.sidebar.header("설정")
target_cpa = st.sidebar.number_input("목표 CPA", 10000, step=1000)
st.sidebar.markdown("---")

preset = st.sidebar.selectbox("기간", ["오늘", "어제", "최근 3일", "최근 7일", "최근 14일", "최근 30일", "이번 달", "지난 달", "전체 기간"], index=4)
today = datetime.now().date()
if preset=="오늘": s=e=today
elif preset=="어제": s=today-timedelta(1); e=s
elif preset=="최근 3일": s=today-timedelta(2); e=today
elif preset=="최근 7일": s=today-timedelta(6); e=today
elif preset=="최근 14일": s=today-timedelta(13); e=today
elif preset=="최근 30일": s=today-timedelta(29); e=today
elif preset=="이번 달": s=date(today.year,today.month,1); e=today
elif preset=="지난 달": first=date(today.year,today.month,1); e=first-timedelta(1); s=date(e.year,e.month,1)
elif preset=="전체 기간": s=date(2020,1,1); e=today
date_range = st.sidebar.date_input("날짜", [s, e])

# 필터링
df_main_fil = df_main.copy()
df_set_fil = df_set.copy()

if len(date_range) == 2:
    df_main_fil = df_main_fil[(df_main_fil['Date'].dt.date >= date_range[0]) & (df_main_fil['Date'].dt.date <= date_range[1])]
    df_set_fil = df_set_fil[(df_set_fil['Date'].dt.date >= date_range[0]) & (df_set_fil['Date'].dt.date <= date_range[1])]

camps = ['전체'] + sorted(df_main_fil['Campaign'].unique().tolist())
sel_camp = st.sidebar.selectbox("캠페인", camps)
if sel_camp != '전체':
    df_main_fil = df_main_fil[df_main_fil['Campaign'] == sel_camp]
    if 'Campaign' in df_set_fil.columns:
        df_set_fil = df_set_fil[df_set_fil['Campaign'] == sel_camp]

# -----------------------------------------------------------------------------
# 4. UI: 메인 화면 (진단)
# -----------------------------------------------------------------------------
st.title("광고 성과 관리")

diag_res = run_diagnosis(df_main_fil, target_cpa)

def color_box(color):
    if color=="Red": return st.error("종료 추천", icon=None)
    if color=="Yellow": return st.warning("판별 필요", icon=None)
    if color=="Blue": return st.info("성과 우수", icon=None)
    return st.container()

if not diag_res.empty:
    for c_name, grp in diag_res.groupby('Campaign'):
        with st.expander(f"📌 {c_name}", expanded=False):
            for i, r in grp.iterrows():
                col1, col2, col3, col4 = st.columns([1,1,1,1])
                with col1: st.markdown(f"**{r['Creative_ID']}**")
                with col2: st.caption(f"3일 CPA: {r['CPA_3']:,.0f}")
                with col3: st.caption(f"14일 CPA: {r['CPA_14']:,.0f}")
                with col4:
                    if st.button("분석하기", key=f"btn_{i}"):
                        st.session_state['chart_target_creative'] = r['Creative_ID']
                        st.rerun()
                st.divider()

# -----------------------------------------------------------------------------
# 5. UI: 상세 분석 (핵심 로직 수정)
# -----------------------------------------------------------------------------
st.markdown("### 📊 상세 분석")

target_creative = st.session_state['chart_target_creative']

if target_creative:
    st.success(f"선택한 소재: **{target_creative}**")
    if st.button("초기화"):
        st.session_state['chart_target_creative'] = None
        st.rerun()

    # [Step 1] 메인 시트에서 해당 소재 데이터 찾기
    chart_data = df_main_fil[df_main_fil['Creative_ID'] == target_creative]
    
    if not chart_data.empty:
        # A. 꺾은선 그래프 (항상 메인 시트 소재 데이터 기준)
        agg = chart_data.groupby('Date').agg({'Conversions':'sum', 'CPA':'mean', 'Impressions':'sum', 'Cost':'sum'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Conversions'], name='전환수', line=dict(color='black', width=3)))
        fig.add_trace(go.Scatter(x=agg['Date'], y=agg['CPA'], name='CPA', line=dict(color='red', width=2), yaxis='y2'))
        
        fig.update_layout(
            title=f"'{target_creative}' 일별 추세",
            yaxis=dict(title="전환수"),
            yaxis2=dict(title="CPA", overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------
        # [Step 2] 플랫폼에 따른 분기 처리 (Meta vs Google)
        # -------------------------------------------------------
        target_platform = chart_data['Platform'].iloc[0]
        target_adgroup = chart_data['AdGroup'].iloc[0]
        
        demo_data = pd.DataFrame()
        source_msg = ""

        if target_platform == 'Google':
            # 구글: 세트 전용 시트에서 [광고 그룹]으로 검색
            source_msg = f"구글 소재입니다. '{target_adgroup}' 광고 그룹 데이터(세트 시트)를 불러옵니다."
            if not df_set_fil.empty:
                demo_data = df_set_fil[df_set_fil['AdGroup'] == target_adgroup]
        
        elif target_platform == 'Meta':
            # 메타: 메인 시트에서 [소재]로 검색 (메타는 메인 시트에 성별/연령 있다고 가정)
            source_msg = f"메타 소재입니다. '{target_creative}' 소재 데이터(메인 시트)를 그대로 분석합니다."
            # 메인 시트에는 날짜별로 쪼개져 있으니, 해당 소재 데이터 그대로 사용
            demo_data = chart_data
        
        else:
            # 플랫폼 알 수 없는 경우 (안전장치: 일단 메인 시트 사용)
            source_msg = "플랫폼 정보 없음. 메인 시트 데이터를 사용합니다."
            demo_data = chart_data

        st.info(source_msg)

        # -------------------------------------------------------
        # [Step 3] 성별/연령 데이터 시각화
        # -------------------------------------------------------
        if not demo_data.empty:
            # 유효한 성별/연령 데이터만 필터링 (Unknown 제외)
            if 'Gender' in demo_data.columns and 'Age' in demo_data.columns:
                valid_demo = demo_data[~demo_data['Gender'].isin(['Unknown', 'unknown', '알수없음'])]
                
                if not valid_demo.empty:
                    # 집계 (날짜 상관없이 합산)
                    demo_agg = valid_demo.groupby(['Age', 'Gender']).agg({'Conversions':'sum', 'Cost':'sum'}).reset_index()
                    demo_agg['CPA'] = np.where(demo_agg['Conversions']>0, demo_agg['Cost']/demo_agg['Conversions'], 0)
                    
                    male = demo_agg[demo_agg['Gender'].str.contains('남성|Male', case=False)]
                    female = demo_agg[demo_agg['Gender'].str.contains('여성|Female', case=False)]
                    
                    # 막대 그래프
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=male['Age'], y=male['Conversions'], name='남성', marker_color='#9EB9F3'))
                    fig_bar.add_trace(go.Bar(x=female['Age'], y=female['Conversions'], name='여성', marker_color='#F8C8C8'))
                    fig_bar.update_layout(title="성별/연령별 전환수", barmode='group')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # 표
                    c1, c2 = st.columns(2)
                    piv_cpa = demo_agg.pivot_table(index='Gender', columns='Age', values='CPA', aggfunc='sum', fill_value=0)
                    piv_cost = demo_agg.pivot_table(index='Gender', columns='Age', values='Cost', aggfunc='sum', fill_value=0)
                    
                    with c1: st.write("**CPA**"); st.dataframe(piv_cpa.style.format("{:,.0f}"), use_container_width=True)
                    with c2: st.write("**비용**"); st.dataframe(piv_cost.style.format("{:,.0f}"), use_container_width=True)
                else:
                    st.warning("성별/연령 상세 데이터가 없습니다. (Unknown 데이터만 있거나 값이 없음)")
            else:
                st.warning("데이터에 'Gender' 또는 'Age' 컬럼이 없습니다.")
        else:
            st.warning("분석할 하단 데이터가 없습니다. (날짜 범위 불일치 또는 데이터 누락)")

    else:
        st.error("메인 데이터에서 해당 소재를 찾을 수 없습니다.")

else:
    # 전체 추세
    st.info("위 진단 리스트에서 '분석하기'를 누르면 상세 차트가 나옵니다.")
    if not df_main_fil.empty:
        agg = df_main_fil.groupby('Date').agg({'Conversions':'sum', 'Cost':'sum'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg['Date'], y=agg['Conversions'], name='전체 전환수', line=dict(color='black', width=3)))
        st.plotly_chart(fig, use_container_width=True)