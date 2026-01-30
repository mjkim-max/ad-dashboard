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
# 1. 메인 시트 (일별/소재별 - Meta는 여기에 성별/연령 포함)
MAIN_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"

# 2. 세트 시트 (Google 전용: 광고그룹별/성별/연령 데이터)
SET_SHEET_URL = "https://docs.google.com/spreadsheets/d/17z8PyqTdVFyF4QuTUKe6b0T_acWw2QbfvUP8DnTo5LM/edit?gid=29934845#gid=29934845"

# [세션 초기화]
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
        
        # Meta 데이터 보정
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].replace({'male': '남성', 'female': '여성', 'Male': '남성', 'Female': '여성'})
        
        return df
    except Exception as e:
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
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. 로직: 진단
# -----------------------------------------------------------------------------
def get_stats(df, days):
    if df.empty or 'Date' not in df.columns: return pd.DataFrame()
    start = df['Date'].max() - timedelta(days=days-1)
    sub = df[df['Date'] >= start]
    grp = sub.groupby(['Campaign', 'AdGroup', 'Creative_ID']).agg({'Cost':'sum','Conversions':'sum'}).reset_index()
    grp['CPA'] = np.where(grp['Conversions']>0, grp['Cost']/grp['Conversions'], np.inf)
    return grp

def run_diagnosis(df, target_cpa):
    if df.empty: return pd.DataFrame()
    s3 = get_stats(df, 3); s7 = get_stats(df, 7); s14 = get_stats(df, 14)
    if s3.empty: return pd.DataFrame() 

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
# 3. UI: 사이드바 & 필터링
# -----------------------------------------------------------------------------
df_main = load_main_sheet()
df_set = load_set_sheet()

st.sidebar.header("목표 설정")
target_cpa = st.sidebar.number_input("목표 CPA", 10000, step=1000)
st.sidebar.markdown("---")

# 기간 설정
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
c_m, c_g = st.sidebar.columns(2)
sel_pl = []
if c_m.checkbox("Meta", True): sel_pl.append("Meta")
if c_g.checkbox("Google", True): sel_pl.append("Google")

# 플랫폼 필터 적용
df_main_fil = df_main.copy()
if 'Platform' in df_main_fil.columns:
    df_main_fil = df_main_fil[df_main_fil['Platform'].isin(sel_pl)]

df_set_fil = df_set.copy()

# 날짜 필터 적용
if len(date_range) == 2:
    if not df_main_fil.empty and 'Date' in df_main_fil.columns:
        df_main_fil = df_main_fil[(df_main_fil['Date'].dt.date >= date_range[0]) & (df_main_fil['Date'].dt.date <= date_range[1])]
    
    if not df_set_fil.empty and 'Date' in df_set_fil.columns:
        df_set_fil = df_set_fil[(df_set_fil['Date'].dt.date >= date_range[0]) & (df_set_fil['Date'].dt.date <= date_range[1])]

# 캠페인/그룹/소재 필터
camps = ['전체']
if not df_main_fil.empty and 'Campaign' in df_main_fil.columns:
    camps += sorted(df_main_fil['Campaign'].unique().tolist())
sel_camp = st.sidebar.selectbox("캠페인", camps)

if sel_camp != '전체':
    if 'Campaign' in df_main_fil.columns: df_main_fil = df_main_fil[df_main_fil['Campaign'] == sel_camp]
    if 'Campaign' in df_set_fil.columns: df_set_fil = df_set_fil[df_set_fil['Campaign'] == sel_camp]

# -----------------------------------------------------------------------------
# 4. 메인 화면: 진단 리포트 (복구된 CPA 분석)
# -----------------------------------------------------------------------------
st.title("광고 성과 관리")

diag_res = run_diagnosis(df_main_fil, target_cpa)

if not diag_res.empty:
    camp_grps = diag_res.groupby('Campaign')
    sorted_camps = []
    
    # 정렬 로직 복구
    for c_name, grp in camp_grps:
        has_red = 'Red' in grp['Status_Color'].values
        has_yellow = 'Yellow' in grp['Status_Color'].values
        prio = 1 if has_red else 2 if has_yellow else 3
        h_col = ":red" if has_red else ":orange" if has_yellow else ":blue"
        
        # 캠페인 합계 계산
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
            # 캠페인 요약
            st.markdown("##### 캠페인 성과 요약")
            c_3d, c_7d, c_14d = st.columns(3)
            def fmt_head(label, cpa, cost, conv):
                return f"""<div style="line-height:1.4;"><strong>{label}</strong><br>CPA <strong>{cpa:,.0f}원</strong><br>비용 {cost:,.0f}원<br>전환 {conv:,.0f}</div>"""
            with c_3d: st.markdown(fmt_head("3일", *item['stats_3']), unsafe_allow_html=True)
            with c_7d: st.markdown(fmt_head("7일", *item['stats_7']), unsafe_allow_html=True)
            with c_14d: st.markdown(fmt_head("14일", *item['stats_14']), unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #f0f2f6;'>", unsafe_allow_html=True)
            
            # 소재 리스트
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
                    if st.button("분석하기", key=f"btn_{item['name']}_{r['Creative_ID']}_{idx}"):
                        st.session_state['chart_target_creative'] = r['Creative_ID']
                        st.rerun()
                st.markdown("<hr style='margin: 5px 0; border: none; border-top: 1px solid #f0f2f6;'>", unsafe_allow_html=True)
else:
    st.info("데이터가 부족하거나 설정된 기간 내 성과가 없습니다.")

# -----------------------------------------------------------------------------
# 5. 상세 분석 (기능 복구 + 로직 수정 완료)
# -----------------------------------------------------------------------------
st.markdown("### 📊 상세 분석")

target_creative = st.session_state['chart_target_creative']

# 지표 선택 및 색상 설정 복구
c_freq, c_opts, c_norm = st.columns([1, 2, 1])
freq_option = c_freq.radio("집계 기준", ["1일", "3일", "7일"], horizontal=True)
freq_map = {"1일": "D", "3일": "3D", "7일": "W"}
metrics = c_opts.multiselect("지표 선택", ['Impressions', 'Clicks', 'CTR', 'CPM', 'CPC', 'CPA', 'Cost', 'Conversions', 'CVR', 'ROAS'], default=['Conversions', 'CPA', 'CTR', 'Impressions'])
use_norm = c_norm.checkbox("데이터 정규화 (0-100%)", value=True)

if target_creative:
    st.success(f"선택한 소재: **{target_creative}**")
    if st.button("초기화"):
        st.session_state['chart_target_creative'] = None
        st.rerun()

    # 1. 꺾은선 그래프 (무조건 메인 시트의 소재 데이터 기준)
    chart_data = df_main_fil[df_main_fil['Creative_ID'] == target_creative]
    
    if not chart_data.empty:
        agg = chart_data.set_index('Date').groupby(pd.Grouper(freq=freq_map[freq_option])).agg({
            'Cost': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
        }).reset_index().sort_values('Date', ascending=True)

        # 지표 계산
        agg['CPA'] = np.where(agg['Conversions']>0, agg['Cost']/agg['Conversions'], 0)
        agg['CPM'] = np.where(agg['Impressions']>0, agg['Cost']/agg['Impressions']*1000, 0)
        agg['CTR'] = np.where(agg['Impressions']>0, agg['Clicks']/agg['Impressions']*100, 0)
        agg['CPC'] = np.where(agg['Clicks']>0, agg['Cost']/agg['Clicks'], 0)
        agg['CVR'] = np.where(agg['Clicks']>0, agg['Conversions']/agg['Clicks']*100, 0)
        agg['ROAS'] = np.where(agg['Cost']>0, agg['Conversion_Value']/agg['Cost']*100, 0)

        # 그래프 그리기 (색상 복구)
        fig = go.Figure()
        style_map = {
            'Conversions': {'color': 'black', 'width': 3},
            'CPA': {'color': 'red', 'width': 3},
            'CTR': {'color': 'blue', 'width': 2},
            'Impressions': {'color': 'green', 'width': 2}
        }
        
        for m in metrics:
            y_data = agg[m]
            y_plot = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100 if use_norm and y_data.max() > 0 else y_data
            style = style_map.get(m, {'color': None, 'width': 2})
            fig.add_trace(go.Scatter(x=agg['Date'], y=y_plot, mode='lines+markers', name=m, line=dict(color=style['color'], width=style['width']), customdata=y_data, hovertemplate=f"{m}: %{{customdata:,.2f}}"))

        fig.update_layout(height=450, hovermode='x unified', title=f"'{target_creative}' 추세 분석", plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        # 상세 데이터 표 복구
        table_df = agg.copy(); table_df['Date'] = table_df['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(table_df[['Date', 'CPA', 'Cost', 'Impressions', 'CPM', 'Clicks', 'Conversions', 'CTR', 'CPC', 'CVR', 'ROAS']], use_container_width=True, hide_index=True)

        # -------------------------------------------------------
        # 2. 막대 그래프 (성별/연령) - 핵심 로직 적용
        # -------------------------------------------------------
        target_platform = chart_data['Platform'].iloc[0]
        target_adgroup = chart_data['AdGroup'].iloc[0]
        
        demo_data = pd.DataFrame()
        source_msg = ""

        if target_platform == 'Google':
            # 구글: 세트 전용 시트에서 [광고 그룹]으로 검색 (날짜 필터 적용된 df_set_fil 사용)
            source_msg = f"🔎 구글 소재입니다. **'{target_adgroup}'** 광고 그룹 데이터(세트 시트)를 불러옵니다."
            if not df_set_fil.empty and 'AdGroup' in df_set_fil.columns:
                demo_data = df_set_fil[df_set_fil['AdGroup'] == target_adgroup]
        
        elif target_platform == 'Meta':
            # 메타: 메인 시트에서 [소재] 데이터 그대로 사용
            source_msg = f"🔎 메타 소재입니다. **'{target_creative}'** 소재 데이터(메인 시트)를 분석합니다."
            demo_data = chart_data
        
        else:
            source_msg = "플랫폼 정보 없음. 메인 시트 데이터를 사용합니다."
            demo_data = chart_data

        st.info(source_msg)

        if not demo_data.empty:
            if 'Gender' in demo_data.columns and 'Age' in demo_data.columns:
                valid_demo = demo_data[~demo_data['Gender'].isin(['Unknown', 'unknown', '알수없음'])]
                
                if not valid_demo.empty:
                    demo_agg = valid_demo.groupby(['Age', 'Gender']).agg({'Conversions':'sum', 'Cost':'sum'}).reset_index()
                    demo_agg['CPA'] = np.where(demo_agg['Conversions']>0, demo_agg['Cost']/demo_agg['Conversions'], 0)
                    
                    male = demo_agg[demo_agg['Gender'].str.contains('남성|Male', case=False)]
                    female = demo_agg[demo_agg['Gender'].str.contains('여성|Female', case=False)]
                    
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(x=male['Age'], y=male['Conversions'], name='남성', marker_color='#9EB9F3'))
                    fig_bar.add_trace(go.Bar(x=female['Age'], y=female['Conversions'], name='여성', marker_color='#F8C8C8'))
                    fig_bar.update_layout(title="성별/연령별 전환수", barmode='group', height=350)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    piv_cpa = demo_agg.pivot_table(index='Gender', columns='Age', values='CPA', aggfunc='sum', fill_value=0)
                    piv_cost = demo_agg.pivot_table(index='Gender', columns='Age', values='Cost', aggfunc='sum', fill_value=0)
                    with c1: st.write("**CPA**"); st.dataframe(piv_cpa.style.format("{:,.0f}"), use_container_width=True)
                    with c2: st.write("**비용**"); st.dataframe(piv_cost.style.format("{:,.0f}"), use_container_width=True)
                else:
                    st.warning("성별/연령 상세 데이터가 없습니다. (Unknown 제외)")
            else:
                st.warning("데이터에 'Gender' 또는 'Age' 컬럼이 없습니다.")
        else:
            st.warning("분석할 하단 데이터가 없습니다. (날짜 범위 또는 광고그룹명 매칭 확인 필요)")

    else:
        st.error("메인 데이터에서 해당 소재를 찾을 수 없습니다.")

else:
    # 초기 진입 시 전체 통합 차트 (지표/색상 적용)
    st.info("위 리스트에서 '분석하기'를 누르면 상세 차트가 나옵니다.")
    if not df_main_fil.empty and 'Date' in df_main_fil.columns:
        agg = df_main_fil.set_index('Date').groupby(pd.Grouper(freq=freq_map[freq_option])).agg({
            'Cost': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Conversions': 'sum', 'Conversion_Value': 'sum'
        }).reset_index().sort_values('Date', ascending=True)
        
        # 지표 계산
        agg['CPA'] = np.where(agg['Conversions']>0, agg['Cost']/agg['Conversions'], 0)
        agg['CPM'] = np.where(agg['Impressions']>0, agg['Cost']/agg['Impressions']*1000, 0)
        agg['CTR'] = np.where(agg['Impressions']>0, agg['Clicks']/agg['Impressions']*100, 0)
        agg['CPC'] = np.where(agg['Clicks']>0, agg['Cost']/agg['Clicks'], 0)
        agg['CVR'] = np.where(agg['Clicks']>0, agg['Conversions']/agg['Clicks']*100, 0)
        agg['ROAS'] = np.where(agg['Cost']>0, agg['Conversion_Value']/agg['Cost']*100, 0)

        fig = go.Figure()
        style_map = {
            'Conversions': {'color': 'black', 'width': 3},
            'CPA': {'color': 'red', 'width': 3},
            'CTR': {'color': 'blue', 'width': 2},
            'Impressions': {'color': 'green', 'width': 2}
        }
        for m in metrics:
            y_data = agg[m]
            y_plot = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * 100 if use_norm and y_data.max() > 0 else y_data
            style = style_map.get(m, {'color': None, 'width': 2})
            fig.add_trace(go.Scatter(x=agg['Date'], y=y_plot, mode='lines+markers', name=m, line=dict(color=style['color'], width=style['width']), customdata=y_data, hovertemplate=f"{m}: %{{customdata:,.2f}}"))
            
        fig.update_layout(title="전체 통합 성과 추세", height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)