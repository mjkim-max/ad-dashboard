import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 관리 BI", page_icon="📊", layout="wide")

# [주소 설정]
META_SHEET_URL = "https://docs.google.com/spreadsheets/d/13PG6s372l1SucujsACowlihRqOl8YDY4wCv_PEYgPTU/edit?gid=29934845#gid=29934845"
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1jEB4zTYPb2mrxZGXriju6RymHo1nEMC8QIVzqgiHwdg/edit?gid=141038195#gid=141038195"

# [세션 상태 초기화]
if 'chart_target_creative' not in st.session_state:
    st.session_state['chart_target_creative'] = None

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
        '상태': 'Status', '소재 상태': 'Status', '광고 상태': 'Status',
        '성별': 'Gender', 'Gender': 'Gender',
        '연령': 'Age', 'Age': 'Age', 'Age Group': 'Age'
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
    
    # [가상 데이터 생성] 성별/연령 컬럼이 없으면 데모용 랜덤 생성 (실제 데이터 있으면 무시됨)
    if 'Gender' not in df.columns:
        np.random.seed(42)
        df['Gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    if 'Age' not in df.columns:
        np.random.seed(42)
        df['Age'] = np.random.choice(['10대', '20대', '30대', '40대', '50대', '60대 이상'], size=len(df))
            
    return df

# -----------------------------------------------------------------------------
# 2. 공통 사이드바 (메뉴 및 필터)
# -----------------------------------------------------------------------------
df_raw = load_data()

with st.sidebar:
    st.title("🎛️ 분석 메뉴")
    # [메뉴 분리]
    menu = st.radio("페이지 선택", ["📊 종합 성과 진단", "🎯 타겟 & 페르소나 분석"])
    
    st.markdown("---")
    st.header("기간 설정")
    preset = st.selectbox("기간선택", ["오늘", "어제", "최근 3일", "최근 7일", "최근 14일", "최근 30일", "이번 달", "지난 달", "최근 90일"])
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
    date_range = st.date_input("날짜범위", [s, e])
    
    st.markdown("---")
    st.header("필터 설정")
    c_m, c_g = st.columns(2)
    sel_pl = []
    if c_m.checkbox("Meta", True): sel_pl.append("Meta")
    if c_g.checkbox("Google", True): sel_pl.append("Google")
    
    # 1차 필터링
    df_base = df_raw.copy()
    if 'Platform' in df_base.columns: df_base = df_base[df_base['Platform'].isin(sel_pl)]
    if len(date_range) == 2:
        df_base = df_base[(df_base['Date'].dt.date >= date_range[0]) & (df_base['Date'].dt.date <= date_range[1])]

    camps = ['전체'] + sorted(df_base['Campaign'].unique().tolist())
    sel_camp = st.selectbox("캠페인", camps)

    grps = ['전체']
    if sel_camp != '전체': grps = ['전체'] + sorted(df_base[df_base['Campaign'] == sel_camp]['AdGroup'].unique().tolist())
    sel_grp = st.selectbox("광고그룹", grps)

    # 최종 필터 데이터 (공통)
    target_df = df_base.copy()
    if sel_camp != '전체': target_df = target_df[target_df['Campaign'] == sel_camp]
    if sel_grp != '전체': target_df = target_df[target_df['AdGroup'] == sel_grp]


# =============================================================================
# [PAGE 1] 종합 성과 진단 (기존 기능 복구)
# =============================================================================
if menu == "📊 종합 성과 진단":
    
    # 진단용 함수
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
        m = s3.merge(s7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3', '_7'), how='left')
        m = m.merge(s14, on=['Campaign','AdGroup','Creative_ID'], how='left')
        m = m.rename(columns={'CPA': 'CPA_14', 'Cost': 'Cost_14', 'Conversions': 'Conversions_14'})
        m = m.fillna(0)
        
        results = []
        for _, row in m.iterrows():
            if row['Cost_3'] < 3000: continue
            cpa3, cpa7, cpa14 = row['CPA_3'], row['CPA_7'], row['CPA_14']
            
            # 절대평가 로직
            if (cpa14 <= target_cpa) and (cpa7 <= target_cpa) and (cpa3 <= target_cpa):
                status, title, detail = "Blue", "성과 우수 (Best)", "14일/7일/3일 모두 목표 달성."
            elif (cpa14 > target_cpa) and (cpa7 > target_cpa) and (cpa3 > target_cpa):
                status, title, detail = "Red", "종료 추천 (지속 부진)", "14일/7일/3일 모두 목표 미달성."
            else:
                status, title, detail = "Yellow", "판별 필요 (추이 확인)", "성과가 혼조세임."
                if cpa3 <= target_cpa: title = "성장 가능성 (반등)"; detail = "최근 3일 성과 개선."
                else: title = "관망 필요 (최근 저하)"; detail = "최근 3일 성과 하락."

            row['Status_Color'] = status; row['Diag_Title'] = title; row['Diag_Detail'] = detail
            results.append(row)
        return pd.DataFrame(results)

    st.title("📊 캠페인 성과 진단")
    
    # 목표 설정
    c1, c2 = st.columns(2)
    target_cpa = c1.number_input("목표 CPA", value=100000, step=1000)
    
    # 진단 실행
    diag_base = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - timedelta(days=14))]
    diag_res = run_diagnosis(diag_base, target_cpa)

    def get_color_box(color):
        if color == "Red": return st.error("🚨 종료 추천", icon="🚨")
        elif color == "Yellow": return st.warning("⚠️ 판별 필요", icon="⚠️")
        elif color == "Blue": return st.info("💎 성과 우수", icon="💎")
        else: return st.container(border=True)

    if not diag_res.empty:
        camp_grps = diag_res.groupby('Campaign')
        sorted_camps = []
        for c_name, grp in camp_grps:
            has_red = 'Red' in grp['Status_Color'].values
            has_yellow = 'Yellow' in grp['Status_Color'].values
            prio = 1 if has_red else 2 if has_yellow else 3
            color = ":red" if has_red else ":orange" if has_yellow else ":blue"
            
            stats = {
                '3': (grp['Cost_3'].sum(), grp['Conversions_3'].sum()),
                '7': (grp['Cost_7'].sum(), grp['Conversions_7'].sum()),
                '14': (grp['Cost_14'].sum(), grp['Conversions_14'].sum())
            }
            sorted_camps.append({'name': c_name, 'data': grp, 'prio': prio, 'color': color, 'stats': stats})
        
        sorted_camps.sort(key=lambda x: x['prio'])

        for item in sorted_camps:
            if sel_camp != '전체' and item['name'] != sel_camp: continue
            
            with st.expander(f"{item['color']}[{item['name']}]", expanded=False):
                # 요약
                st.markdown("##### 📊 캠페인 요약")
                cols = st.columns(3)
                for i, d in enumerate(['3', '7', '14']):
                    cost, conv = item['stats'][d]
                    cpa = cost/conv if conv>0 else 0
                    cols[i].metric(f"{d}일 CPA", f"{cpa:,.0f}원")
                    cols[i].caption(f"비용 {cost/10000:.1f}만 / 전환 {conv:,.0f}")
                
                st.divider()
                st.markdown("##### 📂 소재별 진단")
                
                for idx, (_, r) in enumerate(item['data'].iterrows()):
                    with get_color_box(r['Status_Color']):
                        c1, c2, c3 = st.columns([2.5, 1, 0.5])
                        with c1:
                            st.markdown(f"**{r['Creative_ID']}**")
                            def fmt(l, cpa, cost, conv): 
                                cpa_val = "∞" if cpa == np.inf else f"{cpa:,.0f}"
                                return f"**{l}:** CPA [{cpa_val}원] / 비용 {cost:,.0f}원 / 전환 {conv:,.0f}"
                            st.markdown(fmt("3일", r['CPA_3'], r['Cost_3'], r['Conversions_3']))
                            st.markdown(fmt("7일", r['CPA_7'], r['Cost_7'], r['Conversions_7']))
                            st.markdown(fmt("14일", r['CPA_14'], r['Cost_14'], r['Conversions_14']))
                        with c2:
                            t = "red" if r['Status_Color']=="Red" else "blue" if r['Status_Color']=="Blue" else "orange"
                            st.markdown(f":{t}[**{r['Diag_Title']}**]")
                            st.caption(r['Diag_Detail'])
                        with c3:
                            # 버튼 키 유니크하게
                            if st.button("분석", key=f"btn_{item['name']}_{r['Creative_ID']}_{idx}"):
                                st.session_state['chart_target_creative'] = r['Creative_ID']
                                st.rerun()
    
    # 하단 상세 분석 (탭)
    st.markdown("---")
    st.subheader("2. 상세 분석")
    target_creative = st.session_state['chart_target_creative']
    chart_data = target_df.copy()
    
    if target_creative:
        st.info(f"🔎 **'{target_creative}'** 소재 분석 중 (기간: {date_range[0]}~{date_range[1]})")
        chart_data = target_df[target_df['Creative_ID'] == target_creative]
        if st.button("초기화"):
            st.session_state['chart_target_creative'] = None
            st.rerun()
            
    tab1, tab2 = st.tabs(["📈 시계열 추세", "📅 요일별 효율"])
    
    with tab1:
        c_freq, c_opts, c_norm = st.columns([1, 2, 1])
        freq = c_freq.radio("집계", ["1일","3일","7일"], horizontal=True)
        f_map = {"1일":"D", "3일":"3D", "7일":"W"}
        metrics = c_opts.multiselect("지표", ['Impressions','Clicks','CTR','CPM','CPA','Cost','Conversions','ROAS'], ['Impressions','CTR','CPM'])
        norm = c_norm.checkbox("정규화", True)
        
        if not chart_data.empty and metrics:
            agg = chart_data.set_index('Date').groupby(pd.Grouper(freq=f_map[freq])).agg({'Cost':'sum','Impressions':'sum','Clicks':'sum','Conversions':'sum','Conversion_Value':'sum'}).reset_index()
            # 지표계산
            agg['CPA'] = np.where(agg['Conversions']>0, agg['Cost']/agg['Conversions'], 0)
            agg['CPM'] = np.where(agg['Impressions']>0, agg['Cost']/agg['Impressions']*1000, 0)
            agg['CTR'] = np.where(agg['Impressions']>0, agg['Clicks']/agg['Impressions']*100, 0)
            agg['ROAS'] = np.where(agg['Cost']>0, agg['Conversion_Value']/agg['Cost']*100, 0)
            
            fig = go.Figure()
            for m in metrics:
                y = agg[m]
                y_plot = (y - y.min()) / (y.max() - y.min()) * 100 if norm and y.max()>0 else y
                fig.add_trace(go.Scatter(x=agg['Date'], y=y_plot, mode='lines+markers', name=m, customdata=y, hovertemplate=f"{m}: %{{customdata:,.2f}}"))
            
            fig.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(agg.sort_values('Date', ascending=False).style.format({'CPA':'{:,.0f}', 'Cost':'{:,.0f}', 'ROAS':'{:.0f}%'}), use_container_width=True)

    with tab2:
        if not chart_data.empty:
            dow = chart_data.copy()
            dow['Wk'] = dow['Date'].dt.day_name()
            order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            kr = {'Monday':'월','Tuesday':'화','Wednesday':'수','Thursday':'목','Friday':'금','Saturday':'토','Sunday':'일'}
            d_agg = dow.groupby('Wk').agg({'Cost':'sum','Conversions':'sum','Conversion_Value':'sum'}).reindex(order).reset_index()
            d_agg['CPA'] = np.where(d_agg['Conversions']>0, d_agg['Cost']/d_agg['Conversions'], 0)
            d_agg['ROAS'] = np.where(d_agg['Cost']>0, d_agg['Conversion_Value']/d_agg['Cost']*100, 0)
            d_agg['KR'] = d_agg['Wk'].map(kr)
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(d_agg, x='KR', y='CPA', title="요일별 CPA", color='CPA', color_continuous_scale='Reds'), use_container_width=True)
            c2.plotly_chart(px.bar(d_agg, x='KR', y='ROAS', title="요일별 ROAS", color='ROAS', color_continuous_scale='Blues'), use_container_width=True)


# =============================================================================
# [PAGE 2] 타겟 & 페르소나 분석 (요청 디자인 반영)
# =============================================================================
elif menu == "🎯 타겟 & 페르소나 분석":
    st.title("🎯 소재별 타겟 심층 분석")
    st.caption("성별/연령별 CPA와 핵심 지표를 비교 분석합니다.")
    
    # 1. 소재 선택
    creatives_list = sorted(target_df['Creative_ID'].unique())
    selected_creative = st.selectbox("분석할 소재를 선택하세요:", creatives_list)
    
    if selected_creative:
        st.divider()
        
        # 2. 기간 선택 (3일, 7일, 14일)
        col_header, col_radio = st.columns([3, 1])
        with col_header:
            st.markdown(f"### {selected_creative}")
        with col_radio:
            period_opt = st.radio("기간 선택", ["3일", "7일", "14일"], horizontal=True, label_visibility="collapsed")
        
        # 기간 필터링 로직
        max_dt = target_df['Date'].max()
        days_map = {"3일": 3, "7일": 7, "14일": 14}
        start_dt = max_dt - timedelta(days=days_map[period_opt]-1)
        
        # 해당 소재 + 기간 데이터
        cr_df = target_df[
            (target_df['Creative_ID'] == selected_creative) & 
            (target_df['Date'] >= start_dt)
        ]
        
        if cr_df.empty:
            st.warning("선택한 기간에 데이터가 없습니다.")
        else:
            # ---------------------------
            # 데이터 집계 (Age x Gender)
            # ---------------------------
            agg = cr_df.groupby(['Age', 'Gender']).agg({
                'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum'
            }).reset_index()
            agg['CPA'] = np.where(agg['Conversions']>0, agg['Cost']/agg['Conversions'], 0)
            
            # 시각화용 데이터 준비 (남/녀 분리)
            male_data = agg[agg['Gender'].str.contains('Male|남', case=False, na=False)]
            female_data = agg[agg['Gender'].str.contains('Female|여', case=False, na=False)]
            
            # ---------------------------
            # [시각화] CPA 막대 그래프 (상단)
            # ---------------------------
            st.markdown("#### CPA (낮을수록 좋음)")
            
            fig_cpa = go.Figure()
            
            # 남성 막대
            fig_cpa.add_trace(go.Bar(
                x=male_data['Age'], y=male_data['CPA'], name='남성', marker_color='#9EB9F3' # 파스텔 블루
            ))
            
            # 여성 막대
            fig_cpa.add_trace(go.Bar(
                x=female_data['Age'], y=female_data['CPA'], name='여성', marker_color='#F8C8C8' # 파스텔 핑크
            ))
            
            fig_cpa.update_layout(
                barmode='group',
                xaxis_title="연령대",
                yaxis_title="CPA (원)",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_cpa, use_container_width=True)
            
            # ---------------------------
            # [데이터 그리드] 하단 지표 표
            # ---------------------------
            # Pivot Table 생성 함수
            def create_pivot_view(metric, fmt="{:,.0f}"):
                piv = agg.pivot_table(index='Gender', columns='Age', values=metric, aggfunc='sum', fill_value=0)
                return piv.style.format(fmt)

            st.markdown("#### 전환수")
            st.dataframe(create_pivot_view('Conversions', "{:,.0f}"), use_container_width=True)
            
            st.markdown("#### 비용 (지출액)")
            st.dataframe(create_pivot_view('Cost', "{:,.0f}"), use_container_width=True)
            
            st.markdown("#### 노출수")
            st.dataframe(create_pivot_view('Impressions', "{:,.0f}"), use_container_width=True)