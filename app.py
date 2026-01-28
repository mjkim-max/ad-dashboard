import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# [SETUP] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="광고 성과 진단 대시보드", page_icon="🩺", layout="wide")

# [주소 설정] - (사용자님 시트 주소)
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

    # Meta Data
    try:
        csv_url = convert_google_sheet_url(META_SHEET_URL)
        df_meta = pd.read_csv(csv_url)
        df_meta = df_meta.rename(columns=rename_map)
        df_meta['Platform'] = 'Meta'
        if 'Status' not in df_meta.columns: df_meta['Status'] = 'On'
        dfs.append(df_meta)
    except: pass

    # Google Data
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
    
    # 타입 변환
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
# 2. 핵심 로직: 소재별 진단
# -----------------------------------------------------------------------------
def get_creative_stats(df, days):
    max_date = df['Date'].max()
    start_date = max_date - timedelta(days=days-1)
    filtered = df[df['Date'] >= start_date]
    
    stats = filtered.groupby(['Campaign', 'AdGroup', 'Creative_ID']).agg({
        'Cost': 'sum', 'Conversions': 'sum', 'Impressions': 'sum', 'Clicks': 'sum'
    }).reset_index()
    
    stats['CPA'] = np.where(stats['Conversions']>0, stats['Cost']/stats['Conversions'], np.inf)
    stats['CPM'] = np.where(stats['Impressions']>0, stats['Cost']/stats['Impressions']*1000, 0)
    stats['CTR'] = np.where(stats['Impressions']>0, stats['Clicks']/stats['Impressions']*100, 0)
    
    return stats, start_date

def diagnose_creatives(df, target_cpa):
    if df.empty: return pd.DataFrame()

    # 기간별 통계 계산
    stats_3, _ = get_creative_stats(df, 3)
    stats_7, _ = get_creative_stats(df, 7)
    stats_14, _ = get_creative_stats(df, 14)
    stats_all, _ = get_creative_stats(df, 9999) 

    # 데이터 병합
    merged = stats_3.merge(stats_7, on=['Campaign','AdGroup','Creative_ID'], suffixes=('_3', '_7'), how='left')
    merged = merged.merge(stats_14, on=['Campaign','AdGroup','Creative_ID'], how='left')
    merged = merged.rename(columns={'CPA': 'CPA_14', 'Cost': 'Cost_14', 'Conversions': 'Conversions_14'})
    merged = merged.merge(stats_all[['Campaign','AdGroup','Creative_ID']], on=['Campaign','AdGroup','Creative_ID'], how='left')
    
    # 결측치 및 Infinity 처리
    merged = merged.fillna(0)
    merged['CPA_3'] = merged['CPA_3'].replace(0, np.inf)
    merged['CPA_7'] = merged['CPA_7'].replace(0, np.inf)
    merged['CPA_14'] = merged['CPA_14'].replace(0, np.inf)

    results = []
    # 캠페인 내 Best CPA 계산 (상대평가용)
    campaign_best_cpa = merged[merged['Conversions_14'] > 0].groupby('Campaign')['CPA_14'].min().to_dict()

    for idx, row in merged.iterrows():
        # 비용이 너무 적으면(3천원 미만) 패스
        if row['Cost_3'] < 3000: continue 

        cpa_3, cpa_7, cpa_14 = row['CPA_3'], row['CPA_7'], row['CPA_14']
        cpm_3, cpm_7 = row['CPM_3'], row['CPM_7']
        ctr_3, ctr_7 = row['CTR_3'], row['CTR_7']
        camp_best = campaign_best_cpa.get(row['Campaign'], 99999999)

        status = "White"
        diag_title, diag_detail = "", ""

        # [진단 로직]
        
        # 1. 🔴 상대 평가 (에이스 독주 체제)
        # 내 성과는 별로인데, 캠페인 내에 압도적으로 좋은 놈(Best)이 있을 때
        if (cpa_3 > target_cpa) and (camp_best <= target_cpa * 0.9):
            status = "Red"
            diag_title = "종료 추천 (상대적 열위)"
            diag_detail = f"캠페인 내 Best 소재(CPA {camp_best:,.0f}원) 대비 효율 저조. 예산 낭비 방지."
        
        # 2. 🟡 타겟 확장 신호 (보류)
        # 7일은 괜찮았는데 3일이 안 좋음 + 근데 CPM과 CTR이 같이 떨어짐 (저가 입찰 탐색)
        elif (cpa_7 <= target_cpa * 1.2) and (cpa_3 > target_cpa) and (cpm_3 < cpm_7 * 0.9) and (ctr_3 < ctr_7 * 0.9):
            status = "Yellow"
            diag_title = "보류 (타겟 탐색 중)"
            diag_detail = "CPM/CTR 동반 하락(⬇️). 저가 입찰로 신규 타겟 탐색 신호 감지."

        # 3. 🔴 절대 평가 (지속 부진)
        # 14일, 7일, 3일 내내 안 좋음
        elif (cpa_14 > target_cpa) and (cpa_7 > target_cpa) and (cpa_3 > target_cpa):
            status = "Red"
            diag_title = "효율 저조 (지속 부진)"
            diag_detail = "최근 2주간 CPA 목표 미달성. 개선 가능성 낮음."

        # 4. 🟢 성과 개선 (반등) - 화면엔 Green/Blue 박스로 표시
        elif (cpa_7 > target_cpa) and (cpa_3 <= target_cpa):
            status = "Green"
            diag_title = "성과 개선 중 (반등)"
            diag_detail = "이전보다 효율 좋아짐 (골든 크로스)."

        # 5. 🔵 성과 우수 (Best)
        elif (cpa_3 <= target_cpa) and (cpa_7 <= target_cpa):
            status = "Blue"
            diag_title = "성과 우수 (Scale-up)"
            diag_detail = "목표 CPA 달성 중. 증액 검토 가능."

        # 6. 🟡 단순 하락 (흔들림)
        elif (cpa_7 <= target_cpa) and (cpa_3 > target_cpa):
            status = "Yellow"
            diag_title = "최근 흔들림 (주의)"
            diag_detail = "7일 성과 좋았으나 최근 3일 저하. 일시적 현상인지 확인."

        # 결과 저장
        row['Status_Color'] = status
        row['Diag_Title'] = diag_title
        row['Diag_Detail'] = diag_detail
        results.append(row)

    if not results: return pd.DataFrame()
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 3. 화면 렌더링
# -----------------------------------------------------------------------------
df = load_data()

st.sidebar.header("🎯 설정")
target_cpa = st.sidebar.number_input("목표 CPA (원)", value=100000, step=5000)

if 'Status' in df.columns: 
    df = df[df['Status'] == 'On']

st.title("🩺 캠페인별 성과 진단 리포트")
st.caption("색상 구분: 🔴빨강(종료/위험) / 🟡노랑(보류/주의) / 🔵파랑(우수) / 🟢초록(개선)")
st.divider()

if df.empty:
    st.error("데이터가 없습니다. (혹시 모든 광고가 Off 상태인가요?)")
    st.stop()

# 진단 실행
diagnosis_df = diagnose_creatives(df, target_cpa)

if diagnosis_df.empty:
    st.success("데이터는 있지만, 분석 대상(비용 3000원 이상)이 없거나 특이사항이 없습니다.")
    st.stop()

# 캠페인 단위로 묶기 및 정렬 (Red -> Blue -> Yellow -> White)
campaign_groups = diagnosis_df.groupby('Campaign')
sorted_campaigns = []

for campaign_name, group in campaign_groups:
    has_red = 'Red' in group['Status_Color'].values
    has_blue = 'Blue' in group['Status_Color'].values or 'Green' in group['Status_Color'].values
    
    priority = 3
    header_text = f"📂 {campaign_name}"
    
    if has_red: 
        priority = 1
        header_text = f"🚨 {campaign_name} (점검 필요)"
    elif has_blue: 
        priority = 2
        header_text = f"✨ {campaign_name} (우수/증액)"
        
    sorted_campaigns.append({'name': campaign_name, 'data': group, 'priority': priority, 'header': header_text})

sorted_campaigns.sort(key=lambda x: x['priority'])

# [핵심 기능] 색상별 박스 그리기 함수
def get_status_box(status_color):
    if status_color == "Red":
        return st.error(icon="🚨")   # 빨간 박스
    elif status_color == "Yellow":
        return st.warning(icon="✋") # 노란 박스
    elif status_color == "Blue":
        return st.info(icon="💎")    # 파란 박스
    elif status_color == "Green":
        return st.success(icon="📈") # 초록 박스
    else:
        return st.container(border=True) # 기본 박스

# 실제 화면 출력
for camp in sorted_campaigns:
    # 캠페인 헤더 (Red가 있으면 자동으로 펼치기)
    with st.expander(camp['header'], expanded=(camp['priority']==1)):
        
        for _, row in camp['data'].iterrows():
            
            # 여기서 박스 색상을 결정해서 그립니다
            status_box = get_status_box(row['Status_Color'])
            
            with status_box:
                col_left, col_right = st.columns([1.3, 1])
                
                # 왼쪽: 데이터 수치
                with col_left:
                    st.markdown(f"**{row['Creative_ID']}**")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1: 
                        val_3 = "∞" if row['CPA_3'] == np.inf else f"{row['CPA_3']/10000:.1f}만"
                        st.markdown(f"**3일:** {val_3}")
                    with c2: 
                        val_7 = "∞" if row['CPA_7'] == np.inf else f"{row['CPA_7']/10000:.1f}만"
                        st.caption(f"7일: {val_7}")
                    with c3: 
                        val_14 = "∞" if row['CPA_14'] == np.inf else f"{row['CPA_14']/10000:.1f}만"
                        st.caption(f"14일: {val_14}")

                # 오른쪽: AI 진단 내용
                with col_right:
                    if row['Diag_Title']:
                        st.markdown(f"**{row['Diag_Title']}**")
                        st.caption(row['Diag_Detail'])
                        
                        # 보조 지표 (CPM, CTR) 화살표
                        if row['CPM_3'] > 0:
                            cpm_arrow = "⬇️" if row['CPM_3'] < row['CPM_7'] else "⬆️"
                            ctr_arrow = "⬇️" if row['CTR_3'] < row['CTR_7'] else "⬆️"
                            st.caption(f"신호: CPM {cpm_arrow} / CTR {ctr_arrow}")