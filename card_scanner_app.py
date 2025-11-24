"""
名片辨識系統 - Streamlit 網頁版 v17 (已修復 EasyOCR 啟動崩潰)
支援手機和電腦使用 + 圖片旋轉功能

執行方式:
python -m streamlit run card_scanner_app.py
"""

import streamlit as st
import easyocr
from PIL import Image
import cv2
import numpy as np
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import io


st.set_page_config(
    page_title="名片辨識系統",
    page_icon="📇",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {padding: 1rem;}
    /* Streamlit 提示：'use_container_width' 將被移除，建議替換為 'width' */
    .stButton>button {width: 100%; height: 3rem; font-size: 1.2rem;}
    h1 {font-size: 1.8rem !important;}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# ⚠️ 這裡不再有 load_ocr_reader 函數和 @st.cache_resource
# 避免 App 啟動時記憶體崩潰
# =========================================================================


class BusinessCardScanner:
    def __init__(self, reader):
        self.reader = reader
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}',
            'website': r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?',
            'linkedin': r'linkedin\.com/in/[a-zA-Z0-9-]+',
        }
    
    def preprocess_image(self, image):
        img_array = np.array(image)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        return enhanced
    
    def extract_text(self, image):
        processed_img = self.preprocess_image(image)
        results = self.reader.readtext(processed_img)
        text_lines = [result[1] for result in results]
        text = '\n'.join(text_lines)
        return text, results
    
    def parse_info(self, text):
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        info = {
            'name': '', 'title': '', 'company': '', 'email': '',
            'phone': '', 'mobile': '', 'website': '', 'address': '', 'linkedin': '', 'country': ''
        }
        
        email_match = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email_match:
            info['email'] = email_match.group()
        
        phone_matches = re.findall(self.patterns['phone'], text)
        if phone_matches:
            valid_phones = [p for p in phone_matches if len(re.sub(r'[^\d]', '', p)) >= 8]
            if len(valid_phones) >= 1:
                info['phone'] = valid_phones[0]
            if len(valid_phones) >= 2:
                info['mobile'] = valid_phones[1]
        
        website_match = re.search(self.patterns['website'], text, re.IGNORECASE)
        if website_match:
            info['website'] = website_match.group()
        
        linkedin_match = re.search(self.patterns['linkedin'], text, re.IGNORECASE)
        if linkedin_match:
            info['linkedin'] = linkedin_match.group()
        
        non_contact_lines = []
        for line in lines:
            if not any([
                re.search(self.patterns['email'], line, re.IGNORECASE),
                re.search(self.patterns['phone'], line),
                re.search(self.patterns['website'], line, re.IGNORECASE),
                'tel:' in line.lower(), 'fax:' in line.lower(),
            ]):
                non_contact_lines.append(line)
        
        if len(non_contact_lines) >= 1:
            info['name'] = non_contact_lines[0]
        
        if len(non_contact_lines) >= 2:
            potential_title = non_contact_lines[1]
            title_keywords = ['manager', 'director', 'ceo', 'cto', 'president', 
                            'executive', 'officer', 'head', 'lead', 'engineer',
                            'consultant', 'specialist', 'coordinator', 'supervisor']
            if any(keyword in potential_title.lower() for keyword in title_keywords):
                info['title'] = potential_title
            else:
                info['company'] = potential_title
        
        if len(non_contact_lines) >= 3 and not info['company']:
            info['company'] = non_contact_lines[2]
        
        address_keywords = ['street', 'road', 'avenue', 'ave', 'blvd', 'floor',
                           'singapore', 'malaysia', 'thailand', 'indonesia', 
                           'vietnam', 'philippines', 'building', 'tower', 'plaza']
        for line in lines:
            if any(keyword in line.lower() for keyword in address_keywords):
                info['address'] = line
                break
        
        # 偵測國家
        countries = {
            'singapore': 'Singapore', 'malaysia': 'Malaysia', 'thailand': 'Thailand',
            'indonesia': 'Indonesia', 'vietnam': 'Vietnam', 'philippines': 'Philippines',
            'brunei': 'Brunei', 'myanmar': 'Myanmar', 'cambodia': 'Cambodia',
            'laos': 'Laos', 'taiwan': 'Taiwan', 'hong kong': 'Hong Kong',
            'hongkong': 'Hong Kong', 'japan': 'Japan', 'korea': 'South Korea',
            'south korea': 'South Korea',
        }
        
        text_lower = text.lower()
        for keyword, country_name in countries.items():
            if keyword in text_lower:
                info['country'] = country_name
                break
        
        return info


def save_to_excel(card_data, excel_path='business_cards.xlsx', check_duplicate=True):
    df_new = pd.DataFrame([card_data])
    columns_order = ['scan_date', 'name', 'title', 'company', 'country', 'email', 
                    'phone', 'mobile', 'website', 'linkedin', 'address']
    df_new = df_new[columns_order]
    
    if Path(excel_path).exists():
        df_existing = pd.read_excel(excel_path)
        
        # 檢查是否有重複(相同姓名和公司)
        if check_duplicate:
            duplicate_mask = (df_existing['name'] == card_data['name']) & \
                           (df_existing['company'] == card_data['company'])
            
            if duplicate_mask.any():
                # 找到重複,用新資料覆蓋
                duplicate_idx = df_existing[duplicate_mask].index[0]
                df_existing.loc[duplicate_idx] = df_new.iloc[0]
                df_existing.to_excel(excel_path, index=False)
                return 'updated'
        
        # 沒有重複,新增資料
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(excel_path, index=False)
        return 'added'
    else:
        df_new.to_excel(excel_path, index=False)
        return 'added'


def main():
    st.title("📇 名片辨識系統")
    st.markdown("拍照上傳名片,自動辨識並存入 Excel")
    
    # 🔒 密碼設定與 Session State
    SECRET_PASSWORD = "YZsz45;#"  # <<< ⚠️ 請將此處替換為您的密碼！
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    with st.sidebar:
        st.header("⚙️ 設定")
        excel_file = st.text_input("Excel 檔案名稱", value="business_cards.xlsx")
        
        st.markdown("---")
        
        # 🔑 登入區塊
        st.subheader("🔑 資料管理登入")
        password_input = st.text_input("密碼", type="password", key="management_password")
        
        if st.button("登入管理"):
            if password_input == SECRET_PASSWORD:
                st.session_state.authenticated = True
                st.sidebar.success("✅ 登入成功")
            else:
                st.sidebar.error("❌ 密碼錯誤")
                st.session_state.authenticated = False
                
        # 只有在登入成功後才顯示數據管理選項
        if st.session_state.authenticated:
            st.markdown("---")
            if Path(excel_file).exists():
                df = pd.read_excel(excel_file)
                st.success(f"✅ 已儲存 {len(df)} 張名片")
                # 📊 只有登入後才能看到 '查看所有名片' 按鈕
                if st.button("📊 查看所有名片"):
                    st.session_state.show_all = True
            else:
                st.info("📝 尚未儲存任何名片")
            
            if st.button("登出管理"):
                st.session_state.authenticated = False
                st.session_state.show_all = False
                st.rerun() 

        st.markdown("---")
        st.markdown("### 📱 使用說明")
        st.markdown("""
        1. 點擊「拍照或上傳」
        2. 拍攝或選擇名片圖片
        3. 如需要可旋轉圖片調整方向
        4. 點擊「🚀 開始辨識」
        5. 檢查並修改資料
        6. 點擊「存入 Excel」
        """)
        
    
    # =========================================================================
    #
