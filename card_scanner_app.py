"""
名片辨識系統 - Streamlit 網頁版 (最終修復版：延遲載入)
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
# ⚠️ 這裡已移除 load_ocr_reader 函數和 @st.cache_resource
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
        # 使用 Denoising 和 CLAHE 進行圖像增強
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
        
        # 1. 提取標準格式資訊
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
        
        # 2. 猜測非標準格式資訊 (姓名, 職稱, 公司)
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
        
        # ⚠️ 這裡已修復語法錯誤
        if len(non_contact_lines) >= 3 and not info['company']:
            info['company'] = non_contact_lines[2]
        
        # 3. 猜測地址和國家
        address_keywords = ['street', 'road', 'avenue', 'ave', 'blvd', 'floor',
                           'building', 'tower', 'plaza', 'no.']
        for line in lines:
            if any(keyword in line.lower() for keyword in address_keywords):
                info['address'] = line
                break
        
        countries = {
            'singapore': 'Singapore', 'malaysia': 'Malaysia', 'thailand': 'Thailand',
            'indonesia': 'Indonesia', 'vietnam': 'Vietnam', 'philippines': 'Philippines',
            'taiwan': 'Taiwan', 'hong kong': 'Hong Kong', 'japan': 'Japan', 'korea': 'South Korea',
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
    if 'show_all' not in st.session_state:
        st.session_state.show_all = False
    
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
        4. **點擊「🚀 開始辨識」時才會載入 OCR 模型。**
        5. 檢查並修改資料
        6. 點擊「存入 Excel」
        """)
        
    
    # =========================================================================
    # ⚠️ 這裡移除了 EasyOCR 的全局載入邏輯，防止啟動時崩潰！
    # =========================================================================
    
    uploaded_file = st.file_uploader(
        "📸 拍照或上傳名片圖片", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="支援格式: JPG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # 初始化旋轉角度
        if 'rotation' not in st.session_state:
            st.session_state.rotation = 0
        
        # 讀取圖片
        if 'original_image' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_file.name:
            st.session_state.original_image = Image.open(uploaded_file)
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.rotation = 0
            # 重置辨識狀態，避免混亂
            st.session_state.card_info = None
            st.session_state.raw_text = None

        
        # 套用旋轉
        image = st.session_state.original_image.rotate(st.session_state.rotation, expand=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 原始圖片")
            # 由於 st.image 提示 use_container_width 將被移除，我們使用 width='stretch'
            st.image(image, width='stretch')
            
            # 旋轉按鈕
            rot_col1, rot_col2, rot_col3, rot_col4 = st.columns(4)
            with rot_col1:
                if st.button("↶ 90°"):
                    st.session_state.rotation = (st.session_state.rotation + 90) % 360
                    st.rerun()
            with rot_col2:
                if st.button("↷ -90°"):
                    st.session_state.rotation = (st.session_state.rotation - 90) % 360
                    st.rerun()
            with rot_col3:
                if st.button("↻ 180°"):
                    st.session_state.rotation = (st.session_state.rotation + 180) % 360
                    st.rerun()
            with rot_col4:
                if st.button("🔄 重置"):
                    st.session_state.rotation = 0
                    st.rerun()
        
        with col2:
            st.subheader("🔍 辨識結果")
            if st.button("🚀 開始辨識", type="primary"):
                
                # ✅ 關鍵修復：延遲載入 EasyOCR 模型
                if 'scanner' not in st.session_state:
                    with st.spinner('正在載入 OCR 模型 (首次載入可能耗時)...'):
                        try:
                            # 直接建立 Reader 並儲存到 Session State
                            reader = easyocr.Reader(['en'], gpu=False) 
                            st.session_state.scanner = BusinessCardScanner(reader)
                        except Exception as e:
                            st.error(f"❌ 模型載入失敗，請檢查依賴: {e}")
                            return

                with st.spinner('正在辨識中...'):
                    text, _ = st.session_state.scanner.extract_text(image)
                    info = st.session_state.scanner.parse_info(text)
                    st.session_state.card_info = info
                    st.session_state.raw_text = text
                st.success("✅ 辨識完成!")
        
        if st.session_state.get('card_info') is not None:
            st.markdown("---")
            st.subheader("✏️ 編輯資料")
            
            col1, col2 = st.columns(2)
            
            # 確保使用 st.session_state.card_info.get() 來處理可能的 None 值
            with col1:
                name = st.text_input("姓名", value=st.session_state.card_info.get('name', ''))
                title = st.text_input("職稱", value=st.session_state.card_info.get('title', ''))
                company = st.text_input("公司", value=st.session_state.card_info.get('company', ''))
                country_options = ['', 'Singapore', 'Malaysia', 'Thailand', 'Indonesia', 'Vietnam', 'Philippines', 'Brunei', 'Myanmar', 'Cambodia', 'Laos', 'Taiwan', 'Hong Kong', 'Japan', 'South Korea', 'China', 'India', 'Australia', 'New Zealand', 'United States', 'United Kingdom', 'Other']
                country_index = country_options.index(st.session_state.card_info.get('country', '')) if st.session_state.card_info.get('country', '') in country_options else 0
                country = st.selectbox("國家", options=country_options, index=country_index)
                email = st.text_input("Email", value=st.session_state.card_info.get('email', ''))
            
            with col2:
                phone = st.text_input("電話", value=st.session_state.card_info.get('phone', ''))
                mobile = st.text_input("手機", value=st.session_state.card_info.get('mobile', ''))
                website = st.text_input("網站", value=st.session_state.card_info.get('website', ''))
                linkedin = st.text_input("LinkedIn", value=st.session_state.card_info.get('linkedin', ''))
            
            address = st.text_area("地址", value=st.session_state.card_info.get('address', ''))
            
            if st.button("💾 存入 Excel", type="primary"):
                card_data = {
                    'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'name': name, 'title': title, 'company': company, 'country': country, 'email': email,
                    'phone': phone, 'mobile': mobile, 'website': website,
                    'linkedin': linkedin, 'address': address
                }
                
                try:
                    result = save_to_excel(card_data, excel_file)
                    if result == 'updated':
                        st.success(f"✅ 已更新現有名片資料到 {excel_file}")
                    else:
                        st.success(f"✅ 已新增名片到 {excel_file}")
                    # 儲存後清除暫存，準備下一個掃描
                    st.session_state.card_info = None
                    st.session_state.raw_text = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 儲存失敗: {e}")
            
            with st.expander("📄 查看原始辨識文字"):
                st.text(st.session_state.raw_text)
    
    # 🔒 保護資料顯示區塊
    if st.session_state.get('show_all', False) and st.session_state.authenticated:
        st.markdown("---")
        st.subheader("📊 所有已儲存的名片")
        
        if Path(excel_file).exists():
            df = pd.read_excel(excel_file)
            
            st.dataframe(df, width='stretch')
            
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="⬇️ 下載 Excel",
                    data=buffer,
                    file_name=excel_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                if st.button("🔄 刪除重複名片"):
                    df_cleaned = df.sort_values('scan_date', ascending=False)
                    df_cleaned = df_cleaned.drop_duplicates(subset=['name', 'company'], keep='first')
                    df_cleaned = df_cleaned.sort_values('scan_date', ascending=False)
                    df_cleaned.to_excel(excel_file, index=False)
                    st.success(f"✅ 已刪除 {len(df) - len(df_cleaned)} 張重複名片")
                    st.rerun()
            
            with col3:
                if st.button("✏️ 編輯/刪除模式"):
                    st.session_state.edit_mode = True
                    st.rerun()
            
            if st.session_state.get('edit_mode', False):
                st.markdown("---")
                st.subheader("✏️ 編輯或刪除名片")
                
                name_options = [f"{row['name']} - {row['company']}" for idx, row in df.iterrows()]
                selected = st.selectbox("選擇要編輯的名片", options=name_options)
                
                if selected:
                    selected_idx = name_options.index(selected)
                    row = df.iloc[selected_idx]
                    
                    col1, col2 = st.columns(2)
                    
                    # 為了避免重複的 widget key 錯誤，我們使用唯一的 key
                    with col1:
                        edit_name = st.text_input("姓名", value=row['name'], key="edit_name")
                        edit_title = st.text_input("職稱", value=row['title'], key="edit_title")
                        edit_company = st.text_input("公司", value=row['company'], key="edit_company")
                        
                        country_options = ['', 'Singapore', 'Malaysia', 'Thailand', 'Indonesia', 'Vietnam', 'Philippines', 'Brunei', 'Myanmar', 'Cambodia', 'Laos', 'Taiwan', 'Hong Kong', 'Japan', 'South Korea', 'China', 'India', 'Australia', 'New Zealand', 'United States', 'United Kingdom', 'Other']
                        current_country = row['country'] if row['country'] in country_options else ''
                        country_index = country_options.index(current_country)
                        edit_country = st.selectbox("國家", options=country_options, index=country_index, key="edit_country")
                        
                        edit_email = st.text_input("Email", value=row['email'], key="edit_email")
                    
                    with col2:
                        edit_phone = st.text_input("電話", value=row['phone'], key="edit_phone")
                        edit_mobile = st.text_input("手機", value=row['mobile'], key="edit_mobile")
                        edit_website = st.text_input("網站", value=row['website'], key="edit_website")
                        edit_linkedin = st.text_input("LinkedIn", value=row['linkedin'], key="edit_linkedin")
                    
                    edit_address = st.text_area("地址", value=row['address'], key="edit_address")
                    
                    col1, col2, col3 = st.columns([2, 2, 2])
                    
                    with col1:
                        if st.button("💾 儲存修改", type="primary"):
                            # 使用索引進行更新
                            df.loc[selected_idx, 'name'] = edit_name
                            df.loc[selected_idx, 'title'] = edit_title
                            df.loc[selected_idx, 'company'] = edit_company
                            df.loc[selected_idx, 'country'] = edit_country
                            df.loc[selected_idx, 'email'] = edit_email
                            df.loc[selected_idx, 'phone'] = edit_phone
                            df.loc[selected_idx, 'mobile'] = edit_mobile
                            df.loc[selected_idx, 'website'] = edit_website
                            df.loc[selected_idx, 'linkedin'] = edit_linkedin
                            df.loc[selected_idx, 'address'] = edit_address
                            df.to_excel(excel_file, index=False)
                            st.success("✅ 已儲存修改")
                            st.session_state.edit_mode = False
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ 刪除此名片", type="secondary"):
                            df = df.drop(selected_idx).reset_index(drop=True)
                            df.to_excel(excel_file, index=False)
                            st.success("✅ 已刪除名片")
                            st.session_state.edit_mode = False
                            st.rerun()
                    
                    with col3:
                        if st.button("❌ 取消"):
                            st.session_state.edit_mode = False
                            st.rerun()
        
        if st.button("🔙 返回"):
            st.session_state.show_all = False
            st.session_state.edit_mode = False
            st.rerun()


if __name__ == '__main__':
    main()
