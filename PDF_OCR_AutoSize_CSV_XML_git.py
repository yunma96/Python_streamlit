# 25.08.21 11:45 발리스 패턴 선택 시 CSV 저장 형식 변경 기능 추가
# 25.08.26 15:45 git 서버에 올려 streamlit cloud에서 사용하기 위해 Tesseract-OCR 폴더 지정하면 안됨

import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageDraw
import io
import os
import re
import pandas as pd
import traceback
import unicodedata
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import uuid
import csv

# --- 1. Tesseract 실행 파일의 경로를 지정 ---
import sys
import os

# EXE로 실행되었는지 확인하고 경로를 설정하는 함수
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- 1. 핵심 기능 함수 (OCR 분석) ---

def run_ocr_on_page(page, top_margin, bottom_margin, dpi, tile_width, tile_height, overlap, rotation_angle, enhance_image):
    """
    단일 페이지에 대해 OCR을 수행하고 텍스트를 반환하는 핵심 로직 (UI 출력 없음)
    """
    page_text = ""
    original_rect = page.rect
    processing_rect = fitz.Rect(
        original_rect.x0,
        original_rect.y0 + original_rect.height * top_margin,
        original_rect.x1,
        original_rect.y1 - original_rect.height * bottom_margin
    )
    
    num_tiles_x = int(processing_rect.width // tile_width) + 1
    num_tiles_y = int(processing_rect.height // tile_height) + 1

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            x0 = processing_rect.x0 + x * tile_width
            y0 = processing_rect.y0 + y * tile_height
            x1 = min(x0 + tile_width, processing_rect.x1)
            y1 = min(y0 + tile_height, processing_rect.y1)
            
            x0_overlap = max(processing_rect.x0, x0 - overlap)
            y0_overlap = max(processing_rect.y0, y0 - overlap)
            x1_overlap = min(processing_rect.x1, x1 + overlap)
            y1_overlap = min(processing_rect.y1, y1 + overlap)
            
            clip_rect = fitz.Rect(x0_overlap, y0_overlap, x1_overlap, y1_overlap)

            pix = page.get_pixmap(clip=clip_rect, dpi=dpi)
            if pix.width == 0 or pix.height == 0:
                continue

            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            if rotation_angle != 0:
                image = image.rotate(-rotation_angle, expand=True)

            if enhance_image:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)

            image = image.convert('L')
            image = image.point(lambda p: 0 if p < 180 else 255, '1')

            try:
                ocr_text = pytesseract.image_to_string(image, lang='eng', config='--psm 3')
                page_text += ocr_text + "\n"
            except Exception:
                continue
    return page_text


def extract_text_from_pdf_ocr(uploaded_file, top_margin, bottom_margin, dpi, tile_width, tile_height, overlap, rotation_angle, enhance_image):
    """
    업로드된 PDF 파일에서 텍스트를 추출하고, 타일 분할 시각화 이미지를 생성합니다. (UI 출력 포함)
    """
    Image.MAX_IMAGE_PIXELS = None
    full_text = ""
    visualization_images = []
    
    file_bytes = uploaded_file.getvalue()
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        st.info(f"총 {len(doc)} 페이지의 PDF 파일을 처리합니다.")
        
        progress_bar = st.progress(0)
        
        for i, page in enumerate(doc):
            page_num = i + 1
            st.write(f"페이지 {page_num}/{len(doc)} 처리 중...")

            vis_dpi = 150
            page_pix = page.get_pixmap(dpi=vis_dpi)
            page_image = Image.open(io.BytesIO(page_pix.tobytes("png"))).convert("RGBA")
            draw = ImageDraw.Draw(page_image)
            scale = vis_dpi / 72.0

            original_rect = page.rect
            processing_rect = fitz.Rect(
                original_rect.x0,
                original_rect.y0 + original_rect.height * top_margin,
                original_rect.x1,
                original_rect.y1 - original_rect.height * bottom_margin
            )
            
            proc_rect_coords = [
                (processing_rect.x0 - original_rect.x0) * scale,
                (processing_rect.y0 - original_rect.y0) * scale,
                (processing_rect.x1 - original_rect.x0) * scale,
                (processing_rect.y1 - original_rect.y0) * scale
            ]
            draw.rectangle(proc_rect_coords, outline="blue", width=3)
            
            page_text = run_ocr_on_page(page, top_margin, bottom_margin, dpi, tile_width, tile_height, overlap, rotation_angle, enhance_image)
            full_text += page_text
            
            num_tiles_x = int(processing_rect.width // tile_width) + 1
            num_tiles_y = int(processing_rect.height // tile_height) + 1
            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                    x0 = processing_rect.x0 + x * tile_width
                    y0 = processing_rect.y0 + y * tile_height
                    x0_overlap = max(processing_rect.x0, x0 - overlap)
                    y0_overlap = max(processing_rect.y0, y0 - overlap)
                    x1_overlap = min(processing_rect.x1, x0 + tile_width + overlap)
                    y1_overlap = min(processing_rect.y1, y0 + tile_height + overlap)
                    clip_rect = fitz.Rect(x0_overlap, y0_overlap, x1_overlap, y1_overlap)
                    tile_rect_coords = [
                        (clip_rect.x0 - original_rect.x0) * scale,
                        (clip_rect.y0 - original_rect.y0) * scale,
                        (clip_rect.x1 - original_rect.x0) * scale,
                        (clip_rect.y1 - original_rect.y0) * scale
                    ]
                    draw.rectangle(tile_rect_coords, outline="red", width=2)

            visualization_images.append(page_image)
            progress_bar.progress((i + 1) / len(doc))
        
        return full_text, visualization_images

    except Exception as e:
        st.error(f"PDF 처리 중 심각한 오류가 발생했습니다: {e}")
        st.error(traceback.format_exc())
        return None, []

def parse_all_potential_data(raw_text):
    """
    OCR 텍스트에서 위치와 ID를 찾아 데이터 쌍을 추출합니다.
    """
    location_pattern = r"[Kk]\s*[Mm]\s*[0-9OQ_]+\s*\+\s*[0-9OQ_]+"
    id_extraction_pattern = r"[A-Za-z0-9/_-]+"
    
    text_blocks = re.split(f'({location_pattern})', raw_text)
    
    structured_data = []
    
    for i in range(1, len(text_blocks), 2):
        location_text = text_blocks[i]
        id_block = text_blocks[i+1] if i + 1 < len(text_blocks) else ""

        processed_location = location_text.replace('O', '0').replace('Q', '0')
        clean_location = "".join(processed_location.split()).replace('_', '').upper()
        
        lines = [line.strip() for line in id_block.split('\n') if line.strip()]
        if not lines:
            continue
        
        first_line = lines[0]

        if re.search(r'\b(0r|Dr|And|Etc)\b', first_line, re.IGNORECASE):
            continue
        
        first_line = re.sub(r'\bN\b', '', first_line, flags=re.IGNORECASE)
        first_line = re.sub(r'\s+', ' ', first_line).strip()
        if not first_line:
            continue
        
        corrected_line = re.sub(r'\s*[=,ㅡ—,—ㅡ,—,ㅡ]\s*', '-', first_line)
        corrected_line = re.sub(r'--+', '-', corrected_line)
        
        corrected_line = re.sub(r'\b(FP|P)\s+(\d+)\b', r'\1-\2', corrected_line, flags=re.IGNORECASE)
        
        corrected_line = re.sub(r'(\b\d{4})\s*(iN|in|i|N)\b', r'\1', corrected_line, flags=re.IGNORECASE)

        balise_pattern = r'([A-Za-z0-9]{3})\s*[-/_\s]\s*(R|SR|ST|SH|P1|P2)\s*[-/_\s]\s*([\w/-]+)'
        balise_matches = re.findall(balise_pattern, corrected_line, re.IGNORECASE)
        
        for match in balise_matches:
            prefix = match[0].upper().replace('0', 'O')
            middle = match[1].upper()
            suffix_part = match[2].upper()

            if '/' in suffix_part:
                base_id = f"{prefix}-{middle}-"
                split_suffixes = suffix_part.split('/')
                for part in split_suffixes:
                    if part:
                        marker_id = f"{base_id}{part}"
                        structured_data.append({
                            'location': clean_location,
                            'marker_id': marker_id
                        })
            else:
                marker_id = f"{prefix}-{middle}-{suffix_part}"
                structured_data.append({
                    'location': clean_location,
                    'marker_id': marker_id
                })
        
        corrected_line = re.sub(balise_pattern, '', corrected_line, flags=re.IGNORECASE)

        parts = re.findall(id_extraction_pattern, corrected_line)
        all_id_parts = [p for p in parts if p.strip()]

        j = 0
        while j < len(all_id_parts):
            current_part = all_id_parts[j]
            
            if j + 1 < len(all_id_parts):
                next_part = all_id_parts[j+1]
                if re.fullmatch(r'[A-Z]+', current_part, re.IGNORECASE) and not re.fullmatch(r'\d+', next_part):
                    potential_id = f"{current_part}-{next_part}"
                    j += 2
                elif re.fullmatch(r'[A-Z]{1,3}', current_part, re.IGNORECASE) and re.fullmatch(r'\d+', next_part):
                    potential_id = f"{current_part} {next_part}"
                    j += 2
                elif re.search(r'/\d*$', current_part) and re.fullmatch(r'\d+', next_part):
                    potential_id = f"{current_part}{next_part}"
                    j += 2
                elif re.search(r'-\d$', current_part) and re.match(r'\d+/', next_part):
                    potential_id = f"{current_part}{next_part}"
                    j += 2
                else:
                    potential_id = current_part
                    j += 1
            else:
                potential_id = current_part
                j += 1
            
            corrected_id = potential_id

            if re.fullmatch(r"[-_/\\]+", corrected_id) or len(corrected_id) < 2:
                continue
            
            if len(corrected_id) >= 3 and len(set(corrected_id)) == 1 and not corrected_id.isalnum():
                continue

            if re.search(r'[^\w\s/-]', corrected_id):
                continue

            if corrected_id.startswith('-'):
                corrected_id = corrected_id[1:]
            
            corrected_id = re.sub(r'-[A-Za-z]{1,2}-?$', '', corrected_id, flags=re.IGNORECASE)

            if corrected_id:
                structured_data.append({
                    'location': clean_location,
                    'marker_id': corrected_id.upper()
                })

    final_data = []
    seen = set()
    for item in structured_data:
        item_tuple = (item['location'], item['marker_id'])
        if item_tuple not in seen:
            final_data.append(item)
            seen.add(item_tuple)

    return final_data

def filter_displayed_data(full_df, selected_patterns, min_id_length, show_all, require_digit):
    """
    사이드바의 필터 조건에 따라 표시할 데이터를 반환합니다.
    """
    if full_df is None or full_df.empty or 'marker_id' not in full_df.columns:
        return pd.DataFrame(columns=full_df.columns)

    filtered_df = full_df.copy()

    if not show_all:
        if require_digit:
            digit_mask = filtered_df['marker_id'].str.contains(r'\d', regex=True, na=False)
            filtered_df = filtered_df[digit_mask]

        if selected_patterns:
            try:
                id_pattern = "|".join([f"({p})" for p in selected_patterns])
                anchored_pattern = f"^({id_pattern})$"
                pattern_mask = filtered_df['marker_id'].str.match(anchored_pattern, na=False, case=False)
                filtered_df = filtered_df[pattern_mask]
            except re.error as e:
                st.error(f"오류: 패턴이 유효한 정규표현식이 아닙니다. 에러: {e}")
                return pd.DataFrame(columns=full_df.columns)
        else:
            pass
        
        length_mask = filtered_df['marker_id'].str.replace(r'\s+', '', regex=True).str.len() >= min_id_length
        filtered_df = filtered_df[length_mask]

    if not show_all and not selected_patterns:
        st.warning("결과를 보려면 '모든 잠재적 ID 표시'를 선택하거나, 하나 이상의 패턴 필터를 선택하세요.")
        return pd.DataFrame(columns=full_df.columns)
    
    return filtered_df

def safe_filename(filename):
    """
    파일 이름에서 유니코드 문자를 ASCII 문자로 변환하여 다운로드 오류를 방지합니다.
    """
    decomposed = unicodedata.normalize('NFD', filename)
    ascii_filename = decomposed.encode('ascii', 'ignore').decode('utf-8')
    safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', ascii_filename)
    return safe_name

# --- 3. 핵심 기능 함수 (XML 생성) ---

def parse_coord_string_xy(coord_str):
    """'{X=123.4, Y=567.8}' 형식의 문자열에서 X, Y 좌표를 float으로 추출합니다."""
    if not coord_str: return None, None
    match_x = re.search(r'X=\s*(-?[\d\.]+)', coord_str)
    match_y = re.search(r'Y=\s*(-?[\d\.]+)', coord_str)
    x = float(match_x.group(1)) if match_x else None
    y = float(match_y.group(1)) if match_y else None
    return x, y

def extract_data_from_xml_root(root):
    """
    파싱된 XML 루트에서 RailLine 데이터와 StartRealKPValue를 추출합니다.
    """
    try:
        start_kp_value = 0.0
        start_kp_prop = root.find(".//RailRealKPList/RealKP/property[@name='StartRealKPValue']")
        if start_kp_prop is not None and start_kp_prop.get('value'):
            try:
                start_kp_value = float(start_kp_prop.get('value'))
                st.success(f"성공: StartRealKPValue '{start_kp_value}' 값을 로드했습니다.")
            except (ValueError, TypeError):
                st.warning(f"경고: StartRealKPValue 값을 숫자로 변환할 수 없습니다. 기본값 0을 사용합니다.")
        else:
            st.warning("경고: StartRealKPValue를 XML에서 찾을 수 없습니다. 기본값 0을 사용합니다.")

        raillines_data = []
        for rail_line in root.findall('.//RailLines/RailLine'):
            rail_line_id = rail_line.get('Id')
            all_x_coords = []
            all_y_coords = []
            for line_segment in rail_line.findall('.//Lines/Line'):
                p1_prop = line_segment.find("property[@name='P1']")
                p2_prop = line_segment.find("property[@name='P2']")
                if p1_prop is not None and p2_prop is not None:
                    p1_x, p1_y = parse_coord_string_xy(p1_prop.get('value'))
                    p2_x, p2_y = parse_coord_string_xy(p2_prop.get('value'))
                    if p1_x is not None: all_x_coords.append(p1_x)
                    if p2_x is not None: all_x_coords.append(p2_x)
                    if p1_y is not None: all_y_coords.append(p1_y)
                    if p2_y is not None: all_y_coords.append(p2_y)
            if all_x_coords:
                start_x, end_x = min(all_x_coords), max(all_x_coords)
                furthest_y = max(all_y_coords, key=abs) if all_y_coords else 0
                raillines_data.append({'Id': rail_line_id, 'StartX': start_x, 'EndX': end_x, 'FurthestY': furthest_y})

        st.success(f"성공: {len(raillines_data)}개의 RailLine 전체 구간 정보를 로드했습니다.")
        return raillines_data, start_kp_value
    except Exception as e:
        st.error(f"오류: XML 데이터 추출 중 오류 발생: {e}")
        return None, 0

def find_railline_for_kp(kp_x, kp_y, raillines_data):
    """주어진 KP_X, KP_Y 값에 가장 적합한 RailLine을 찾습니다."""
    candidate_raillines = []
    for railline in raillines_data:
        if railline['StartX'] <= kp_x <= railline['EndX']:
            candidate_raillines.append(railline)
    
    if not candidate_raillines:
        return None
        
    best_match = min(candidate_raillines, key=lambda r: abs(r['FurthestY'] - kp_y))
    return best_match
    
def create_property(parent, name, value):
    """'property' 요소를 생성하고 부모 요소에 추가하는 함수"""
    prop = ET.SubElement(parent, 'property')
    prop.set('name', name)
    prop.set('value', str(value))
    return prop

def generate_signal_data(signal_csv_file, raillines_data, start_real_kp_value, insertion_point):
    """
    신호기 CSV를 읽어 기존 RailSignal 요소에 데이터를 추가합니다.
    SignalGroup ID가 기존 ID의 최댓값에 이어서 순차적으로 생성되도록 수정되었습니다.
    """
    try:
        csv_content = signal_csv_file.getvalue().decode('utf-8-sig')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        required_columns = ['KPPoint_X', 'KPPoint_Y', 'Direction', 'Code', 'DefaultCode', 'DisplayName', 'PoleType']
        # CSV 파일에 필수 열이 있는지 확인합니다.
        for col in required_columns:
            if col not in csv_reader.fieldnames:
                st.error(f"오류: 신호기 CSV 파일에 필수 열인 '{col}'가 없습니다.")
                return

        # XML에서 RailSignal 요소를 찾거나, 없으면 새로 생성합니다.
        rail_signal_root = insertion_point.find('RailSignal')
        if rail_signal_root is None:
            rail_signal_root = ET.SubElement(insertion_point, 'RailSignal', Id="RSG_1", Name="RSG_1")
            create_property(rail_signal_root, "Id", "RSG_1")
            create_property(rail_signal_root, "Name", "RSG_1")

        # RailSignal 요소 아래에 SignalGroupList를 찾거나, 없으면 새로 생성합니다.
        signal_group_list = rail_signal_root.find('SignalGroupList')
        if signal_group_list is None:
            signal_group_list = ET.SubElement(rail_signal_root, 'SignalGroupList')

        # --- 코드 수정 부분 시작 ---
        # 기존 SignalGroup ID의 최댓값을 찾습니다.
        max_id = -1
        existing_groups = signal_group_list.findall('SignalGroup')
        if existing_groups:
            for group in existing_groups:
                group_id_str = group.get('Id', 'SG_-1')
                try:
                    # 'SG_123'과 같은 ID에서 숫자 부분만 추출합니다.
                    numeric_part = int(re.search(r'\d+', group_id_str).group())
                    if numeric_part > max_id:
                        max_id = numeric_part
                except (ValueError, AttributeError):
                    # ID 형식이 예상과 다를 경우 건너뜁니다.
                    st.warning(f"경고: SignalGroup ID '{group_id_str}'의 숫자 부분을 분석할 수 없습니다.")
                    continue
        
        # 새 SignalGroup ID의 시작 번호를 설정합니다.
        start_index = max_id + 1
        # --- 코드 수정 부분 끝 ---

        # CSV 파일의 각 행을 순회하며 SignalGroup을 생성합니다.
        for i, row in enumerate(csv_reader):
            current_index = start_index + i
            
            # KPPoint_X와 KPPoint_Y 값을 계산합니다.
            final_kp_x = float(row['KPPoint_X']) - start_real_kp_value
            final_kp_y = float(row['KPPoint_Y']) * 600
            
            # 좌표에 맞는 RailLine을 찾습니다.
            matched_railline = find_railline_for_kp(final_kp_x, final_kp_y, raillines_data)
            rail_line_id, rail_length = ("N/A", "N/A")
            if matched_railline:
                rail_line_id = matched_railline['Id']
                rail_length = int(final_kp_x - matched_railline['StartX'])
            else:
                st.warning(f"신호기 경고 (행 {i+2}): KP_X 값 {final_kp_x}에 해당하는 라인을 찾을 수 없습니다.")

            # 새로운 SignalGroup 요소를 생성하고 ID를 부여합니다.
            sg = ET.SubElement(signal_group_list, 'SignalGroup', Id=f"SG_{current_index}")
            create_property(sg, "Id", f"SG_{current_index}")
            create_property(sg, "Name", f"SG_{current_index}")
            
            signals = ET.SubElement(sg, 'Signals')
            signal = ET.SubElement(signals, 'Signal')
            
            # CSV 데이터와 기본값을 사용하여 property를 설정합니다.
            create_property(signal, "Id", f"Signal_{current_index}")
            create_property(signal, "KPPoint", f"{{X={final_kp_x}, Y={final_kp_y}}}")
            create_property(signal, "RailLineID", rail_line_id)
            create_property(signal, "Direction", row['Direction'])
            create_property(signal, "Code", row['Code'])
            create_property(signal, "DefaultCode", row['DefaultCode'])
            create_property(signal, "DisplayName", row['DisplayName'])
            create_property(signal, "PoleType", row['PoleType'])
            create_property(signal, "RailLength", rail_length)
            
            # 고정된 기본 property들을 추가합니다.
            for name, value in [("Offset", "-300"), ("Type", "0"), ("IsSubLine", "False"), 
                                ("IsAutoChange", "True"), ("IsVertical", "False"), 
                                ("Color", "Color [Gray]"), ("Width", "2"), ("RailOffset", "-300"), 
                                ("RailHeightOffset", "30"), ("IsVisualText", "True")]:
                create_property(signal, name, value)

        st.success("성공: 신호기 CSV 파일에서 데이터를 처리하여 XML에 추가했습니다.")
    except Exception as e:
        st.error(f"오류: 신호기 데이터 처리 중 오류 발생: {e}")
        st.error(traceback.format_exc())


def generate_balis_data(balis_csv_file, raillines_data, start_real_kp_value, insertion_point):
    """발리스 CSV를 읽어 기존 RailBalis 요소에 데이터를 추가합니다."""
    try:
        csv_content = balis_csv_file.getvalue().decode('utf-8-sig')
        csv_reader = csv.DictReader(io.StringIO(csv_content))
        
        required_columns = ['KPPoint_X', 'KPPoint_Y', 'Direction', 'Name']
        for col in required_columns:
            if col not in csv_reader.fieldnames:
                st.error(f"오류: 발리스 CSV 파일에 필수 열인 '{col}'가 없습니다.")
                return

        rail_balis_root = insertion_point.find('RailBalis')
        if rail_balis_root is None:
            balis_id = str(uuid.uuid4())
            rail_balis_root = ET.SubElement(insertion_point, 'RailBalis', Id=balis_id, Name="")
            create_property(rail_balis_root, "Id", balis_id)
            create_property(rail_balis_root, "Name", "")
        
        balis_list = rail_balis_root.find('BalisList')
        if balis_list is None:
            balis_list = ET.SubElement(rail_balis_root, 'BalisList')
        
        start_index = len(balis_list.findall('Balis'))

        for i, row in enumerate(csv_reader):
            current_index = start_index + i
            final_kp_x = float(row['KPPoint_X']) - start_real_kp_value
            final_kp_y = float(row['KPPoint_Y'])

            matched_railline = find_railline_for_kp(final_kp_x, final_kp_y, raillines_data)
            rail_line_id, rail_length = ("N/A", "N/A")
            if matched_railline:
                rail_line_id = matched_railline['Id']
                rail_length = int(final_kp_x - matched_railline['StartX'])
            else:
                st.warning(f"발리스 경고 (행 {i+2}): KP_X 값 {final_kp_x}에 해당하는 라인을 찾을 수 없습니다.")

            balis = ET.SubElement(balis_list, 'Balis')
            create_property(balis, "Id", str(current_index))
            create_property(balis, "KPPoint", f"{{X={final_kp_x}, Y={final_kp_y}}}")
            create_property(balis, "Name", row['Name'])
            create_property(balis, "Direction", row['Direction'])
            create_property(balis, "RailLineId", rail_line_id)
            create_property(balis, "RailLength", rail_length)
            for name, value in [("SignalCount", "0"), ("TypeCode", "0"),
                                ("strLinkSignalName", ""), ("BalisType", "2"), ("ERTMS_Mode", "0"),
                                ("ERTMS_Level", "0"), ("ERTMS_StartKP", "0"), ("ERTMS_EndKP", "0"),
                                ("RailOffset", "0"), ("RailHeightOffset", "-23"), ("BalisActive", "True")]:
                create_property(balis, name, value)
        st.success("성공: 발리스 CSV 파일에서 데이터를 처리했습니다.")
    except Exception as e:
        st.error(f"오류: 발리스 데이터 처리 중 오류 발생: {e}")


# --- 4. Streamlit UI 및 상태 관리 ---
st.set_page_config(layout="wide")
st.title("📄 PDF OCR 및 패턴 분석기")


# try:
#     # Tesseract 실행 파일의 경로를 지정
#     # 위에서 --add-data 옵션으로 "Tesseract-OCR" 폴더를 추가했으므로
#     # EXE 내부에서는 해당 폴더 안의 tesseract.exe를 찾아야 함
#     tesseract_cmd_path = resource_path("Tesseract-OCR/tesseract.exe")
#     pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
#     # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Administrator\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# except Exception:
#     st.warning("Tesseract 경로를 찾을 수 없습니다. 스크립트 내에서 경로를 직접 설정해주세요.")

# 세션 상태 초기화
if 'full_results_df' not in st.session_state:
    st.session_state.full_results_df = None
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = None
if 'visualization_images' not in st.session_state:
    st.session_state.visualization_images = []
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""
if 'reliability_results_df' not in st.session_state:
    st.session_state.reliability_results_df = None
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = None
if 'data_with_selection' not in st.session_state:
    st.session_state.data_with_selection = None
if 'generated_xml' not in st.session_state:
    st.session_state.generated_xml = None

# 사이드바 UI
with st.sidebar:
    st.header("⚙️ 분석 설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    
    st.subheader("페이지 회전")
    rotation_angle = st.selectbox("회전 각도", [0, 90, 180, 270], index=0)

    st.subheader("영역 설정")
    top_margin = st.slider("상단 제외 비율(%)", 0, 100, 15) / 100.0
    bottom_margin = st.slider("하단 제외 비율(%)", 0, 100, 15) / 100.0
    
    st.subheader("OCR 설정")
    dpi = st.slider("DPI (해상도)", 100, 800, 550, 50)
    
    st.subheader("타일 설정")
    tile_width = st.slider("타일 너비", 100, 1500, 1000, 50)
    tile_height = st.slider("타일 높이", 100, 1500, 1000, 50)
    overlap = st.slider("타일 겹침 (pixels)", 0, 200, 20, 10)

    st.subheader("분석 실행")
    col1, col2 = st.columns(2)
    with col1:
        analyze_button = st.button("단일 분석", use_container_width=True)
    with col2:
        reliability_button = st.button("신뢰도 분석 시작", type="primary", use_container_width=True)

    with st.expander("신뢰도 분석 타일 설정"):
        tile_options_input = st.text_area(
            "테스트할 타일 크기 입력 (너비,높이)",
            value="100,100\n200,200\n300,300\n1000,1000",
            help="한 줄에 '너비,높이' 형식으로 하나씩 입력하세요."
        )

    st.markdown("---")
    st.header("📊 결과 필터링")
    
    show_all = st.checkbox("모든 잠재적 ID 표시", value=False)
    require_digit = st.checkbox("ID에 숫자 포함", value=True, disabled=show_all)
    min_id_length = st.slider("최소 ID 길이", 1, 10, 3, disabled=show_all)
    
    st.subheader("패턴 선택")
    
    predefined_patterns = {
        "똑똑한 범용 패턴 (추천)": r"[A-Za-z0-9]+[-_][A-Za-z0-9_/-]+",
        "'WD' 패턴": r"W\s*D\s*\d+",
        "'FP/P' 패턴": r"(FP|P)\s*[-_/]?\s*\d+",
        "신호기 패턴 (숫자 또는 숫자/숫자)": r"\b\d{4}\b\s*[-/]\s*\b\d{4}\b|\b\d{4}\b|(EOL)[-_]\s*\b\d{4}[A-Za-z]?\b",
        "발리스 패턴 (범용)": r"[A-Z]{3}-(R|SR|ST|SH|P1|P2)-[\w/-]+"
    }

    selected_patterns = [p for label, p in predefined_patterns.items() if st.checkbox(label, value=False, disabled=show_all)]

# --- 메인 화면 탭 ---
main_tab1, main_tab2, main_tab3 = st.tabs(["분석 결과", "CSV 편집기", "XML 생성기"])

with main_tab1:
    # --- 신뢰도 분석 로직 ---
    if reliability_button:
        if uploaded_file:
            results_with_source = defaultdict(set)
            file_bytes = uploaded_file.getvalue()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            tile_options = []
            if tile_options_input:
                lines = tile_options_input.strip().split('\n')
                for line in lines:
                    try:
                        width, height = map(int, line.strip().split(','))
                        tile_options.append({'width': width, 'height': height})
                    except ValueError:
                        st.error(f"잘못된 타일 크기 형식입니다: '{line}'.")
                        st.stop()
            if not tile_options:
                st.error("신뢰도 분석을 위한 타일 크기를 하나 이상 입력해주세요.")
                st.stop()
            
            quality_options = [{'enhance': True}, {'enhance': False}]
            total_tests = len(tile_options) * len(quality_options)

            with st.spinner(f"통합 신뢰도 분석 중... (총 테스트 개수: {total_tests})"):
                test_progress = st.progress(0)
                count = 0
                for tile_opt in tile_options:
                    for quality_opt in quality_options:
                        count += 1
                        current_tile_width = tile_opt['width']
                        current_tile_height = tile_opt['height']
                        current_enhance = quality_opt['enhance']
                        
                        quality_str = "향상" if current_enhance else "원본"
                        st.write(f"테스트 {count}/{total_tests}: {current_tile_width}x{current_tile_height} ({quality_str})...")
                        source_str = f"{current_tile_width}x{current_tile_height} ({quality_str})"

                        full_text = ""
                        for page in doc:
                            full_text += run_ocr_on_page(page, top_margin, bottom_margin, dpi, current_tile_width, current_tile_height, overlap, rotation_angle, current_enhance)
                        
                        parsed_data = parse_all_potential_data(full_text)
                        for item in parsed_data:
                            key = (item['location'], item['marker_id'])
                            results_with_source[key].add(source_str)
                        test_progress.progress(count / total_tests)

            st.success("신뢰도 분석 완료!")
            
            reliability_data = []
            for (location, marker_id), sources in results_with_source.items():
                data_entry = {
                    "location": location,
                    "marker_id": marker_id,
                    "발견 횟수": len(sources),
                    "발견 조건": ", ".join(sorted(list(sources)))
                }
                reliability_data.append(data_entry)
                
            if reliability_data:
                df = pd.DataFrame(reliability_data)
                st.session_state.reliability_results_df = df.sort_values(by="발견 횟수", ascending=False)
            else:
                st.session_state.reliability_results_df = pd.DataFrame()

            st.session_state.full_results_df = None
            st.session_state.uploaded_file_name = uploaded_file.name

        else:
            st.error("먼저 PDF 파일을 업로드해주세요.")

    # 단일 분석 시작 로직
    if analyze_button and uploaded_file:
        with st.spinner("PDF를 분석 중입니다. 잠시만 기다려주세요..."):
            enhance_image_single = True 
            raw_text, viz_images = extract_text_from_pdf_ocr(uploaded_file, top_margin, bottom_margin, dpi, tile_width, tile_height, overlap, rotation_angle, enhance_image_single)
            
            st.session_state.raw_text = raw_text
            st.session_state.visualization_images = viz_images
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.reliability_results_df = None

            if raw_text:
                parsed_data = parse_all_potential_data(raw_text)
                st.session_state.full_results_df = pd.DataFrame(parsed_data)
                st.success("분석 완료! 아래에서 결과를 확인하고 필터링할 수 있습니다.")
            else:
                st.session_state.full_results_df = None
                st.error("PDF에서 텍스트를 추출하지 못했습니다.")

    # --- 결과 표시 및 다운로드 로직 (공통) ---
    df_to_display = None
    analysis_type = ""
    if st.session_state.reliability_results_df is not None:
        df_to_display = st.session_state.reliability_results_df
        analysis_type = "신뢰도"
        st.header("🔬 통합 신뢰도 분석 결과")
    elif st.session_state.full_results_df is not None:
        df_to_display = st.session_state.full_results_df
        analysis_type = "단일"
        st.header("📊 단일 분석 결과")

    if df_to_display is not None:
        if not df_to_display.empty:
            
            if analysis_type == "신뢰도":
                max_count = 1
                if tile_options_input:
                    lines = tile_options_input.strip().split('\n')
                    tile_count = len([line for line in lines if line.strip()])
                    max_count = tile_count * 2
                
                min_discovery_count = st.slider(
                    "최소 발견 횟수 필터", 1, max_count, 1, 1,
                    help=f"설정한 값 이상으로 발견된 데이터만 표시합니다. (최대 {max_count}회)"
                )
                df_to_display = df_to_display[df_to_display['발견 횟수'] >= min_discovery_count]

            filtered_df = filter_displayed_data(df_to_display, selected_patterns, min_id_length, show_all, require_digit)
            
            total_count = len(df_to_display)
            filtered_count = len(filtered_df)
            st.info(f"필터링된 항목: {filtered_count} / {total_count} 개")

            if not filtered_df.empty:
                df_with_selection = filtered_df.copy()
                df_with_selection.insert(0, "선택", True)
                st.session_state.data_with_selection = st.data_editor(
                    df_with_selection, 
                    key=f"{analysis_type}_editor",
                    hide_index=True
                )

        else:
            st.warning("분석 결과 데이터가 없습니다.")

        if st.session_state.visualization_images:
            with st.expander("📊 타일 분할 시각화 보기"):
                for idx, img in enumerate(st.session_state.visualization_images):
                    st.image(img, caption=f"페이지 {idx + 1}", use_container_width=True)

        with st.expander("OCR로 추출된 전체 텍스트 보기 (디버깅용)"):
            st.text_area("", st.session_state.raw_text, height=400)

    elif (analyze_button or reliability_button) and not uploaded_file:
        st.error("분석을 시작하기 전에 PDF 파일을 업로드해주세요.")
    elif not analyze_button and not reliability_button:
        st.info("왼쪽 사이드바에서 PDF 파일을 업로드하고 설정을 조정한 뒤 분석 버튼을 눌러주세요.")


with main_tab2:
    st.header("✏️ CSV 편집기")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("분석 결과에서 선택한 데이터 가져오기"):
            if st.session_state.data_with_selection is not None and not st.session_state.data_with_selection.empty:
                selected_rows = st.session_state.data_with_selection[st.session_state.data_with_selection["선택"]]
                if not selected_rows.empty:
                    df_to_edit = selected_rows.drop(columns=["선택"])
                    
                    is_signal = predefined_patterns["신호기 패턴 (숫자 또는 숫자/숫자)"] in selected_patterns
                    is_balise = predefined_patterns["발리스 패턴 (범용)"] in selected_patterns

                    if is_signal:
                        csv_list = []
                        for _, row in df_to_edit.iterrows():
                            loc, mid = row['location'], row['marker_id']
                            kpx = '0'
                            match = re.match(r"KM(\d+)\+(\d+)", loc)
                            if match:
                                try:
                                    kpx = int(match.group(1) + match.group(2)) * 100
                                except (ValueError, TypeError): pass
                            
                            parts = re.split(r'[-/]', mid)
                            for part in parts:
                                part = part.strip()
                                if part:
                                    csv_list.append({'KPPoint_X': kpx,'KPPoint_Y': '0', 'Direction': '0', 'Code': '2102', 'DefaultCode': '2102', 'DisplayName': part, 'PoleType': '2'})
                        st.session_state.edited_df = pd.DataFrame(csv_list, columns=['KPPoint_X', 'KPPoint_Y', 'Direction', 'Code', 'DefaultCode', 'DisplayName', 'PoleType'])
                    
                    elif is_balise:
                        csv_list = []
                        for _, row in df_to_edit.iterrows():
                            loc, mid = row['location'], row['marker_id']
                            kpx = '0'
                            match = re.match(r"KM(\d+)\+(\d+)", loc)
                            if match:
                                try:
                                    kpx = int(match.group(1) + match.group(2)) * 100
                                except (ValueError, TypeError): pass
                            csv_list.append({'KPPoint_X': kpx, 'KPPoint_Y': '0', 'Direction': '0', 'Name': mid})
                        st.session_state.edited_df = pd.DataFrame(csv_list, columns=['KPPoint_X', 'KPPoint_Y', 'Direction', 'Name'])

                    else:
                        st.session_state.edited_df = df_to_edit

                    st.success(f"{len(st.session_state.edited_df)}개의 행을 가져왔습니다.")
                else:
                    st.warning("분석 결과 탭에서 가져올 데이터를 선택해주세요.")
            else:
                st.warning("먼저 분석을 실행하고 데이터를 선택해주세요.")

    with col2:
        uploaded_csv = st.file_uploader("새 CSV 파일 업로드", type=["csv"])
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.session_state.edited_df = df
            except Exception as e:
                st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
                st.session_state.edited_df = None


    if st.session_state.edited_df is not None:
        st.info("표를 직접 수정하거나, 행을 추가/삭제할 수 있습니다.")
        
        st.session_state.edited_df = st.data_editor(st.session_state.edited_df, num_rows="dynamic", key="csv_editor")
        
        st.markdown("---")
        
        if not st.session_state.edited_df.empty:
            csv_data = st.session_state.edited_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="수정된 CSV 다운로드",
                data=csv_data,
                file_name="edited_data.csv",
                mime='text/csv'
            )

with main_tab3:
    st.header("🔧 XML 생성기")
    
    main_xml_file = st.file_uploader("1. 원본 속성 XML 파일을 업로드하세요.", type=["xml"])
    signal_csv_file = st.file_uploader("2. 신호기 데이터 CSV 파일을 업로드하세요. (선택 사항)", type=["csv"])
    balis_csv_file = st.file_uploader("3. 발리스 데이터 CSV 파일을 업로드하세요. (선택 사항)", type=["csv"])
    
    output_filename = st.text_input("4. 저장할 파일 이름", value="output.xml")

    if st.button("XML 생성 시작", type="primary"):
        if main_xml_file:
            try:
                tree = ET.parse(main_xml_file)
                xml_root = tree.getroot()
                
                raillines_data, start_real_kp_value = extract_data_from_xml_root(xml_root)
                
                if raillines_data is not None:
                    # insertion_point = xml_root.find('.//Railway') or xml_root
                    # 수정된 코드
                    insertion_point = xml_root
                    
                    if signal_csv_file:
                        generate_signal_data(signal_csv_file, raillines_data, start_real_kp_value, insertion_point)
                    
                    if balis_csv_file:
                        generate_balis_data(balis_csv_file, raillines_data, start_real_kp_value, insertion_point)

                        
                    # rough_string = ET.tostring(xml_root, 'utf-8')
                    # reparsed = minidom.parseString(rough_string)
                    # pretty_xml_as_string = reparsed.toprettyxml(indent="  ", encoding="utf-8")
                    
                    # st.session_state.generated_xml = pretty_xml_as_string
                    
                    # minidom을 사용한 pretty-printing을 제거하여 불필요한 줄바꿈을 없앱니다.
                    # encoding을 'unicode'로 설정하여 문자열로 직접 변환한 후, 다운로드를 위해 utf-8로 인코딩합니다.
                    xml_string = ET.tostring(xml_root, encoding='unicode')
                    
                    st.session_state.generated_xml = xml_string.encode('utf-8')
                    st.success(f"'{output_filename}' 생성이 완료되었습니다. 아래 버튼으로 다운로드하세요.")

            except ET.ParseError as e:
                st.error(f"오류: 원본 XML 파일 파싱 중 오류 발생: {e}")
            except Exception as e:
                st.error(f"오류: XML 생성 중 예상치 못한 오류 발생: {e}")
        else:
            st.warning("먼저 원본 속성 XML 파일을 업로드해주세요.")

    if st.session_state.generated_xml:
        st.download_button(
            label="생성된 XML 다운로드",
            data=st.session_state.generated_xml,
            file_name=output_filename,
            mime="application/xml"
        )
