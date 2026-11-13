"""
Module: teencode_preprocessing.py
Dự án: Phân loại ngôn từ thù ghét ViHSD bằng PhoBERT
Tác giả: Trần Thị Mỹ Dung (Phụ trách Tiền xử lý & Chuẩn hóa Teencode)
Bàn giao cho: Bạn Lam (Leader - Mô hình hợp nhất Final Optimized) & Nhóm ViHSD

Chức năng:
1. Đọc và chuẩn hóa từ điển Teencode từ file CSV (hỗ trợ cả teencode_toi.csv và teencode_co.csv).
2. Chuẩn hóa văn bản: Unicode NFC, xử lý ký tự ẩn danh (USER, LOCATION, URL), HTML entities.
3. Thay thế Teencode/từ lóng tiếng Việt an toàn:
   - Ưu tiên cụm từ dài trước (Longest Match First)
   - Tránh thay thế nhầm từ con (Subword collision boundary)
   - Giữ nguyên Emoji, từ phủ định và từ ngữ mang sắc thái thù ghét/xúc phạm.
4. Tách từ ghép tiếng Việt bằng PyVi (ViTokenizer.tokenize) sau khi đã bung Teencode.
"""

import re
import html
import unicodedata
import csv
from typing import Dict, Optional, Tuple, Callable

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

# Bộ ký tự từ tiếng Việt dùng cho Regex Word Boundary an toàn
VIETNAMESE_WORD_CHARS = (
    r'a-zA-Z0-9_'
    r'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'
    r'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ'
)

def clean_teencode_replacement(val: str, clean_notes: bool = True) -> str:
    """
    Làm sạch giá trị thay thế trong từ điển:
    - Bỏ khoảng trắng thừa.
    - Nếu clean_notes=True: loại bỏ chú thích dạng '(nhấn mạnh)', '(lóng)',
      và nếu có nhiều nghĩa ngăn cách bởi dấu '/' thì lấy nghĩa chuẩn đầu tiên.
    """
    val = str(val).strip()
    if clean_notes:
        # Bỏ phần trong ngoặc đơn
        val = re.sub(r'\(.*?\)', '', val).strip()
        # Nếu có dấu '/', lấy phương án đầu tiên
        if '/' in val:
            parts = [p.strip() for p in val.split('/') if p.strip()]
            val = parts[0] if parts else val
    return re.sub(r'\s+', ' ', val).strip()

def load_teencode_dict(filepath: str, clean_notes: bool = True) -> Dict[str, str]:
    """
    Đọc file CSV từ điển Teencode và trả về Python Dictionary {teencode: replacement}.
    Sử dụng thư viện chuẩn csv của Python (không phụ thuộc pandas).
    Tự động nhận diện cấu trúc file:
    - Định dạng của tôi: ['teencode', 'replacement']
    - Định dạng của cô: ['STT', 'Teencode', 'Nghĩa', 'Dạng chuẩn hóa', 'Loại teencode']
    """
    mapping = {}
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return mapping
            
        cols = [c.strip() for c in header]
        
        # 1. Định dạng teencode_toi: teencode, replacement
        if 'teencode' in cols and 'replacement' in cols:
            idx_k = cols.index('teencode')
            idx_v = cols.index('replacement')
            for row in reader:
                if len(row) > max(idx_k, idx_v):
                    k = str(row[idx_k]).strip().lower()
                    v = str(row[idx_v]).strip()
                    if k and v:
                        mapping[k] = v
                        
        # 2. Định dạng teencode_co: CHỈ XÉT 2 CỘT 'Teencode' VÀ 'Nghĩa' (theo yêu cầu)
        elif 'Teencode' in cols and 'Nghĩa' in cols:
            idx_k = cols.index('Teencode')
            idx_v = cols.index('Nghĩa')
            for row in reader:
                if len(row) > max(idx_k, idx_v):
                    k = str(row[idx_k]).strip().lower()
                    raw_v = row[idx_v].strip()
                    v = clean_teencode_replacement(raw_v, clean_notes=clean_notes)
                    if k and v:
                        mapping[k] = v
        else:
            # Giả định cột 0 là key, cột 1 là value
            for row in reader:
                if len(row) >= 2:
                    k = str(row[0]).strip().lower()
                    v = clean_teencode_replacement(row[1], clean_notes=clean_notes)
                    if k and v:
                        mapping[k] = v
                    
    return mapping

def build_teencode_replacer(teencode_dict: Dict[str, str]) -> Callable[[str], str]:
    """
    Biên dịch regex pattern tối ưu cho việc thay thế Teencode.
    - Sắp xếp từ dài trước (Longest Match First) để cụm từ nhiều token (e.g. 'ảo vl')
      được match trước từ đơn (e.g. 'vl').
    - Sử dụng Word Boundary an toàn cho tiếng Việt.
    """
    if not teencode_dict:
        return lambda text: text
    
    # Sắp xếp theo độ dài giảm dần
    sorted_keys = sorted(teencode_dict.keys(), key=lambda x: len(x), reverse=True)
    
    # Tạo Pattern regex
    escaped_keys = [re.escape(k) for k in sorted_keys]
    pattern_str = (
        rf'(?<![{VIETNAMESE_WORD_CHARS}])'
        rf'(' + '|'.join(escaped_keys) + rf')'
        rf'(?![{VIETNAMESE_WORD_CHARS}])'
    )
    pattern = re.compile(pattern_str, flags=re.IGNORECASE)
    lower_dict = {k.lower(): v for k, v in teencode_dict.items()}
    
    def replace_match(match: re.Match) -> str:
        matched_text = match.group(0)
        return lower_dict.get(matched_text.lower(), matched_text)
        
    def replacer(text: str) -> str:
        return pattern.sub(replace_match, text)
        
    return replacer

def preprocess_teencode(
    text: str, 
    teencode_dict: Optional[Dict[str, str]] = None,
    replacer_func: Optional[Callable[[str], str]] = None
) -> str:
    """
    Hàm tiền xử lý và chuẩn hóa Teencode hoàn chỉnh cho mô hình PhoBERT.
    
    Quy trình 7 bước:
    1. Chuẩn hóa Unicode NFC.
    2. Giải mã HTML entities (html.unescape).
    3. Chuẩn hóa token ẩn danh:
       - <person>, <person name>, @username -> USER
       - <loc> -> LOCATION
       - URL -> URL
       - Thẻ HTML -> khoảng trắng
    4. Loại bỏ các ký tự điều khiển (Control Characters - category 'Cc').
    5. Bung Teencode / từ lóng về từ tiếng Việt chuẩn.
    6. Chuẩn hóa khoảng trắng.
    7. Tách từ ghép tiếng Việt bằng PyVi (ViTokenizer.tokenize).
    
    Đặc điểm quan trọng:
    - Giữ nguyên Emoji và các biểu tượng cảm xúc.
    - Giữ nguyên từ phủ định và từ ngữ nhạy cảm/thù ghét phục vụ phân loại Hate Speech.
    """
    text = str(text)
    if not text.strip():
        return ''
        
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. HTML unescape
    text = html.unescape(text)
    
    # 3. Chuẩn hóa token ẩn danh & URL (đồng bộ với chính sách 'shared' của notebook gốc)
    text = re.sub(r'<\s*person(?:\s+name)?\s*>', ' USER ', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*loc\s*>', ' LOCATION ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'(?<!\w)@[\w.]+', ' USER ', text)
    text = re.sub(r'</?[A-Za-z][^>]*>', ' ', text)
    
    # 4. Loại bỏ control characters
    text = ''.join(' ' if unicodedata.category(c) == 'Cc' else c for c in text)
    
    # 5. Bung Teencode / Từ lóng
    if replacer_func is not None:
        text = replacer_func(text)
    elif teencode_dict is not None:
        compiled = build_teencode_replacer(teencode_dict)
        text = compiled(text)
        
    # 6. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Tách từ ghép tiếng Việt bằng PyVi (nếu có thư viện)
    if not text:
        return ''
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(text)
    return text

if __name__ == '__main__':
    print("=== KIỂM TRA MODULE TEENCODE PREPROCESSING ===")
    
    # Dữ liệu từ điển mẫu
    sample_dict = {
        'ko': 'không',
        'k': 'không',
        'dc': 'được',
        'ảo vl': 'không thể tin',
        'vl': 'vãi',
        'cmnr': 'con mẹ nó rồi',
        'lol': 'lồn',
        'dell': 'đéo',
        'mng': 'mọi người'
    }
    
    replacer = build_teencode_replacer(sample_dict)
    
    test_cases = [
        "thằng này ảo vl luôn mng ạ 😡",
        "tao ko thích mày đâu, dell hiểu kiểu gì cmnr",
        "học kỳ này k dc điểm cao https://fb.com/test",
        "<person> ơi con bé đó nhìn khôn ngoan ghê",
        "@nam_nguyen đúng là đồ con lol :))"
    ]
    
    print("\nKết quả thử nghiệm tiền xử lý:")
    for raw in test_cases:
        processed = preprocess_teencode(raw, replacer_func=replacer)
        print(f"[Gốc] : {raw}")
        print(f"[Xử lý]: {processed}")
        print("-" * 50)
