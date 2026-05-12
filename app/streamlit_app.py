"""
app/streamlit_app.py - Giao diện phân tích trầm cảm
=====================================================
Chạy: streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import streamlit as st
import string
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.loader import (
    load_model, load_global_importance,
    run_lime, run_shap_local, preprocess_input
)

# ── Paths (chỉnh nếu cần) ──
MODEL_PATH = "models_saved/experiments"
GLOBAL_IMP_JSON = "reports/figures/shap/global_importance.json"

# ══════════════════════════════════════════════════
# Config & Style
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="Depression Analysis",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.main { background: #F7F6F3; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
}

.result-card {
    background: #fff;
    border: 1px solid #E5E3DE;
    border-radius: 10px;
    padding: 20px;
    height: 100%;
}

.result-dep {
    border-left: 4px solid #C0392B;
}
.result-non {
    border-left: 4px solid #27AE60;
}

.big-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    margin: 6px 0 2px;
}
.dep-color { color: #C0392B; }
.non-color { color: #27AE60; }

.conf-text {
    font-size: 13px;
    color: #888;
    margin-bottom: 12px;
}

.conf-bar-bg {
    background: #F0EEE9;
    border-radius: 4px;
    height: 6px;
    width: 100%;
}

.explain-card {
    background: #fff;
    border: 1px solid #E5E3DE;
    border-radius: 10px;
    padding: 16px;
}

.token-highlight {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    margin: 2px;
    font-size: 15px;
    line-height: 2;
}

.global-card {
    background: #fff;
    border: 1px solid #E5E3DE;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 16px;
}

.global-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid #F0EEE9;
}
.global-row:last-child { border-bottom: none; }

.rank-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #aaa;
    width: 30px;
    flex-shrink: 0;
}

.word-pill {
    font-size: 13px;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 12px;
    flex-shrink: 0;
}

.pill-in  { background: #FDEDEC; color: #922B21; }
.pill-out { background: #F0F0F0; color: #666; }

.bar-outer {
    flex: 1;
    background: #F0EEE9;
    border-radius: 3px;
    height: 8px;
    overflow: hidden;
}
.bar-inner { height: 100%; border-radius: 3px; }
.bar-in  { background: #C0392B; }
.bar-out { background: #BDC3C7; }

.score-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #888;
    width: 46px;
    text-align: right;
    flex-shrink: 0;
}

.stTextArea textarea {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 15px !important;
    border: 1px solid #E5E3DE !important;
    border-radius: 8px !important;
    background: #fff !important;
}

div.stButton > button {
    background: #1A1A1A;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: .04em;
    cursor: pointer;
    transition: background .2s;
}
div.stButton > button:hover { background: #333; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# Load resources (cache)
# ══════════════════════════════════════════════════
@st.cache_resource(show_spinner="Đang tải mô hình...")
def get_model():
    return load_model(MODEL_PATH)

@st.cache_resource(show_spinner="Đang tải SHAP global...")
def get_global_importance():
    return load_global_importance(GLOBAL_IMP_JSON)


# ══════════════════════════════════════════════════
# Helper: vẽ highlight text bằng HTML
# ══════════════════════════════════════════════════
def val_to_color(v: float, max_abs: float) -> str:
    """SHAP/LIME value → màu hex."""
    if max_abs == 0:
        return "#F5F5F5", "#333"
    ratio = min(abs(v) / max_abs, 1.0)
    if v > 0:
        r = int(255 - ratio * 60)
        g = int(255 - ratio * 170)
        b = int(255 - ratio * 170)
        text = "#7B1A1A" if ratio > 0.4 else "#333"
    else:
        r = int(255 - ratio * 170)
        g = int(255 - ratio * 60)
        b = int(255 - ratio * 255)
        text = "#0A3D6B" if ratio > 0.4 else "#333"
    return f"rgb({r},{g},{b})", text


def render_highlight(word_val_pairs: list) -> str:
    """Trả về HTML highlight từng từ theo giá trị."""
    if not word_val_pairs:
        return ""
    max_abs = max(abs(v) for _, v in word_val_pairs) or 1e-9
    html = '<div style="line-height:2.2;font-size:15px;">'
    for word, val in word_val_pairs:
        bg, fg = val_to_color(val, max_abs)
        html += (
            f'<span title="{val:+.3f}" style="'
            f'display:inline-block;padding:1px 5px;margin:2px;'
            f'border-radius:4px;background:{bg};color:{fg};'
            f'font-family:IBM Plex Sans,sans-serif">{word}</span> '
        )
    html += '</div>'
    return html


# ══════════════════════════════════════════════════
# Helper: bar chart matplotlib gọn
# ══════════════════════════════════════════════════
def plot_bar(word_val_pairs: list, title: str):
    """Vẽ horizontal bar chart từ list (word, value)."""
    if not word_val_pairs:
        return None
    words, vals = zip(*word_val_pairs[:10])
    colors = ['#C0392B' if v > 0 else '#2980B9' for v in vals]

    fig, ax = plt.subplots(figsize=(4.5, max(2.5, len(words) * 0.38)))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAF8')

    bars = ax.barh(range(len(words)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='#ccc', linewidth=0.8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Score', fontsize=9)
    ax.tick_params(labelsize=9)
    ax.spines[['top','right']].set_visible(False)

    dep_patch = mpatches.Patch(color='#C0392B', label='→ Depression')
    non_patch = mpatches.Patch(color='#2980B9', label='→ Non-depression')
    ax.legend(handles=[dep_patch, non_patch], fontsize=8, loc='lower right')

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════
# Helper: waterfall chart
# ══════════════════════════════════════════════════
def plot_waterfall(tokens, values, baseline: float, final_score: float, title: str):
    """
    Vẽ waterfall plot đúng chuẩn:
    - Hàng đầu: baseline
    - Mỗi token: thanh bắt đầu từ điểm kết thúc của thanh trước (tích lũy)
    - Hàng cuối: kết quả
    """
    # Bỏ qua special tokens, lấy top 8 có |value| lớn nhất
    SPECIAL = {'[cls]', '[sep]', '[pad]', '[mask]', '[unk]'}
    filtered = [(t, v) for t, v in zip(tokens, values) if str(t).lower() not in SPECIAL]
    pairs = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:8]
    if not pairs:
        return None
    words, vals = zip(*pairs)

    # Tính vị trí tích lũy đúng chuẩn waterfall
    running = baseline
    lefts, widths, colors_w = [], [], []
    for v in vals:
        if v >= 0:
            lefts.append(running)       # bắt đầu từ running
            widths.append(v)            # chiều rộng = v
        else:
            lefts.append(running + v)   # bắt đầu từ running+v (thấp hơn)
            widths.append(abs(v))
        colors_w.append('#C0392B' if v > 0 else '#2980B9')
        running += v

    # Thêm hàng baseline (đầu) và kết quả (cuối)
    all_labels  = [f'baseline={baseline:.2f}'] + list(words) + ['Kết quả']
    all_lefts   = [0]        + lefts  + [0]
    all_widths  = [baseline] + list(widths) + [final_score]
    all_colors  = ['#95A5A6'] + colors_w   + ['#2C3E50']

    fig, ax = plt.subplots(figsize=(4.8, max(3.0, len(all_labels) * 0.42)))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAF8')

    ax.barh(range(len(all_labels)), all_widths, left=all_lefts,
            color=all_colors, height=0.55)

    # Đường connector nối các thanh
    running2 = baseline
    for i, v in enumerate(vals):
        end = running2 + v
        # Vẽ đường nối từ cuối thanh hiện tại đến đầu thanh tiếp theo
        ax.plot([end, end], [i + 0.5 + 0.8, i + 1.5 + 0.2],
                color='#ccc', linewidth=0.8, linestyle='--')
        running2 += v

    # Label giá trị trên mỗi thanh
    for i, (left, width, v) in enumerate(zip(all_lefts, all_widths, [baseline] + list(vals) + [final_score])):
        sign = '+' if v > 0 else ''
        ax.text(left + width + 0.005, i, f'{sign}{v:.3f}',
                va='center', fontsize=8, color='#555')

    ax.axvline(0, color='#ddd', linewidth=0.6)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Xác suất Depression', fontsize=9)
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlim(left=min(0, baseline - 0.05))

    dep_patch  = mpatches.Patch(color='#C0392B', label='→ Depression')
    non_patch  = mpatches.Patch(color='#2980B9', label='→ Non-dep')
    base_patch = mpatches.Patch(color='#95A5A6', label='baseline')
    ax.legend(handles=[dep_patch, non_patch, base_patch], fontsize=7, loc='lower right')

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════
# Helper: global panel HTML
# ══════════════════════════════════════════════════
def render_global_panel(global_imp: dict, input_words: list):
    """
    CẬP NHẬT: Sắp xếp từ theo thứ hạng cao -> thấp. 
    Từ không có trong Global Top sẽ nằm ở dưới cùng.
    """
    # 1. Chuẩn bị dữ liệu Global
    sorted_global = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
    word_to_info = {word: (rank, score) for rank, (word, score) in enumerate(sorted_global, 1)}
    max_score = sorted_global[0][1] if sorted_global else 1.0

    # 2. Lấy danh sách từ duy nhất từ input
    unique_input_words = list(dict.fromkeys([w.lower().strip() for w in input_words if w.lower().strip()]))

    # 3. Phân loại thành 2 nhóm để sắp xếp
    ranked_list = []
    unranked_list = []

    for token in unique_input_words:
        if token in word_to_info:
            ranked_list.append(token)
        else:
            unranked_list.append(token)

    # 4. Sắp xếp nhóm có hạng: Thằng nào Rank nhỏ (Score cao) đứng trước
    ranked_list.sort(key=lambda x: word_to_info[x][0]) 

    # 5. Gộp lại: Có hạng trên đầu, không hạng dưới chót
    final_display_list = ranked_list + unranked_list

    rows_html = ""
    for token in final_display_list:
        if token in word_to_info:
            rank, score = word_to_info[token]
            bar_pct = round(score / max_score * 100)
            bar_chars = "█" * int(bar_pct / 5) + "░" * (20 - int(bar_pct / 5))
            
            rows_html += f"""
            <div style="display:flex;align-items:center;gap:15px;padding:8px 0;border-bottom:1px solid #efefef;">
                <span style="font-family:monospace;font-size:12px;color:#aaa;width:40px;">#{rank}</span>
                <span style="font-size:14px;font-weight:600;min-width:100px;">{token}</span>
                <span style="font-family:monospace;color:#C0392B;letter-spacing:1px;flex:1;">{bar_chars}</span>
                <span style="font-family:monospace;font-size:12px;color:#888;width:50px;text-align:right;">{score:.4f}</span>
            </div>"""
        else:
            rows_html += f"""
            <div style="display:flex;align-items:center;gap:15px;padding:8px 0;border-bottom:1px solid #efefef;">
                <span style="font-family:monospace;font-size:12px;color:#ccc;width:40px;">—</span>
                <span style="font-size:14px;color:#bbb;min-width:100px;">{token}</span>
                <span style="font-size:12px;color:#ddd;flex:1;">không có trong global top</span>
            </div>"""

    if not unique_input_words:
        rows_html = '<p style="color:#aaa;text-align:center;padding:10px;">Nhập văn bản để tra cứu thứ hạng toàn cục...</p>'

    return f"""
    <div style="background:#fff;border:1px solid #E5E3DE;border-radius:10px;padding:20px;margin-top:20px;">
        <p class="section-label">TOÀN CỤC — Từ trong câu bạn nhập có trong global top</p>
        {rows_html}
    </div>"""


# ══════════════════════════════════════════════════
# Helper: gộp subword token SHAP → word level
# ══════════════════════════════════════════════════
def merge_subword_shap(tokens_arr: list, values_arr: list, words_in_text: list) -> list:
    """
    Gộp token BPE lại thành từ gốc, cộng SHAP values.
    Tokenizer này dùng khoảng trắng SAU token để đánh dấu ranh giới từ:
      - 'iv' (không có space sau) → prefix, chưa kết thúc từ
      - 'e ' (có space sau)       → cuối từ, kết thúc → gộp thành 'ive'
    Bỏ qua special tokens (empty string, [CLS], [SEP]...).
    """
    import sys
    SPECIAL_TOKENS = {"[cls]", "[sep]", "[pad]", "[mask]", "[unk]"}

    # Bước 1: Gộp token theo trailing space
    merged = []
    current_word = ""
    current_val  = 0.0

    for tok, val in zip(tokens_arr, values_arr):
        tok_raw = str(tok)
        tok_str = tok_raw.strip()

        # Bỏ qua empty hoặc special token
        if not tok_str or tok_str.lower() in SPECIAL_TOKENS:
            continue

        # Xử lý WordPiece (##)
        if tok_str.startswith("##"):
            current_word += tok_str[2:]
            current_val  += val
            # Nếu token gốc có space sau → kết thúc từ
            if tok_raw.endswith(" "):
                merged.append((current_word.lower(), current_val))
                current_word = ""
                current_val  = 0.0
        else:
            # Token thường
            current_word += tok_str
            current_val  += val
            # Nếu token gốc có space sau → kết thúc từ
            if tok_raw.endswith(" ") or tok_raw == tok_raw.rstrip():
                # Có space sau = ranh giới từ
                if tok_raw != tok_raw.rstrip():  # có trailing space thật
                    merged.append((current_word.lower(), current_val))
                    current_word = ""
                    current_val  = 0.0
                # Không có trailing space → tiếp tục gộp

    # Lưu từ cuối nếu còn
    if current_word and current_word.lower() not in SPECIAL_TOKENS:
        merged.append((current_word.lower(), current_val))

    # Bước 2: Build map {word: shap_value}
    shap_word_map = {}
    for word, val in merged:
        shap_word_map[word] = shap_word_map.get(word, 0.0) + val

    print(f"[SHAP merge] merged: {list(shap_word_map.keys())[:15]}", file=sys.stderr)
    print(f"[SHAP merge] input:  {[w.lower() for w in words_in_text]}", file=sys.stderr)

    # Bước 3: Map về câu gốc — strip punct cuối nếu cần
    result = []
    for w in words_in_text:
        w_lower = w.lower()
        if w_lower in shap_word_map:
            result.append((w, shap_word_map[w_lower]))
        else:
            w_stripped = w_lower.rstrip(".,!?;:")
            result.append((w, shap_word_map.get(w_stripped, 0.0)))
    return result


# ══════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════
st.markdown('<h1 style="font-family:IBM Plex Mono,monospace;font-size:32px;font-weight:500;text-align:center;margin-bottom:4px;">Phân tích nguy cơ trầm cảm</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#888;font-size:13px;margin-bottom:20px;text-align:center;">Nhập đoạn văn bản — mô hình DistilBERT sẽ phân tích và giải thích dự đoán bằng LIME & SHAP.</p>', unsafe_allow_html=True)

# ── Input ──
user_text = st.text_area(
    label="Nhập văn bản",
    placeholder="Nhập văn bản vào đây... (tiếng Anh)",
    height=110,
    label_visibility="collapsed"
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    analyze = st.button("Phân tích →")

st.divider()

# ── Load resources ──
predictor, tokenizer, device = get_model()
global_imp = get_global_importance()

# ── PHẦN 1: LOGIC PHÂN TÍCH (Chỉ chạy khi bấm nút) ──
if analyze:
    text = user_text.strip()
    if not text:
        st.warning("Vui lòng nhập văn bản trước khi phân tích.")
        st.stop()

    # Tạo khung trạng thái để phản hồi ngay lập tức[cite: 3]
    with st.status("Đang tiến hành phân tích chuyên sâu...", expanded=True) as status:
        
        st.write("Bước 1: Tiền xử lý văn bản...")
        print("\n[LOG] Đang làm sạch dữ liệu...")
        cleaned_text = preprocess_input(text) #[cite: 2, 5]
        cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))


        st.write("Bước 2: Mô hình DistilBERT đang dự đoán...")
        print("[LOG] Đang chạy mô hình dự đoán...")
        probs = predictor.predict_proba([cleaned_text])[0] #[cite: 1, 5]
        
        # Tính toán các thông số để hiển thị
        dep_prob   = float(probs[1])
        non_prob   = float(probs[0])
        is_dep     = dep_prob >= 0.5
        label      = "Depression" if is_dep else "Non-depression"
        confidence = dep_prob if is_dep else non_prob
        card_cls   = "result-dep" if is_dep else "result-non"
        label_cls  = "dep-color"  if is_dep else "non-color"
        icon       = "🔴" if is_dep else "🟢"

        st.write("Bước 3: Đang tính toán giải thích LIME...")
        print("[LOG] Đang chạy LIME (1000 samples)...")
        # Tính toán LIME một lần duy nhất và lưu vào biến[cite: 2, 5]
        lime_exp = run_lime(predictor, cleaned_text, num_features=10, num_samples=1000) 

        st.write("Bước 4: Đang tính toán giá trị SHAP local...")
        print("[LOG] Đang chạy SHAP local...")
        # Tính toán SHAP một lần duy nhất và lưu vào biến[cite: 2, 5]
        shap_sv = run_shap_local(predictor, tokenizer, cleaned_text, max_evals=300)
        
        status.update(label="Phân tích hoàn tất!", state="complete", expanded=False)

    # ── PHẦN 2: HIỂN THỊ KẾT QUẢ RA GIAO DIỆN ──
    st.markdown("---")
    if cleaned_text != text:
        st.caption(f"📝 Text sau khi tiền xử lý: _{cleaned_text}_")

    col_result, col_explain = st.columns([1, 2.5])

    # --- Cột bên trái: Thẻ kết quả (Chỉ giữ 1 khối duy nhất) ---
    with col_result:
        st.markdown('<div class="section-label">Kết quả</div>', unsafe_allow_html=True)
        conf_pct = round(confidence * 100)
        dep_pct  = round(dep_prob * 100)
        non_pct  = round(non_prob * 100)
        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div style="font-size:26px">{icon}</div>
            <div class="big-label {label_cls}">{label}</div>
            <div class="conf-text">{conf_pct}% tự tin</div>
            <div style="font-size:12px;color:#aaa;margin-bottom:4px">Depression</div>
            <div class="conf-bar-bg">
                <div style="width:{dep_pct}%;height:6px;border-radius:4px;background:#C0392B"></div>
            </div>
            <div style="font-size:12px;color:#C0392B;margin-bottom:8px">{dep_pct}%</div>
            <div style="font-size:12px;color:#aaa;margin-bottom:4px">Non-depression</div>
            <div class="conf-bar-bg">
                <div style="width:{non_pct}%;height:6px;border-radius:4px;background:#27AE60"></div>
            </div>
            <div style="font-size:12px;color:#27AE60">{non_pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Cột bên phải: LIME & SHAP local ---
    with col_explain:
        st.markdown('<div class="section-label">Giải thích cụ thể</div>', unsafe_allow_html=True)
        col_lime, col_shap = st.columns(2)

        # ── Hiển thị LIME ──
        with col_lime:
            st.markdown('<div class="explain-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">LIME — local</div>', unsafe_allow_html=True)
            
            # Sử dụng kết quả lime_exp đã tính ở trên, không tính lại
            lime_feats = lime_exp.as_list(label=1)
            lime_map = {w.lower(): v for w, v in lime_feats}
            words_in_text = cleaned_text.split()

            # Match: thử trực tiếp, nếu không khớp thì strip punct cuối
            import sys
            word_val_lime = []
            for w in words_in_text:
                w_lower = w.lower()
                if w_lower in lime_map:
                    word_val_lime.append((w, lime_map[w_lower]))
                else:
                    w_stripped = w_lower.rstrip(".,!?;:")
                    word_val_lime.append((w, lime_map.get(w_stripped, 0.0)))

            print(f"[LIME] lime_map keys: {list(lime_map.keys())}", file=sys.stderr)
            print(f"[LIME] word_val_lime: {word_val_lime}", file=sys.stderr)

            st.markdown(render_highlight(word_val_lime), unsafe_allow_html=True)

            fig_lime = plot_bar(lime_feats, "LIME — top từ")
            if fig_lime:
                st.pyplot(fig_lime, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Hiển thị SHAP local ──
        with col_shap:
            st.markdown('<div class="explain-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">SHAP local — waterfall</div>', unsafe_allow_html=True)
            
            # Sử dụng kết quả shap_sv đã tính ở trên, không tính lại[cite: 5]
            sv_dep = shap_sv[0, :, 'Depression']
            tokens_arr = [str(t) for t in sv_dep.data]
            values_arr = [float(v) for v in sv_dep.values]
            baseline   = float(shap_sv.base_values[0, 1])

            # Gộp subword token → word level rồi mới highlight
            word_val_shap = merge_subword_shap(tokens_arr, values_arr, cleaned_text.split())

            # Debug log
            import sys
            print(f"[SHAP] tokens_arr: {tokens_arr}", file=sys.stderr)
            print(f"[SHAP] values_arr: {[round(v,4) for v in values_arr]}", file=sys.stderr)
            print(f"[SHAP] word_val_shap: {word_val_shap}", file=sys.stderr)

            st.markdown(render_highlight(word_val_shap), unsafe_allow_html=True)

            fig_wf = plot_waterfall(
                tokens_arr, values_arr,
                baseline=baseline,
                final_score=dep_prob,
                title="SHAP waterfall"
            )
            if fig_wf:
                st.pyplot(fig_wf, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

# ── PHẦN 3: PHÂN TÍCH TOÀN CỤC (Luôn hiện dưới cùng) ──
# Dùng cleaned_words để tra cứu cho chính xác với dữ liệu SHAP[cite: 6]

cleaned_input_words = (
    [
        re.sub(r'[^\w\s]', '', word)
        for word in preprocess_input(user_text).split()
    ]
    if user_text.strip()
    else []
)

html_content = render_global_panel(global_imp, cleaned_input_words)
st.html(f"""<div style="font-family:sans-serif;">{html_content}</div>""")