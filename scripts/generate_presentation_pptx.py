#!/usr/bin/env python3
"""
Generates a 12-Slide Widescreen (16:9) PowerPoint Presentation (.pptx)
for "Transport-Aware QoE Modeling for WebRTC: Predicting Playback Freezes with Machine Learning"
Includes:
- Modern academic slide theme with header banners, KPI cards, formatted tables, and 300-DPI charts
- Full Speaker Notes (口頭報告逐字講稿) in every slide
- Outputs to /home/user/桌面/WebRTC_Transport_Aware_QoE_報告.pptx and /home/user/桌面/webrtc-qoe-github-pages/assets/
"""

import os
import shutil
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

CHART_DIR = "/home/user/桌面/webrtc-qoe-github-pages/assets/charts"
OUT_DESKTOP = "/home/user/桌面/WebRTC_Transport_Aware_QoE_報告.pptx"
OUT_GH_PAGES = "/home/user/桌面/webrtc-qoe-github-pages/assets/WebRTC_Transport_Aware_QoE_Presentation.pptx"

FONT_FAMILY = "Noto Sans CJK TC"

# Theme Colors
C_NAVY = RGBColor(15, 23, 42)       # #0f172a
C_BLUE = RGBColor(37, 99, 235)      # #2563eb
C_LIGHT_BG = RGBColor(248, 250, 252) # #f8fafc
C_CARD_BG = RGBColor(255, 255, 255)
C_BORDER = RGBColor(203, 213, 225)
C_TEXT_DARK = RGBColor(15, 23, 42)
C_TEXT_MUTED = RGBColor(71, 85, 105)
C_WHITE = RGBColor(255, 255, 255)
C_GREEN = RGBColor(5, 150, 105)
C_RED = RGBColor(220, 38, 38)
C_AMBER = RGBColor(217, 119, 6)
C_PURPLE = RGBColor(124, 58, 237)


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title_text: str, subtitle_text: str = "", badge_text: str = ""):
    set_slide_bg(slide, C_LIGHT_BG)
    # Top navy bar
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    banner.fill.solid()
    banner.fill.fore_color.rgb = C_NAVY
    banner.line.fill.background()

    # Accent stripe
    stripe = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.02), Inches(13.333), Inches(0.05))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = C_BLUE
    stripe.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(10.5), Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_FAMILY
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = FONT_FAMILY
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = RGBColor(147, 197, 253)

    if badge_text:
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.9), Inches(0.26), Inches(2.05), Inches(0.48))
        badge.fill.solid()
        badge.fill.fore_color.rgb = C_BLUE
        badge.line.fill.background()
        btf = badge.text_frame
        bp = btf.paragraphs[0]
        bp.text = badge_text
        bp.alignment = PP_ALIGN.CENTER
        bp.font.name = FONT_FAMILY
        bp.font.size = Pt(11)
        bp.font.bold = True
        bp.font.color.rgb = C_WHITE


def add_footer(slide, page_num: int, total_pages: int = 12):
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(7.12), Inches(12.4), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = f"Transport-Aware QoE Modeling for WebRTC | https://liu20001107.github.io/webrtc-qoe-ml/   ({page_num} / {total_pages})"
    p.alignment = PP_ALIGN.RIGHT
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = C_TEXT_MUTED


def add_notes(slide, notes_text: str):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_text


def add_picture_fitted(slide, img_path: str, left_in: float, top_in: float, max_w_in: float, max_h_in: float):
    if not os.path.exists(img_path):
        return
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    aspect = w_px / float(h_px)
    box_aspect = max_w_in / float(max_h_in)
    if aspect >= box_aspect:
        w = max_w_in
        h = max_w_in / aspect
    else:
        h = max_h_in
        w = max_h_in * aspect
    x = left_in + (max_w_in - w) / 2.0
    y = top_in + (max_h_in - h) / 2.0
    slide.shapes.add_picture(img_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_card(slide, left: float, top: float, width: float, height: float, title: str, bullets: list, accent_color=C_BLUE):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = C_CARD_BG
    card.line.color.rgb = C_BORDER
    card.line.width = Pt(1.2)

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.1), Inches(height))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent_color
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.12), Inches(width - 0.3), Inches(height - 0.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.name = FONT_FAMILY
    p0.font.size = Pt(14.5)
    p0.font.bold = True
    p0.font.color.rgb = accent_color
    p0.space_after = Pt(6)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_FAMILY
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT_DARK
        p.space_after = Pt(4)


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # Slide 1: Cover Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, C_NAVY)

    # Top decorative bar
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.15))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = C_BLUE
    top_bar.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(0.8), Inches(1.1), Inches(11.7), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "Transport-Aware QoE Modeling for WebRTC:\nPredicting Playback Freezes with Machine Learning"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = C_WHITE
    p.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "基於傳輸層感知（UDP / QUIC / TCP）與機器學習之 WebRTC 畫面凍結預測與主動式防凍控制系統"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(96, 165, 250)
    p2.space_after = Pt(18)

    p3 = tf1.add_paragraph()
    p3.text = "實測環境：mediasoup WebRTC SFU + Linux Kernel tc-netem 真實損傷注入 + Scikit-Learn (RF-Pro / Deep MLP-Pro)\nGitHub 專案網站：https://liu20001107.github.io/webrtc-qoe-ml/"
    p3.font.name = FONT_FAMILY
    p3.font.size = Pt(12.5)
    p3.font.color.rgb = RGBColor(203, 213, 225)

    # 4 KPI Highlight Boxes on Cover
    kpis = [
        ("93.04% F1", "RF-Pro 凍結預測準確度", "1,020 筆跨協定與跨影片樣本\nROC-AUC 高達 95.70%", C_BLUE),
        ("71.9% → 15.8%", "主動防凍降低畫面凍結率", "Foreman 實測減少近 4/5 卡頓\n單窗凍結時長縮短 88%", C_GREEN),
        ("89.24% F1", "提前 1.5 秒預警下一窗凍結", "結合 22 維因果時序趨勢特徵\n於卡頓發生前主動降載保幀率", C_PURPLE),
        ("7.1% → 92.9%", "跨傳輸層 (TCP) 預測躍升", "解決可靠傳輸隱藏掉包之盲點\n驗證 Transport-Aware 必要性", C_AMBER),
    ]
    for idx, (val, lbl, desc, col) in enumerate(kpis):
        left = 0.8 + idx * 2.98
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(4.65), Inches(2.8), Inches(2.1))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(30, 41, 59)
        box.line.color.rgb = col
        box.line.width = Pt(2)

        btf = box.text_frame
        btf.word_wrap = True
        bp1 = btf.paragraphs[0]
        bp1.text = val
        bp1.font.name = FONT_FAMILY
        bp1.font.size = Pt(23)
        bp1.font.bold = True
        bp1.font.color.rgb = C_WHITE
        bp1.space_after = Pt(4)

        bp2 = btf.add_paragraph()
        bp2.text = lbl
        bp2.font.name = FONT_FAMILY
        bp2.font.size = Pt(12)
        bp2.font.bold = True
        bp2.font.color.rgb = RGBColor(147, 197, 253)
        bp2.space_after = Pt(6)

        bp3 = btf.add_paragraph()
        bp3.text = desc
        bp3.font.name = FONT_FAMILY
        bp3.font.size = Pt(10)
        bp3.font.color.rgb = RGBColor(203, 213, 225)

    add_notes(s1, "【講者講稿 - 第 1 頁：封面】\n大家好，今天要報告的主題是《Transport-Aware QoE Modeling for WebRTC: Predicting Playback Freezes with Machine Learning》。\n我們在自己的 mediasoup WebRTC SFU 伺服器上，結合 Linux 核心 tc-netem 真實網路損傷注入與瀏覽器 W3C getStats() 遙測，收集了超過 1,000 筆跨傳輸協定與跨影片的實測樣本，訓練出 F1-Score 達 93.04% 的 RF-Pro 凍結預測模型，並將模型接回發送端實現主動式防凍控制，成功將畫面凍結率從 71.9% 壓低至 15.8%。")

    # =========================================================================
    # Slide 2: Motivation & Problem Statement
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. 研究背景與問題定義 (Motivation & Problem Statement)", "為什麼傳統只看「掉包率 (Packet Loss)」的 WebRTC QoE 模型會失效？", "研究動機")
    add_footer(s2, 2)

    add_card(
        s2, 0.45, 1.35, 6.05, 2.65,
        "📌 傳統 WebRTC (SRTP over UDP) 的行為",
        [
            "標準 WebRTC 預設運行於不可靠的 UDP 傳輸層之上。",
            "當網路發生擁塞掉包時，RTP 封包序號（Sequence Number）會直接跳號。",
            "接收端瀏覽器立即偵測到 packetsLost 增加，並發送 RTCP NACK 請求重傳（RTX）或 PLI 關鍵幀請求。",
            "因此在純 UDP 環境下，packetsLost 與 nackCount 是預測畫面卡頓的直接指標。",
        ],
        C_BLUE,
    )

    add_card(
        s2, 6.8, 1.35, 6.05, 2.65,
        "⚠️ 現代多工與可靠傳輸 (QUIC / ICE-TCP) 的盲點",
        [
            "在防火牆受限或新興架構下，WebRTC 常切換至 ICE-TCP 或 QUIC 傳輸通道。",
            "可靠傳輸層會在作業系統底層自動重傳遺失封包，確保依序送達上層。",
            "結果：WebRTC 應用層看到的 packetsLost 與 RTX 重傳幾乎為 0（掩蓋了網路掉包）！",
            "但等待重傳引發嚴重的「隊頭阻塞 (Head-of-Line Blocking)」，導致 RTT 暴增與 Jitter Buffer 劇烈膨脹，畫面依然嚴重凍結！",
        ],
        C_RED,
    )

    add_card(
        s2, 0.45, 4.25, 12.4, 2.65,
        "🎯 本研究三大核心目標 (Research Objectives)",
        [
            "目標一【跨協定實測資料集建構】：於真實 mediasoup SFU 伺服器上，透過 Linux 核心 tc-netem 針對 UDP、QUIC、TCP 注入真實掉包、抖動與頻寬限制，採集 36 維度客戶端 getStats() 遙測數據。",
            "目標二【22 維因果時序機器學習模型 (RF-Pro & Deep MLP-Pro)】：在不偷看 FPS 與幀間延遲的前提下，結合 EMA-3 滑動趨勢、斜率與相對碼率驟降比，同時達成高精準度「當窗凍結判定 (t)」與「提前 1.5 秒預警 (t+1)」。",
            "目標三【閉迴路主動防凍調控 (Proactive Freeze Prevention)】：將預訓練模型接回 WebRTC 發送端，於預測風險升高時主動調降解析度與碼率、排空佇列（Bufferbloat），驗證「有模型 vs. 原本沒模型」在多部未知新影片上的實質畫質改善。",
        ],
        C_GREEN,
    )

    add_notes(s2, "【講者講稿 - 第 2 頁：研究背景與問題定義】\n這篇研究要解決的核心問題是：傳統 WebRTC 監控習慣看『掉包率 (packetsLost)』跟『NACK 重傳次數』來判斷畫面會不會卡頓。這在純 UDP 下可行，但當 WebRTC 走 ICE-TCP 或 QUIC 等具重傳或多工機制的傳輸層時，底層把掉包修補並隱藏起來了，應用層看到的 packetsLost 近乎是 0，但因為隊頭阻塞（Head-of-Line Blocking），畫面照樣嚴重定格凍結！因此我們必須建立具備『傳輸層感知（Transport-Aware）』且結合緩衝區延遲動態的機器學習模型。")

    # =========================================================================
    # Slide 3: System Architecture & Linux tc-netem Testbed
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. 系統架構與 Linux tc-netem 實驗平台", "mediasoup WebRTC SFU + 位元遮罩精準視訊埠損傷注入 + 閉迴路遙測", "系統架構")
    add_footer(s3, 3)

    add_card(
        s3, 0.45, 1.35, 5.4, 5.55,
        "🛠️ 實驗平台關鍵技術設計",
        [
            "SFU 媒體伺服器：Node.js + mediasoup，支援動態切換 UDP / QUIC / ICE-TCP 傳輸通道。",
            "虛擬客戶端群：Selenium + Chrome，透過 --use-file-for-fake-video-capture 注入 ITU/Xiph 標準 Y4M 測試影片（Akiyo, Foreman, Bus, Football）。",
            "Linux 核心 tc-netem 精準攔截：使用 prio bands 4 搭配 u32 位元遮罩 (0x0800/0xff00)，僅攔截 mediasoup 專用視訊埠 (Port 2048–2303)。",
            "為何不用伺服器 pause() 模擬？單純暫停 consumer 不會造成 RTP 序號跳號（掉包與 NACK 全為 0）；使用真實 tc-netem 才能觸發真實 NACK、RTX 重傳與解碼掉幀！",
            "即時網頁監測儀表板：每 600ms 於瀏覽器右上角即時渲染 W3C freezeCount、JitterBufferDelay 與模型推論結果。",
        ],
        C_BLUE,
    )

    add_picture_fitted(s3, os.path.join(CHART_DIR, "sfu_architecture_diagram.png"), 6.05, 1.35, 6.8, 5.55)

    add_notes(s3, "【講者講稿 - 第 3 頁：系統架構】\n這是我們的實驗架構。有一個非常關鍵的工程細節：如果只用伺服器端的 consumer.pause() 模擬斷流，RTP 封包序號不會跳號，瀏覽器看到的掉包跟 NACK 會全部是 0，數據會失真。因此我們撰寫了 tc_netem_ctl.sh，利用 Linux 核心的 u32 位元遮罩過濾器，精準只針對 mediasoup 的視訊連接埠 2048 到 2303 注入真實掉包、常態分佈抖動與頻寬限制，讓瀏覽器觸發 100% 真實的 RTCP NACK、RTX 重傳與解碼器掉幀。")

    # =========================================================================
    # Slide 4: Freeze Measurement & 22 Causal Temporal Features
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. 畫面凍結測量標準與 22 維因果時序特徵工程", "W3C 原生凍結判定公式 + 嚴格防範目標洩漏 (Zero Target Leakage)", "方法論")
    add_footer(s4, 4)

    add_card(
        s4, 0.45, 1.35, 6.05, 5.55,
        "🔍 畫面凍結 (Playback Freeze) 是怎麼測量的？",
        [
            "第 1 層【W3C WebRTC 官方 C++ 原生計數器】：每 1.5 秒讀取 inbound-rtp 之 freezeCount 與 totalFreezesDuration。",
            "Chromium 原生判定公式：當連續兩幀渲染間隔 Δt ≥ max(3 × 平均幀距, 平均幀距 + 150ms) 時，自動將 freezeCount + 1 並累加凍結秒數。",
            "第 2 層【前端 HTML5 requestVideoFrameCallback】：於 <video> 元素監聽每一幀渲染時間戳，捕捉 >330ms 之肉眼可見停頓（排除靜態畫面省電）。",
            "第 3 層【重度定格邊界捕捉】：由於 W3C freezeCount 在「下一張新畫面解出時」才結算，若單窗 1.5 秒內完全卡死，結合幀距 >90ms 與低幀率即時標記。",
            "嚴格防作弊原則：訓練機器學習模型時，完全剔除 fps 與 inter_frame_delay，僅允許使用純網路傳輸與緩衝區指標作為輸入特徵 (X)！",
        ],
        C_RED,
    )

    add_card(
        s4, 6.8, 1.35, 6.05, 5.55,
        "🧠 22 維因果時序特徵工程 (Causal Temporal Features)",
        [
            "基礎 10 維快照指標：rtt_ms, rx_bitrate_kbps, packets_lost_delta, packet_loss_rate_pct, retransmitted_packets_delta, jitter_ms, jitter_buffer_delay_ms, nack_delta, pli_delta, frames_dropped_delta。",
            "特徵升級 1【相對碼率驟降比 (bitrate_drop_ratio) — 重要度第 1 名】：當下碼率除以該串流歷史健康峰值碼率，消除不同影片（靜態主播 vs. 美式足球）天生基礎碼率差異！",
            "特徵升級 2【指數移動平均趨勢 (EMA-3)】：計算過去 3 窗 (4.5秒) 之 rtt_ema3, jb_delay_ema3, jitter_ema3, nack_ema3, loss_rate_ema3，過濾單窗突發雜訊。",
            "特徵升級 3【瞬時惡化斜率 (ΔRTT, ΔJB Delay)】：區分佇列正在塞爆膨脹（正斜率）或排空恢復（負斜率）。",
            "特徵升級 4【未救回淨掉包數 (Lost − RTX)】：區分「已被 RTX 重傳救回之掉包」與「真正造成解碼缺口之掉包」。",
        ],
        C_PURPLE,
    )

    add_notes(s4, "【講者講稿 - 第 4 頁：凍結測量與特徵工程】\n左邊說明我們如何測量凍結：主要依據 W3C WebRTC 官方的 freezeCount 與 totalFreezesDuration，只要兩幀間隔超過 3 倍平均幀距或平均加 150ms，Chrome 底層就會自動計入凍結。右邊則是我們讓模型準確率突破 93% 的關鍵：我們刻意把 FPS 從輸入特徵剔除防止作弊，並沿時間軸算出 12 個過去視窗的因果趨勢特徵，特別是『相對碼率驟降比 (bitrate_drop_ratio)』，成功消除了不同影片基礎碼率不同的問題。")

    # =========================================================================
    # Slide 5: 5-Minute Continuous Cycling Time-Series Analysis
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. 實驗一：5 分鐘 (300 秒 / 5 輪循環) 連續網路損傷時序分析", "每 60 秒完成一輪四階段損傷循環 (0% → 3.5% → 7.5% → 14% Loss)，共 570 筆時間視窗樣本", "時序實測")
    add_footer(s5, 5)

    add_picture_fitted(s5, os.path.join(CHART_DIR, "qoe_5min_timeseries_curves.png"), 0.4, 1.25, 8.6, 5.7)

    add_card(
        s5, 9.15, 1.3, 3.75, 5.6,
        "📊 5 分鐘連續循環關鍵發現",
        [
            "階梯式劣化清晰可見：隨掉包率由 0% → 3.5% → 7.5% → 14% 遞增，單窗 NACK 由 0 飆升至 >400 次，累計 NACK 達 15,532 次。",
            "重度損傷期 (紅區)：平均碼率跌至 18.8 Kbps，FPS 降至 1.13，凍結發生率達 99.32%。",
            "佇列排空與 GCC 恢復延遲 (Queue Draining Lag)：第 2~5 輪切回綠色正常網路初期，前一輪積壓之過期封包瞬間排空結算，且 GCC 採加法遞增 (AIMD) 需數秒爬升碼率。",
        ],
        C_BLUE,
    )

    add_notes(s5, "【講者講稿 - 第 5 頁：5 分鐘時序分析】\n這張圖是我們透過 ngrok 連續跑滿 5 分鐘（300 秒、共 5 個完整循環、570 筆樣本）的時間序列圖。可以看到每當進入橘色（7.5% 掉包、850kbps）與紅色（14% 掉包、400kbps）區域，第二排的掉包數與 NACK 重傳請求立刻飆高，第三排的 Jitter Buffer 延遲突破 1000ms，第四排的 W3C 凍結時長與凍結率也同步拉滿。")

    # =========================================================================
    # Slide 6: Transport-Aware Feature Shift (UDP vs QUIC vs TCP)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. 實驗二：跨傳輸層協定 (UDP vs. QUIC vs. TCP) 特徵重要度位移", "實證為何 WebRTC QoE 模型必須具備「傳輸層感知 (Transport-Aware)」", "跨協定對比")
    add_footer(s6, 6)

    add_picture_fitted(s6, os.path.join(CHART_DIR, "qoe_transport_feature_importance.png"), 0.4, 1.25, 8.7, 5.7)

    add_card(
        s6, 9.25, 1.3, 3.65, 5.6,
        "🔍 跨傳輸層物理位移結論",
        [
            "UDP 模式 (藍條)：NACK 重傳請求 (19.9%) 與 RTT (19.1%) 為首要凍結預測指標。",
            "QUIC 多工模式 (綠條)：RTX 重傳封包數 (27.9%)、RTT (16.2%) 與 Jitter (14.7%) 主導預測。",
            "TCP 可靠模式 (紅條)：OS 底層重傳掩蓋了掉包，使 Packets Lost (0.07%) 與 RTX (0.0%) 完全失效！",
            "特徵主導權轉移：TCP 下緩衝區延遲 JitterBufferDelay (32.93%) 與 Jitter (22.56%) 躍升為前兩名！",
            "跨協定泛化實證 (右圖)：傳統僅看掉包模型在 TCP 下 F1 崩跌至 7.1%，而論文 Transport-Aware 模型高達 92.9%！",
        ],
        C_GREEN,
    )

    add_notes(s6, "【講者講稿 - 第 6 頁：跨傳輸層特徵重要度位移】\n這張圖完美重現並驗證了論文的核心結論。左圖可以看到：在藍色 UDP 下，NACK 跟掉包非常重要；但切到紅色 TCP 後，因為作業系統在底層把掉包重傳補齊了，Packets Lost 與 RTX 重傳的重要度直接掉到 0%！取而代之的是 Jitter Buffer Delay 暴增到 32.93% 成為第一名。右圖更證明：如果拿傳統只看掉包跟 NACK 的模型去預測 TCP，F1-Score 只有 7.1% 完全失效；而加入緩衝與傳輸感知的 Transport-Aware 模型則高達 92.9%！")

    # =========================================================================
    # Slide 7: Enhanced RF-Pro & Deep MLP-Pro Model Evaluation
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. 實驗三：強化版 RF-Pro 與 Deep MLP-Pro 模型訓練與評估", "800 棵樹軟投票集成森林 vs. 三層抗離群值深度神經網路 (1,020 筆 5-Fold 交叉驗證)", "模型評估")
    add_footer(s7, 7)

    add_picture_fitted(s7, os.path.join(CHART_DIR, "qoe_ml_model_comparison_roc_cm.png"), 0.4, 1.25, 8.6, 5.7)

    add_card(
        s7, 9.15, 1.3, 3.75, 5.6,
        "🏆 模型升級與驗證結果",
        [
            "RF-Pro 架構：400 棵 RandomForest + 400 棵 ExtraTrees 進行機率加權軟投票 (Soft Voting)。",
            "Deep MLP-Pro 架構：RobustScaler (抑制極端 RTT 離群值) + StandardScaler + (128→64→32) ReLU 網路。",
            "當窗凍結判定 (t)：RF-Pro 達 93.04% F1、94.92% Precision、95.70% ROC-AUC；MLP-Pro 達 92.16% F1。",
            "下一窗提前 1.5s 預警 (t→t+1)：RF-Pro 達 89.24% F1 / 91.99% AUC，證明可在卡頓發生前成功預警！",
            "Top-3 關鍵特徵 (右下圖)：相對碼率驟降比、碼率 EMA-3 趨勢、接收碼率與 NACK 密集度。",
        ],
        C_PURPLE,
    )

    add_notes(s7, "【講者講稿 - 第 7 頁：強化版模型訓練與評估】\n在全部 1,020 筆真實樣本上進行 5-Fold 分層交叉驗證，我們的 RF-Pro 集成模型在當窗凍結判定達到 93.04% F1-Score 與 95.70% ROC-AUC；更重要的是，在『提前 1.5 秒預警下一個視窗 (t+1) 會不會凍結』的任務上，也達到了 89.24% F1-Score 與 91.99% AUC。右下角可以看到，我們設計的『相對碼率驟降比』與 EMA 時序趨勢特徵名列前茅。")

    # =========================================================================
    # Slide 8: Cross-Video Generalization (Foreman, Bus, Football)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. 實驗四：3 部未知新影片跨影片泛化測試 (Cross-Video Generalization)", "完全不重新訓練模型，直接測試 Foreman (中動態)、Bus (高紋理橫移)、Football (劇烈運動)", "跨影片泛化")
    add_footer(s8, 8)

    add_picture_fitted(s8, os.path.join(CHART_DIR, "qoe_cross_video_generalization.png"), 0.4, 1.25, 8.6, 5.7)

    add_card(
        s8, 9.15, 1.3, 3.75, 5.6,
        "🎬 零重訓跨影片推論表現",
        [
            "測試設計：將僅在靜態主播 (Akiyo) 上訓練的模型，直接用於推論 3 部從未見過的國際標準測試影片 (共 207 筆新樣本)。",
            "Foreman (中動態人物)：RF 達 95.05% F1 / 96.38% AUC；MLP 達 96.00% F1 / 97.06% AUC。",
            "Bus (高細節街景公車)：RF 達 87.23% F1 / 94.32% AUC。",
            "Football (極高動態美式足球)：ROC-AUC 達 100.0%！因劇烈運動每個 Frame 封包量極大，對頻寬壓縮更敏感，適度調降分類機率門檻即可完美命中。",
            "三部未知影片整體表現：Precision 高達 96.88%，整體 ROC-AUC 達 95.64%！",
        ],
        C_AMBER,
    )

    add_notes(s8, "【講者講稿 - 第 8 頁：跨影片泛化測試】\n為了驗證模型是不是只會背單一影片的答案，我們下載了三部模型完全沒看過、動態複雜度截然不同的國際標準影片：Foreman、Bus 跟 Football 美式足球。在不重新訓練的情況下直接推論，三部影片整體的 ROC-AUC 高達 95.64%、Precision 達 96.88%，證明模型學到的是真正的傳輸層與緩衝區物理規律。")

    # =========================================================================
    # Slide 9: With ML vs Without ML (Split Charts 1 & 2)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. 正面對決 (一)：有模型主動防凍 vs. 原本沒模型 — 凍結率與緩衝延遲", "當 RF 模型預警風險機率升高時，主動透過 RTCRtpSender.setParameters 降載保幀率", "核心對比 1 & 2")
    add_footer(s9, 9)

    add_picture_fitted(s9, os.path.join(CHART_DIR, "split_1_freeze_rate.png"), 0.4, 1.3, 6.1, 4.45)
    add_picture_fitted(s9, os.path.join(CHART_DIR, "split_2_jitter_buffer_delay.png"), 6.8, 1.3, 6.1, 4.45)

    add_card(
        s9, 0.45, 5.85, 12.4, 1.15,
        "💡 關鍵改善解析：為什麼接上 RF 模型能大幅降低畫面凍結與緩衝延遲？",
        [
            "當模型預測凍結機率 ≥ 0.30 時，閉迴路控制器立即設定 degradationPreference = 'maintain-framerate'、縮放解析度 2x 並將單路碼率壓在瓶頸容量內，同時加入 35 Kbps 防震盪門檻避免頻繁吐出巨大 I-Frame，使 Foreman 凍結率由 71.9% 暴降至 15.8%、Football 緩衝延遲降低 63%！",
        ],
        C_GREEN,
    )

    add_notes(s9, "【講者講稿 - 第 9 頁：有模型 vs 沒模型（圖 1 與圖 2）】\n接下來我們把訓練好的隨機森林模型真正接進 WebRTC 發送端做閉迴路主動防凍控制，跟『原本沒模型』進行正面對決。左邊圖 1 可以看到：接上模型後，Foreman 的畫面凍結率從 71.9% 直接暴降到只剩 15.8%，Bus 與 Football 也顯著下降；右邊圖 2 則顯示接收端的 Jitter Buffer 延遲在 Football 上從 885.1 毫秒大幅壓低了 63% 到 326.1 毫秒。")

    # =========================================================================
    # Slide 10: With ML vs Without ML (Split Charts 3 & 4)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. 正面對決 (二)：有模型主動防凍 vs. 原本沒模型 — 排隊延遲與預警準確度", "徹底消除路由器佇列塞爆 (Bufferbloat) + 預警準確度超越所有傳統固定門檻規則", "核心對比 3 & 4")
    add_footer(s10, 10)

    add_picture_fitted(s10, os.path.join(CHART_DIR, "split_3_queueing_rtt.png"), 0.4, 1.3, 6.1, 4.45)
    add_picture_fitted(s10, os.path.join(CHART_DIR, "split_4_prediction_accuracy.png"), 6.8, 1.3, 6.1, 4.45)

    add_card(
        s10, 0.45, 5.85, 12.4, 1.15,
        "💡 關鍵改善解析：消除 Bufferbloat 排隊延遲 + 多維度機器學習全面勝出單一門檻規則",
        [
            "左圖 (圖 3)：主動碼率收斂讓封包不再塞滿 Linux netem 瓶頸佇列，往返延遲 RTT 大降 60%~65%（990ms → 348ms）。右圖 (圖 4)：傳統單看 Loss > 5% 的規則 F1 僅 63.6%，而強化版 RF-Pro 達 93.0% F1 / 95.7% AUC！",
        ],
        C_BLUE,
    )

    add_notes(s10, "【講者講稿 - 第 10 頁：有模型 vs 沒模型（圖 3 與圖 4）】\n左邊圖 3 展示了網路排隊延遲（RTT）的對比：原本沒模型時，發送端硬塞高碼率導致佇列塞爆（Bufferbloat），RTT 高達近 1000 毫秒；接上模型主動抑制碼率後，RTT 大幅下降 60% 到 65%（只剩 348 到 372 毫秒）。右邊圖 4 則比較了預警準確度：傳統沒模型單看掉包率大於 5% 的規則，F1 只有 63.6%，漏抓了一半以上的卡頓；而我們的強化版 RF-Pro 與 Deep MLP-Pro 則達到 93.0% 與 92.2% 的高 F1-Score。")

    # =========================================================================
    # Slide 11: Quantitative Summary Table
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "10. 實測數據總表 (Comprehensive Quantitative Evaluation)", "「原本沒模型」vs.「有預訓練模型主動防凍」跨影片全指標匯總", "數據總表")
    add_footer(s11, 11)

    rows_data = [
        ["測試影片", "控制模式", "畫面凍結率\n(Freeze Rate)", "單窗平均凍結時長\n(Freeze Duration)", "緩衝區延遲\n(JitterBuffer)", "排隊往返延遲\n(RTT ms)", "平均播放幀率\n(Avg FPS)", "累計掉包數\n(Packets Lost)"],
        ["Foreman\n(中動態人物)", "🔴 原本沒模型 (Baseline)", "71.93%", "1588.9 ms", "349.5 ms", "933.5 ms", "5.58 FPS", "987"],
        ["Foreman\n(中動態人物)", "🟢 有 RF 模型主動防凍", "15.79% (-56.1%)", "185.3 ms (-88%)", "235.5 ms (-33%)", "372.1 ms (-60%)", "13.56 FPS (2.4x)", "33 (-96.7%)"],
        ["Bus\n(高細節公車)", "🔴 原本沒模型 (Baseline)", "77.19%", "1034.4 ms", "545.1 ms", "990.2 ms", "5.21 FPS", "688"],
        ["Bus\n(高細節公車)", "🟢 有 RF 模型主動防凍", "52.63% (-24.6%)", "452.2 ms (-56%)", "318.6 ms (-42%)", "347.7 ms (-65%)", "10.24 FPS (2.0x)", "21 (-96.9%)"],
        ["Football\n(極高動態運動)", "🔴 原本沒模型 (Baseline)", "73.68%", "1054.0 ms", "885.1 ms", "600.6 ms", "6.27 FPS", "1,356"],
        ["Football\n(極高動態運動)", "🟢 有 RF 模型主動防凍", "56.14% (-17.5%)", "546.7 ms (-48%)", "326.1 ms (-63%)", "402.3 ms (-33%)", "9.36 FPS (+49%)", "31 (-97.7%)"],
    ]

    table_shape = s11.shapes.add_table(len(rows_data), 8, Inches(0.45), Inches(1.4), Inches(12.4), Inches(5.3))
    table = table_shape.table
    col_widths = [1.45, 2.15, 1.55, 1.65, 1.5, 1.45, 1.4, 1.25]
    for i, w_in in enumerate(col_widths):
        table.columns[i].width = Inches(w_in)

    for r_idx, row in enumerate(rows_data):
        is_header = (r_idx == 0)
        is_ml_row = ("🟢" in row[1])
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if is_header:
                cell.fill.fore_color.rgb = C_NAVY
            elif is_ml_row:
                cell.fill.fore_color.rgb = RGBColor(236, 253, 245)
            else:
                cell.fill.fore_color.rgb = C_WHITE

            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = val
            p.alignment = PP_ALIGN.CENTER if c_idx >= 2 else PP_ALIGN.LEFT
            p.font.name = FONT_FAMILY
            p.font.size = Pt(10.5 if not is_header else 11)
            p.font.bold = (is_header or is_ml_row or c_idx == 0)
            if is_header:
                p.font.color.rgb = C_WHITE
            elif is_ml_row and c_idx >= 1:
                p.font.color.rgb = C_GREEN
            elif "🔴" in val:
                p.font.color.rgb = C_RED
            else:
                p.font.color.rgb = C_TEXT_DARK

    add_notes(s11, "【講者講稿 - 第 11 頁：實測數據總表】\n這張總表整理了三部測試影片在『原本沒模型』與『有預訓練 RF 模型主動防凍』下的六大關鍵指標。可以看到在所有影片上，接上模型後不僅凍結率下降、平均凍結時長縮短 48% 到 88%、緩衝延遲與排隊 RTT 大減，更把因為佇列塞爆而溢位丟棄的封包數（Packets Lost）減少了 96% 以上，同時讓平均播放 FPS 提升了 1.5 到 2.4 倍。")

    # =========================================================================
    # Slide 12: Conclusion & Open-Source Portal
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "11. 結論與開源成果展示 (Conclusion & Open-Source Portal)", "完整程式碼、預訓練模型、300 DPI 圖表與 1,020 筆 CSV 資料集已全數開源", "總結")
    add_footer(s12, 12)

    add_card(
        s12, 0.45, 1.35, 6.05, 5.45,
        "✅ 四大研究與工程貢獻總結",
        [
            "1. 真實核心網路損傷測試床：以 Linux tc-netem 位元遮罩精準攔截 WebRTC 媒體埠，克服傳統 pause() 模擬無法產生真實 NACK 與掉包的缺陷。",
            "2. 實證 Transport-Aware 必要性：證明可靠傳輸 (TCP) 會掩蓋 RTP 掉包並將凍結前兆轉移至 JitterBufferDelay (32.93%)，使傳統模型 F1 由 7.1% 躍升至 92.9%。",
            "3. 22 維因果時序模型 (RF-Pro & Deep MLP-Pro)：導入相對碼率驟降比與 EMA 趨勢，達成 93.04% 當窗 F1、89.24% 下一窗提前 1.5s 預警 F1，跨未知影片 AUC 達 95.64%。",
            "4. 閉迴路主動防凍實證：結合預訓練模型與 RTCRtpSender 動態調控，實測將畫面凍結率由 71.9% 大幅壓低至 15.8%、凍結時長縮短 88%。",
        ],
        C_BLUE,
    )

    add_card(
        s12, 6.8, 1.35, 6.05, 5.45,
        "🌐 GitHub 公開展示網站與開源儲存庫",
        [
            "GitHub Pages 互動展示網站：\nhttps://liu20001107.github.io/webrtc-qoe-ml/",
            "GitHub 完整原始碼與資料集儲存庫：\nhttps://github.com/liu20001107/webrtc-qoe-ml",
            "開源內容包含：",
            "  • scripts/：全部 7 支 Python 機器學習訓練、採集與閉迴路控制程式碼",
            "  • models/：訓練完成之 RF-Pro 與 Deep MLP-Pro (.joblib) 模型權重",
            "  • assets/data/：超過 1,020 筆跨協定、5 分鐘循環與跨影片 CSV 資料集",
            "  • assets/charts/：全部 300 DPI 論文級高解析度實驗圖表",
        ],
        C_GREEN,
    )

    add_notes(s12, "【講者講稿 - 第 12 頁：結論與開源成果】\n總結來說，本研究從底層 Linux tc-netem 真實網路損傷注入出發，完整走完了 WebRTC 遙測採集、跨傳輸協定特徵位移驗證、22 維時序特徵工程、RF-Pro 與 Deep MLP-Pro 訓練、跨影片泛化驗證，最後更落實了閉迴路主動防凍控制。我們所有的程式碼、預訓練模型、圖表與 CSV 資料集都已經完整開源在 GitHub Pages 網站上。謝謝大家！")

    prs.save(OUT_DESKTOP)
    shutil.copy(OUT_DESKTOP, OUT_GH_PAGES)
    print(f"[✓] 已成功生成 12 頁 16:9 專業報告 PPTX: {OUT_DESKTOP}")


if __name__ == "__main__":
    build_presentation()
