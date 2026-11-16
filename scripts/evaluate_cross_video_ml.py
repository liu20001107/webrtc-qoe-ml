#!/usr/bin/env python3
"""
Cross-Video Generalization Test for Pre-Trained Transport-Aware WebRTC QoE Models
1. Trains Random Forest (RF) and MLP strictly on the original training video (`test_video.y4m` = Akiyo News Anchor, 570 samples)
   and persists the trained models (`rf_qoe_model.joblib`, `mlp_qoe_model.joblib`).
2. Streams 3 brand-new, unseen ITU/Xiph standard videos with different spatial & temporal complexity:
   - Video 1: `foreman_cif.y4m` (Medium Motion: Facial expressions + construction site camera pan)
   - Video 2: `bus_cif.y4m` (High Spatial/Temporal: Fast street bus panning)
   - Video 3: `football_cif.y4m` (Extreme Motion: High-speed American Football action)
3. Uses the pre-trained RF & MLP models (without retraining!) to predict Playback Freezes on each unseen video,
   while displaying live video + real-time ML predictions in a visible Chrome window via ngrok!
"""

import os
import sys
import time
import json
import csv
import shutil
import signal
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
if os.path.exists("/run/user/1000/.mutter-Xwaylandauth.7ONZY3"):
    os.environ.setdefault("XAUTHORITY", "/run/user/1000/.mutter-Xwaylandauth.7ONZY3")

font_names = [f.name for f in fm.fontManager.ttflist]
cjk_pref = ["Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
chosen_font = "DejaVu Sans"
for f in cjk_pref:
    if f in font_names:
        chosen_font = f
        break
plt.rcParams["font.sans-serif"] = [chosen_font] + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = "/home/user/文件/mediasoup-sfu/tests"
CHROME_BIN = os.path.join(BASE_DIR, "chrome/chrome-linux64/chrome")
DRIVER_BIN = os.path.join(BASE_DIR, "chrome/chromedriver-linux64/chromedriver")
TC_SCRIPT = os.path.join(BASE_DIR, "tc_netem_ctl.sh")
TARGET_URL = "https://totemically-knifelike-ha.ngrok-free.dev"

TRAIN_CSV = os.path.join(BASE_DIR, "transport_qoe_5min_ngrok.csv")
CROSS_VIDEO_CSV = os.path.join(BASE_DIR, "cross_video_test_dataset.csv")
CROSS_VIDEO_REPORT = os.path.join(BASE_DIR, "cross_video_ml_report.json")
RF_MODEL_PATH = os.path.join(BASE_DIR, "rf_qoe_model.joblib")
MLP_MODEL_PATH = os.path.join(BASE_DIR, "mlp_qoe_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_charts")
DESKTOP_DIR = "/home/user/桌面"
PICTURES_DIR = "/home/user/圖片"

FEATURE_COLS = [
    "rtt_ms",
    "rx_bitrate_kbps",
    "packets_lost_delta",
    "packet_loss_rate_pct",
    "retransmitted_packets_delta",
    "jitter_ms",
    "jitter_buffer_delay_ms",
    "nack_delta",
    "pli_delta",
    "frames_dropped_delta",
]

drivers = []


def clear_tc():
    try:
        subprocess.run(["sudo", "-n", TC_SCRIPT, "clear"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def apply_tc(delay_ms: int, jitter_ms: int, loss_pct: float, rate_kbit: int):
    try:
        subprocess.run(
            ["sudo", "-n", TC_SCRIPT, "set", str(delay_ms), str(jitter_ms), str(loss_pct), str(rate_kbit)],
            check=True,
        )
    except Exception as e:
        print(f"[-] 套用 tc netem 失敗: {e}")


def cleanup(sig=None, frame=None):
    clear_tc()
    for d in drivers:
        try:
            d.quit()
        except Exception:
            pass
    drivers.clear()
    if sig:
        sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def train_and_save_models():
    print("================================================================================")
    print("🧠 [步驟 1] 載入原訓練集 (test_video.y4m / Akiyo, 570 筆) 訓練並保存 RF 與 MLP 模型")
    print("================================================================================")
    df_train = pd.read_csv(TRAIN_CSV)
    X_train = df_train[FEATURE_COLS].fillna(0).values
    y_train = df_train["playback_freeze_label"].astype(int).values

    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, class_weight="balanced"
    )
    mlp_clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=600, random_state=42),
    )

    rf_clf.fit(X_train, y_train)
    mlp_clf.fit(X_train, y_train)

    joblib.dump(rf_clf, RF_MODEL_PATH)
    joblib.dump(mlp_clf, MLP_MODEL_PATH)
    print(f"  [✓] Random Forest 模型已保存至: {RF_MODEL_PATH}")
    print(f"  [✓] MLP 神經網路模型已保存至: {MLP_MODEL_PATH}")
    return rf_clf, mlp_clf


def launch_browser(name: str, room_id: str, video_file: str, headless: bool = True):
    options = webdriver.ChromeOptions()
    if os.path.exists(CHROME_BIN):
        options.binary_location = CHROME_BIN
    if headless:
        options.add_argument("--headless=new")
    else:
        options.add_argument("--window-size=1480,920")
        options.add_argument("--window-position=60,40")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--use-fake-ui-for-media-stream")
    options.add_argument("--use-fake-device-for-media-stream")
    options.add_argument(f"--use-file-for-fake-video-capture={video_file}")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--autoplay-policy=no-user-gesture-required")

    service = Service(executable_path=DRIVER_BIN) if os.path.exists(DRIVER_BIN) else None
    driver = webdriver.Chrome(service=service, options=options) if service else webdriver.Chrome(options=options)
    drivers.append(driver)

    driver.get(TARGET_URL)
    time.sleep(1.5)

    for b in driver.find_elements(By.TAG_NAME, "button"):
        if "visit" in b.text.lower():
            b.click()
            time.sleep(1.5)
            break

    wait = WebDriverWait(driver, 20)
    room_input = wait.until(EC.presence_of_element_located((By.ID, "roomInput")))
    room_input.clear()
    room_input.send_keys(room_id)

    name_input = wait.until(EC.presence_of_element_located((By.ID, "nameInput")))
    name_input.clear()
    name_input.send_keys(name)

    cam_checkbox = wait.until(EC.presence_of_element_located((By.ID, "checkCam")))
    if not cam_checkbox.is_selected():
        driver.execute_script("arguments[0].click();", cam_checkbox)

    join_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnJoin")))
    join_btn.click()
    return driver


def sample_raw_webrtc_stats(driver):
    stats_js = """
    return (async () => {
        let out = {
            candidateProtocol: 'udp',
            rttMs: 0,
            packetsReceived: 0,
            packetsLost: 0,
            bytesReceived: 0,
            retransmittedPacketsReceived: 0,
            jitterMs: 0,
            jitterBufferDelayS: 0,
            jitterBufferTargetDelayS: 0,
            jitterBufferMinimumDelayS: 0,
            jitterBufferEmittedCount: 0,
            fps: 0,
            framesReceived: 0,
            framesDecoded: 0,
            framesDropped: 0,
            keyFramesDecoded: 0,
            totalDecodeTimeS: 0,
            totalInterFrameDelayS: 0,
            nackCount: 0,
            pliCount: 0,
            firCount: 0,
            qpSum: 0,
            freezeCount: 0,
            totalFreezesDurationS: 0,
            uiFreezeCount: 0,
            uiFreezeRatioPct: 0.0
        };

        const recvPc = window.recvPeerConnection;
        let videoStreamCount = 0;
        if (recvPc) {
            try {
                const stats = await recvPc.getStats();
                let localCandidateIds = new Set();
                stats.forEach(r => {
                    if (r.type === 'candidate-pair' && r.state === 'succeeded') {
                        if (r.currentRoundTripTime !== undefined && r.currentRoundTripTime > 0) {
                            out.rttMs = Math.max(out.rttMs, r.currentRoundTripTime * 1000);
                        }
                        if (r.localCandidateId) localCandidateIds.add(r.localCandidateId);
                    }
                });
                stats.forEach(r => {
                    if (r.type === 'local-candidate' && localCandidateIds.has(r.id) && r.protocol) {
                        out.candidateProtocol = r.protocol;
                    }
                    if (r.type === 'inbound-rtp' && r.kind === 'video') {
                        videoStreamCount++;
                        out.packetsReceived += (r.packetsReceived || 0);
                        out.packetsLost += Math.max(0, r.packetsLost || 0);
                        out.bytesReceived += (r.bytesReceived || 0);
                        out.retransmittedPacketsReceived += (r.retransmittedPacketsReceived || 0);
                        out.jitterMs += ((r.jitter || 0) * 1000);
                        out.jitterBufferDelayS += (r.jitterBufferDelay || 0);
                        out.jitterBufferTargetDelayS += (r.jitterBufferTargetDelay || 0);
                        out.jitterBufferMinimumDelayS += (r.jitterBufferMinimumDelay || 0);
                        out.jitterBufferEmittedCount += (r.jitterBufferEmittedCount || 0);
                        out.fps += (r.framesPerSecond || 0);
                        out.framesReceived += (r.framesReceived || 0);
                        out.framesDecoded += (r.framesDecoded || 0);
                        out.framesDropped += (r.framesDropped || 0);
                        out.keyFramesDecoded += (r.keyFramesDecoded || 0);
                        out.totalDecodeTimeS += (r.totalDecodeTime || 0);
                        out.totalInterFrameDelayS += (r.totalInterFrameDelay || 0);
                        out.nackCount += (r.nackCount || 0);
                        out.pliCount += (r.pliCount || 0);
                        out.firCount += (r.firCount || 0);
                        out.qpSum += (r.qpSum || 0);
                        out.freezeCount += (r.freezeCount || 0);
                        out.totalFreezesDurationS += (r.totalFreezesDuration || 0);
                    }
                });
                if (videoStreamCount > 1) {
                    out.jitterMs /= videoStreamCount;
                    out.fps /= videoStreamCount;
                }
            } catch (e) {}
        }

        const freezeEls = document.querySelectorAll('.freeze-stats');
        let uiRatios = [];
        freezeEls.forEach(el => {
            const text = el.innerText || '';
            const m = text.match(/凍結:\\s*(\\d+)次\\s*\\(([0-9.]+)%\\)/);
            if (m) {
                out.uiFreezeCount += parseInt(m[1], 10);
                uiRatios.push(parseFloat(m[2]));
            }
        });
        if (uiRatios.length > 0) {
            out.uiFreezeRatioPct = uiRatios.reduce((a, b) => a + b, 0) / uiRatios.length;
        }
        return out;
    })();
    """
    try:
        return driver.execute_script(stats_js)
    except Exception:
        return None


def test_single_video(video_key: str, video_title: str, video_path: str, rf_clf, mlp_clf, samples_per_phase: int = 6):
    global drivers
    drivers = []
    rows = []

    print("\n" + "=" * 80)
    print(f"🎬 開始測試未見過的新影片: 【{video_title}】 ({os.path.basename(video_path)})")
    print("=" * 80)

    clear_tc()
    room_id = "office"

    d1 = launch_browser(f"{video_key}_發送端1", room_id=room_id, video_file=video_path, headless=True)
    time.sleep(1.0)
    d2 = launch_browser(f"{video_key}_發送端2", room_id=room_id, video_file=video_path, headless=True)
    time.sleep(1.5)
    gui_driver = launch_browser(f"{video_key}_主控觀測", room_id=room_id, video_file=video_path, headless=False)
    time.sleep(4.0)

    # Inject live ML prediction banner on the visible webpage
    try:
        gui_driver.execute_script(f"""
            const btnAdap = document.getElementById('btnToggleAdaptive');
            if (btnAdap && btnAdap.innerText.includes('開啟')) btnAdap.click();
            const btnBot = document.getElementById('btnToggleBottleneck');
            if (btnBot && btnBot.innerText.includes('惡劣擁塞')) btnBot.click();

            const body = document.getElementById('qoe-dashboard-body');
            if (body && !document.getElementById('ml-live-pred-box')) {{
                const mlDiv = document.createElement('div');
                mlDiv.id = 'ml-live-pred-box';
                mlDiv.style.cssText = 'background:#311042; border:2px solid #c084fc; border-radius:8px; padding:8px 10px; font-size:12px; font-weight:bold; color:#f3e8ff; text-align:center;';
                mlDiv.innerHTML = '🤖 測試影片: <span style="color:#fde047;">{video_title}</span><br><span id="ml-pred-text">正在等待模型即時推論...</span>';
                body.insertBefore(mlDiv, body.firstChild);
            }}
        """)
    except Exception:
        pass

    phases = [
        ("good_wan", "🟢 正常網路 (0% Loss)", 12, 3, 0.0, 12000),
        ("moderate_loss_jitter", "🟡 中度掉包 (3.5% Loss)", 30, 12, 3.5, 3000),
        ("congested_bottleneck", "🟠 擁塞瓶頸 (7.5% Loss, 850k)", 55, 22, 7.5, 850),
        ("severe_impairment", "🔴 重度損傷 (14% Loss, 400k)", 85, 35, 14.0, 400),
    ]

    prev_state = {}
    for phase_code, phase_label, delay, jitter, loss, rate in phases:
        print(f"  [*] [{video_key}] 切換網路階段 -> {phase_label}")
        apply_tc(delay, jitter, loss, rate)
        try:
            gui_driver.execute_script(f"""
                const lbl = document.getElementById('qoe-phase-label');
                if (lbl) lbl.innerText = '目前階段: {phase_label} [{video_key}]';
            """)
        except Exception:
            pass

        for step in range(samples_per_phase):
            time.sleep(1.5)
            now = time.time()

            for idx, (u_name, d) in enumerate([(f"{video_key}_1", d1), (f"{video_key}_2", d2), (f"{video_key}_GUI", gui_driver)]):
                raw = sample_raw_webrtc_stats(d)
                if not raw:
                    continue

                prev = prev_state.get(idx)
                prev_state[idx] = (now, raw)
                if not prev:
                    continue

                prev_t, prev_raw = prev
                dt = max(0.1, now - prev_t)

                d_bytes_rx = max(0, raw["bytesReceived"] - prev_raw["bytesReceived"])
                d_pkts_rx = max(0, raw["packetsReceived"] - prev_raw["packetsReceived"])
                d_pkts_lost = max(0, raw["packetsLost"] - prev_raw["packetsLost"])
                d_retx_rx = max(0, raw["retransmittedPacketsReceived"] - prev_raw["retransmittedPacketsReceived"])

                d_jb_delay_s = max(0.0, raw["jitterBufferDelayS"] - prev_raw["jitterBufferDelayS"])
                d_jb_emitted = max(0, raw["jitterBufferEmittedCount"] - prev_raw["jitterBufferEmittedCount"])

                d_frames_dec = max(0, raw["framesDecoded"] - prev_raw["framesDecoded"])
                d_frames_drop = max(0, raw["framesDropped"] - prev_raw["framesDropped"])
                d_inter_frame_s = max(0.0, raw["totalInterFrameDelayS"] - prev_raw["totalInterFrameDelayS"])

                d_nack = max(0, raw["nackCount"] - prev_raw["nackCount"])
                d_pli = max(0, raw["pliCount"] - prev_raw["pliCount"])

                d_freeze_cnt = max(0, raw["freezeCount"] - prev_raw["freezeCount"])
                d_freeze_dur_ms = max(0.0, (raw["totalFreezesDurationS"] - prev_raw["totalFreezesDurationS"]) * 1000.0)
                d_ui_freeze_cnt = max(0, raw["uiFreezeCount"] - prev_raw["uiFreezeCount"])

                rx_bitrate_kbps = (d_bytes_rx * 8.0) / (dt * 1000.0)
                total_expected = d_pkts_rx + d_pkts_lost
                pkt_loss_rate_pct = (d_pkts_lost / total_expected * 100.0) if total_expected > 0 else 0.0
                jb_delay_ms = (d_jb_delay_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0
                avg_inter_frame_ms = (d_inter_frame_s / d_frames_dec * 1000.0) if d_frames_dec > 0 else (1000.0 / max(1.0, raw["fps"]))
                fps = raw["fps"] if raw["fps"] > 0 else (d_frames_dec / dt)

                is_freeze = 1 if (
                    d_freeze_cnt > 0
                    or d_freeze_dur_ms > 30.0
                    or d_ui_freeze_cnt > 0
                    or (avg_inter_frame_ms > 90.0 and fps < 14.0)
                ) else 0

                feat_dict = {
                    "rtt_ms": round(raw["rttMs"], 2),
                    "rx_bitrate_kbps": round(rx_bitrate_kbps, 2),
                    "packets_lost_delta": d_pkts_lost,
                    "packet_loss_rate_pct": round(pkt_loss_rate_pct, 3),
                    "retransmitted_packets_delta": d_retx_rx,
                    "jitter_ms": round(raw["jitterMs"], 3),
                    "jitter_buffer_delay_ms": round(jb_delay_ms, 3),
                    "nack_delta": d_nack,
                    "pli_delta": d_pli,
                    "frames_dropped_delta": d_frames_drop,
                }

                x_vec = np.array([[feat_dict[c] for c in FEATURE_COLS]])
                rf_pred = int(rf_clf.predict(x_vec)[0])
                rf_prob = float(rf_clf.predict_proba(x_vec)[0, 1])
                mlp_pred = int(mlp_clf.predict(x_vec)[0])
                mlp_prob = float(mlp_clf.predict_proba(x_vec)[0, 1])

                # Update live ML inference display on the GUI browser
                if idx == 2:
                    match_str = "✅ 預測命中" if rf_pred == is_freeze else "⚠️ 邊界誤差"
                    pred_lbl = f"🚨 預測凍結 ({rf_prob*100:.0f}%)" if rf_pred == 1 else f"🟢 預測順暢 ({(1-rf_prob)*100:.0f}%)"
                    actual_lbl = "🚨 實際凍結" if is_freeze == 1 else "🟢 實際順暢"
                    try:
                        gui_driver.execute_script(f"""
                            const el = document.getElementById('ml-pred-text');
                            if (el) el.innerText = 'RF 模型: {pred_lbl} | {actual_lbl} ({match_str})';
                        """)
                    except Exception:
                        pass

                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "video_name": video_key,
                    "video_title": video_title,
                    "network_phase": phase_code,
                    "user_id": u_name,
                    **feat_dict,
                    "fps": round(fps, 2),
                    "avg_inter_frame_delay_ms": round(avg_inter_frame_ms, 2),
                    "freeze_count_delta": max(d_freeze_cnt, d_ui_freeze_cnt),
                    "freeze_duration_delta_ms": round(d_freeze_dur_ms, 2),
                    "actual_freeze_label": is_freeze,
                    "rf_pred_label": rf_pred,
                    "rf_freeze_prob": round(rf_prob, 4),
                    "mlp_pred_label": mlp_pred,
                    "mlp_freeze_prob": round(mlp_prob, 4),
                }
                rows.append(row)

    clear_tc()
    cleanup()
    print(f"  [✓] 影片 【{video_title}】 實測推論完成！共收集 {len(rows)} 筆樣本。")
    return rows


def plot_cross_video_results(df_all: pd.DataFrame, summary: dict):
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    videos = list(summary["per_video"].keys())
    v_titles = [summary["per_video"][v]["title"] for v in videos]

    # (A) Bar chart: RF Accuracy, F1-Score, ROC-AUC across the 3 unseen videos
    ax_a = axs[0, 0]
    x = np.arange(len(videos))
    w = 0.25
    rf_acc = [summary["per_video"][v]["Random_Forest"]["accuracy"] for v in videos]
    rf_f1 = [summary["per_video"][v]["Random_Forest"]["f1_score"] for v in videos]
    rf_auc = [summary["per_video"][v]["Random_Forest"]["roc_auc"] for v in videos]

    b1 = ax_a.bar(x - w, rf_acc, width=w, color="#2563eb", label="Accuracy (準確率 %)")
    b2 = ax_a.bar(x, rf_f1, width=w, color="#059669", label="F1-Score (%)")
    b3 = ax_a.bar(x + w, rf_auc, width=w, color="#d97706", label="ROC-AUC (%)")
    for bars in [b1, b2, b3]:
        for b in bars:
            val = b.get_height()
            ax_a.text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(v_titles, fontweight="bold", fontsize=9.5)
    ax_a.set_ylim(50, 108)
    ax_a.set_ylabel("百分比 (%)", fontweight="bold")
    ax_a.set_title("(A) 預訓練 Random Forest 模型在 3 部未知新影片上之泛化表現", fontsize=12, fontweight="bold")
    ax_a.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_a.legend(loc="lower right", fontsize=9)

    # (B) ROC Curves for each unseen video (RF & MLP)
    ax_b = axs[0, 1]
    colors = {"Foreman": "#2563eb", "Bus": "#059669", "Football": "#dc2626"}
    for v in videos:
        sub = df_all[df_all["video_name"] == v]
        y_true = sub["actual_freeze_label"].values
        y_prob_rf = sub["rf_freeze_prob"].values
        fpr, tpr, _ = roc_curve(y_true, y_prob_rf)
        auc_val = roc_auc_score(y_true, y_prob_rf)
        ax_b.plot(fpr, tpr, linewidth=2.3, color=colors.get(v, "#1d4ed8"), label=f"RF on {summary['per_video'][v]['title']} (AUC={auc_val:.3f})")
    ax_b.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax_b.set_xlabel("False Positive Rate (誤報率)", fontweight="bold")
    ax_b.set_ylabel("True Positive Rate (召回率)", fontweight="bold")
    ax_b.set_title("(B) 跨影片測試集 ROC 曲線 (零重訓直接推論)", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4)
    ax_b.legend(loc="lower right", fontsize=8.8)

    # (C) Overall Confusion Matrix on all 3 unseen videos combined
    ax_c = axs[1, 0]
    cm = np.array(summary["overall_unseen_videos"]["Random_Forest"]["confusion_matrix"])
    ax_c.imshow(cm, cmap="Greens")
    ax_c.set_xticks([0, 1])
    ax_c.set_yticks([0, 1])
    ax_c.set_xticklabels(["模型預測: 順暢 (0)", "模型預測: 凍結 (1)"], fontweight="bold")
    ax_c.set_yticklabels(["實際發生: 順暢 (0)", "實際發生: 凍結 (1)"], fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm.sum() * 100
            ax_c.text(j, i, f"{val} 筆\n({pct:.1f}%)", ha="center", va="center",
                      color="white" if val > cm.max() / 2 else "#0f172a", fontsize=12, fontweight="bold")
    ax_c.set_title(f"(C) 3 部未知新影片總合混淆矩陣 (共 {len(df_all)} 筆新影片樣本)", fontsize=12, fontweight="bold")

    # (D) Bitrate & Freeze Probability Comparison across Video Motion Profiles
    ax_d = axs[1, 1]
    phases_order = ["good_wan", "moderate_loss_jitter", "congested_bottleneck", "severe_impairment"]
    phase_names_zh = ["正常網路\n(0% Loss)", "中度掉包\n(3.5% Loss)", "擁塞瓶頸\n(7.5% Loss)", "重度損傷\n(14% Loss)"]
    px = np.arange(len(phases_order))
    for v in videos:
        sub = df_all[df_all["video_name"] == v]
        prob_means = [sub[sub["network_phase"] == p]["rf_freeze_prob"].mean() * 100 for p in phases_order]
        ax_d.plot(px, prob_means, marker="o", linewidth=2.2, color=colors.get(v, "#1d4ed8"), label=f"{summary['per_video'][v]['title']} (預測凍結率)")
    ax_d.set_xticks(px)
    ax_d.set_xticklabels(phase_names_zh, fontweight="bold")
    ax_d.set_ylim(-5, 108)
    ax_d.set_ylabel("模型預測凍結機率 (%)", fontweight="bold")
    ax_d.set_title("(D) 各網路損傷階段下模型對不同動態複雜度影片之凍結預測曲線", fontsize=12, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4)
    ax_d.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "qoe_cross_video_generalization.png")
    plt.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(PICTURES_DIR, "qoe_cross_video_generalization.png"))
    shutil.copy(out_path, os.path.join(DESKTOP_DIR, "WebRTC_跨影片模型實測對比圖.png"))
    plt.close()
    print(f"[✓] 已產出跨影片泛化測試圖表: {out_path}")
    return out_path


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_true, y_prob) * 100, 2),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    rf_clf, mlp_clf = train_and_save_models()

    test_videos = [
        ("Foreman", "Foreman (中動態人物與工地轉場)", os.path.join(BASE_DIR, "foreman_cif.y4m")),
        ("Bus", "Bus (高細節街景公車快速橫移)", os.path.join(BASE_DIR, "bus_cif.y4m")),
        ("Football", "Football (極高動態美式足球賽事)", os.path.join(BASE_DIR, "football_cif.y4m")),
    ]

    all_rows = []
    try:
        for v_key, v_title, v_path in test_videos:
            rows = test_single_video(v_key, v_title, v_path, rf_clf, mlp_clf, samples_per_phase=6)
            all_rows.extend(rows)
            time.sleep(2.0)
    finally:
        clear_tc()

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(CROSS_VIDEO_CSV, index=False, encoding="utf-8")

    summary = {"per_video": {}, "overall_unseen_videos": {}}
    for v_key, v_title, _ in test_videos:
        sub = df_all[df_all["video_name"] == v_key]
        y_t = sub["actual_freeze_label"].values
        summary["per_video"][v_key] = {
            "title": v_title,
            "samples": len(sub),
            "avg_good_wan_bitrate_kbps": round(float(sub[sub["network_phase"] == "good_wan"]["rx_bitrate_kbps"].mean()), 2),
            "actual_freeze_rate_pct": round(float(y_t.mean() * 100), 2),
            "Random_Forest": evaluate_predictions(y_t, sub["rf_pred_label"].values, sub["rf_freeze_prob"].values),
            "MLP": evaluate_predictions(y_t, sub["mlp_pred_label"].values, sub["mlp_freeze_prob"].values),
        }

    y_all = df_all["actual_freeze_label"].values
    summary["overall_unseen_videos"] = {
        "total_samples": len(df_all),
        "Random_Forest": evaluate_predictions(y_all, df_all["rf_pred_label"].values, df_all["rf_freeze_prob"].values),
        "MLP": evaluate_predictions(y_all, df_all["mlp_pred_label"].values, df_all["mlp_freeze_prob"].values),
    }

    with open(CROSS_VIDEO_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_cross_video_results(df_all, summary)

    print("\n" + "=" * 80)
    print("✅ 跨影片泛化實測完成！")
    print(f"📁 新影片推論資料集: {CROSS_VIDEO_CSV}")
    print(f"📁 評估報告 JSON: {CROSS_VIDEO_REPORT}")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
