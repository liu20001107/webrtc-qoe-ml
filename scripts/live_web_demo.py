#!/usr/bin/env python3
"""
5-Minute Live Visual WebRTC QoE & Freeze Detection Test via ngrok (300 Seconds)
- Connects 2 background senders + 1 visible GUI observer via https://totemically-knifelike-ha.ngrok-free.dev into room 'office'
- Cycles through 4 real Linux kernel `tc netem` network stages continuously for 5 minutes (300 seconds)
- Updates on-screen countdown timer & telemetry dashboard in real time
- Saves all 1.5s interval samples live to /home/user/文件/mediasoup-sfu/tests/transport_qoe_5min_ngrok.csv
"""

import os
import sys
import time
import json
import csv
import signal
import subprocess
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

BASE_DIR = "/home/user/文件/mediasoup-sfu/tests"
CHROME_BIN = os.path.join(BASE_DIR, "chrome/chrome-linux64/chrome")
DRIVER_BIN = os.path.join(BASE_DIR, "chrome/chromedriver-linux64/chromedriver")
VIDEO_PATH = os.path.join(BASE_DIR, "test_video.y4m")
TC_SCRIPT = os.path.join(BASE_DIR, "tc_netem_ctl.sh")
TARGET_URL = "https://totemically-knifelike-ha.ngrok-free.dev"
ROOM_ID = "office"

OUTPUT_CSV = os.path.join(BASE_DIR, "transport_qoe_5min_ngrok.csv")
OUTPUT_SUMMARY_JSON = os.path.join(BASE_DIR, "transport_qoe_5min_summary.json")
TOTAL_DURATION_SEC = 300  # 5 minutes

drivers = []


def clear_tc():
    try:
        subprocess.run(["sudo", "-n", TC_SCRIPT, "clear"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def cleanup(sig=None, frame=None):
    clear_tc()
    for d in drivers:
        try:
            d.quit()
        except Exception:
            pass
    if sig:
        sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def launch_browser(name: str, headless: bool = True):
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
    if os.path.exists(VIDEO_PATH):
        options.add_argument(f"--use-file-for-fake-video-capture={VIDEO_PATH}")
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
    room_input.send_keys(ROOM_ID)

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


def main():
    clear_tc()
    print(f"[1/3] 透過 ngrok ({TARGET_URL}) 啟動 2 個虛擬發送端進入預設房間 '{ROOM_ID}'...")
    d1 = launch_browser("模擬用戶_1", headless=True)
    time.sleep(1.0)
    d2 = launch_browser("模擬用戶_2", headless=True)
    time.sleep(1.5)

    print(f"[2/3] 在桌面開啟 ngrok 可視化 Chrome 視窗 ({TARGET_URL})...")
    gui_driver = launch_browser("主控觀測端", headless=False)
    time.sleep(4.0)

    # 停用舊的 consumer.pause 模擬，並插入 5 分鐘進度條至右側面板
    try:
        gui_driver.execute_script("""
            const btnAdap = document.getElementById('btnToggleAdaptive');
            if (btnAdap && btnAdap.innerText.includes('開啟')) btnAdap.click();
            const btnBot = document.getElementById('btnToggleBottleneck');
            if (btnBot && btnBot.innerText.includes('惡劣擁塞')) btnBot.click();

            const body = document.getElementById('qoe-dashboard-body');
            if (body && !document.getElementById('qoe-timer-bar')) {
                const timerDiv = document.createElement('div');
                timerDiv.id = 'qoe-timer-bar';
                timerDiv.style.cssText = 'background:#1e3a8a; border:1px solid #60a5fa; border-radius:6px; padding:7px 10px; font-size:12px; font-weight:bold; color:#e0f2fe; text-align:center;';
                timerDiv.innerText = '⏱️ 5分鐘壓測進行中: 00:00 / 05:00 (已採集 0 筆)';
                body.insertBefore(timerDiv, body.firstChild);
            }
        """)
    except Exception:
        pass

    stages = [
        ("good_wan", "🟢 1. 正常廣域網 (0% Loss)", 12, 3, 0.0, 12000),
        ("moderate_loss_jitter", "🟡 2. 中度掉包抖動 (3.5% Loss)", 30, 12, 3.5, 3000),
        ("congested_bottleneck", "🟠 3. 擁塞頻寬瓶頸 (7.5% Loss, 850k)", 55, 22, 7.5, 850),
        ("severe_impairment", "🔴 4. 重度網路損傷 (14% Loss, 400k)", 85, 35, 14.0, 400),
    ]

    fieldnames = [
        "timestamp", "elapsed_sec", "cycle", "network_phase", "user_id", "candidate_protocol",
        "rtt_ms", "rx_bitrate_kbps", "packets_received_delta", "packets_lost_delta", "packet_loss_rate_pct",
        "retransmitted_packets_delta", "retransmission_ratio_pct",
        "jitter_ms", "jitter_buffer_delay_ms", "jitter_buffer_target_delay_ms", "jitter_buffer_min_delay_ms",
        "fps", "frames_received_delta", "frames_decoded_delta", "frames_dropped_delta", "key_frames_decoded_delta",
        "avg_decode_time_ms", "avg_inter_frame_delay_ms", "avg_qp",
        "nack_delta", "pli_delta", "fir_delta",
        "freeze_count_total", "freeze_count_delta", "freeze_duration_delta_ms", "ui_freeze_ratio_pct",
        "label_w3c_native_freeze", "playback_freeze_label"
    ]

    f_csv = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    writer.writeheader()
    f_csv.flush()

    all_rows = []
    prev_state = {}
    start_time = time.time()
    current_stage_idx = -1

    print(f"[3/3] 🚀 開始執行 5 分鐘 (300 秒) 連續循環壓測與即時數據寫入 ({OUTPUT_CSV})...")

    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed >= TOTAL_DURATION_SEC:
            break

        # 每 15 秒切換一次網路階段 (60 秒完成一輪 4 階段，5 分鐘共循環 5 輪)
        stage_idx = int(elapsed // 15) % len(stages)
        cycle_num = int(elapsed // 60) + 1
        phase_code, phase_label, delay, jitter, loss, rate = stages[stage_idx]

        if stage_idx != current_stage_idx:
            current_stage_idx = stage_idx
            print(f"  [⏱️ {int(elapsed):03d}s / 300s | 第 {cycle_num}/5 輪] 切換階段 -> {phase_label}")
            try:
                gui_driver.execute_script(f"if (window.applyTcPhase) window.applyTcPhase('{phase_label} (第{cycle_num}輪)', {delay}, {jitter}, {loss}, {rate});")
            except Exception:
                pass

        # 更新網頁上的 5 分鐘倒數計時器
        mm_cur, ss_cur = divmod(int(elapsed), 60)
        try:
            gui_driver.execute_script(f"""
                const tb = document.getElementById('qoe-timer-bar');
                if (tb) tb.innerText = '⏱️ 5分鐘壓測進行中: {mm_cur:02d}:{ss_cur:02d} / 05:00 | 第 {cycle_num}/5 輪 | 已採集 {len(all_rows)} 筆';
            """)
        except Exception:
            pass

        for idx, (u_name, d) in enumerate([("模擬用戶_1", d1), ("模擬用戶_2", d2), ("主控觀測端", gui_driver)]):
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
            d_jb_target_s = max(0.0, raw["jitterBufferTargetDelayS"] - prev_raw["jitterBufferTargetDelayS"])
            d_jb_min_s = max(0.0, raw["jitterBufferMinimumDelayS"] - prev_raw["jitterBufferMinimumDelayS"])
            d_jb_emitted = max(0, raw["jitterBufferEmittedCount"] - prev_raw["jitterBufferEmittedCount"])

            d_frames_rx = max(0, raw["framesReceived"] - prev_raw["framesReceived"])
            d_frames_dec = max(0, raw["framesDecoded"] - prev_raw["framesDecoded"])
            d_frames_drop = max(0, raw["framesDropped"] - prev_raw["framesDropped"])
            d_keyframes = max(0, raw["keyFramesDecoded"] - prev_raw["keyFramesDecoded"])
            d_decode_time_s = max(0.0, raw["totalDecodeTimeS"] - prev_raw["totalDecodeTimeS"])
            d_inter_frame_s = max(0.0, raw["totalInterFrameDelayS"] - prev_raw["totalInterFrameDelayS"])
            d_qp_sum = max(0, raw["qpSum"] - prev_raw["qpSum"])

            d_nack = max(0, raw["nackCount"] - prev_raw["nackCount"])
            d_pli = max(0, raw["pliCount"] - prev_raw["pliCount"])
            d_fir = max(0, raw["firCount"] - prev_raw["firCount"])

            d_freeze_cnt = max(0, raw["freezeCount"] - prev_raw["freezeCount"])
            d_freeze_dur_ms = max(0.0, (raw["totalFreezesDurationS"] - prev_raw["totalFreezesDurationS"]) * 1000.0)
            d_ui_freeze_cnt = max(0, raw["uiFreezeCount"] - prev_raw["uiFreezeCount"])

            rx_bitrate_kbps = (d_bytes_rx * 8.0) / (dt * 1000.0)
            total_expected = d_pkts_rx + d_pkts_lost
            pkt_loss_rate_pct = (d_pkts_lost / total_expected * 100.0) if total_expected > 0 else 0.0
            retx_ratio_pct = (d_retx_rx / d_pkts_rx * 100.0) if d_pkts_rx > 0 else 0.0

            jb_delay_ms = (d_jb_delay_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0
            jb_target_delay_ms = (d_jb_target_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0
            jb_min_delay_ms = (d_jb_min_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0

            avg_decode_ms = (d_decode_time_s / d_frames_dec * 1000.0) if d_frames_dec > 0 else 0.0
            avg_inter_frame_ms = (d_inter_frame_s / d_frames_dec * 1000.0) if d_frames_dec > 0 else (1000.0 / max(1.0, raw["fps"]))
            avg_qp = (d_qp_sum / d_frames_dec) if d_frames_dec > 0 else 0.0
            fps = raw["fps"] if raw["fps"] > 0 else (d_frames_dec / dt)

            w3c_native_freeze = 1 if (d_freeze_cnt > 0 or d_freeze_dur_ms > 30.0) else 0
            is_freeze = 1 if (w3c_native_freeze == 1 or d_ui_freeze_cnt > 0 or (avg_inter_frame_ms > 90.0 and fps < 14.0)) else 0

            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_sec": round(elapsed, 1),
                "cycle": cycle_num,
                "network_phase": phase_code,
                "user_id": u_name,
                "candidate_protocol": raw["candidateProtocol"],
                "rtt_ms": round(raw["rttMs"], 2),
                "rx_bitrate_kbps": round(rx_bitrate_kbps, 2),
                "packets_received_delta": d_pkts_rx,
                "packets_lost_delta": d_pkts_lost,
                "packet_loss_rate_pct": round(pkt_loss_rate_pct, 3),
                "retransmitted_packets_delta": d_retx_rx,
                "retransmission_ratio_pct": round(retx_ratio_pct, 3),
                "jitter_ms": round(raw["jitterMs"], 3),
                "jitter_buffer_delay_ms": round(jb_delay_ms, 3),
                "jitter_buffer_target_delay_ms": round(jb_target_delay_ms, 3),
                "jitter_buffer_min_delay_ms": round(jb_min_delay_ms, 3),
                "fps": round(fps, 2),
                "frames_received_delta": d_frames_rx,
                "frames_decoded_delta": d_frames_dec,
                "frames_dropped_delta": d_frames_drop,
                "key_frames_decoded_delta": d_keyframes,
                "avg_decode_time_ms": round(avg_decode_ms, 3),
                "avg_inter_frame_delay_ms": round(avg_inter_frame_ms, 3),
                "avg_qp": round(avg_qp, 2),
                "nack_delta": d_nack,
                "pli_delta": d_pli,
                "fir_delta": d_fir,
                "freeze_count_total": max(raw["freezeCount"], raw["uiFreezeCount"]),
                "freeze_count_delta": max(d_freeze_cnt, d_ui_freeze_cnt),
                "freeze_duration_delta_ms": round(d_freeze_dur_ms, 2),
                "ui_freeze_ratio_pct": round(raw["uiFreezeRatioPct"], 2),
                "label_w3c_native_freeze": w3c_native_freeze,
                "playback_freeze_label": is_freeze,
            }
            writer.writerow(row)
            all_rows.append(row)

        f_csv.flush()
        time.sleep(1.5)

    f_csv.close()
    clear_tc()

    try:
        gui_driver.execute_script(f"""
            if (window.applyTcPhase) window.applyTcPhase('🟢 5分鐘測試完成！已恢復正常網路 (0% Loss)', 12, 3, 0.0, 12000);
            const tb = document.getElementById('qoe-timer-bar');
            if (tb) {{
                tb.style.background = '#065f46';
                tb.style.borderColor = '#10b981';
                tb.innerText = '✅ 5分鐘壓測已完成 (05:00 / 05:00) | 共採集 {len(all_rows)} 筆真實樣本！';
            }}
        """)
    except Exception:
        pass

    # Save summary by network phase
    summary = {"total_samples": len(all_rows), "duration_sec": TOTAL_DURATION_SEC, "phases": {}}
    for p_code, p_label, _, _, _, _ in stages:
        p_rows = [r for r in all_rows if r["network_phase"] == p_code]
        if not p_rows:
            continue
        summary["phases"][p_code] = {
            "samples": len(p_rows),
            "avg_fps": round(sum(r["fps"] for r in p_rows) / len(p_rows), 2),
            "avg_rtt_ms": round(sum(r["rtt_ms"] for r in p_rows) / len(p_rows), 2),
            "avg_rx_bitrate_kbps": round(sum(r["rx_bitrate_kbps"] for r in p_rows) / len(p_rows), 2),
            "avg_jitter_ms": round(sum(r["jitter_ms"] for r in p_rows) / len(p_rows), 2),
            "avg_jitter_buffer_delay_ms": round(sum(r["jitter_buffer_delay_ms"] for r in p_rows) / len(p_rows), 2),
            "total_packets_lost": sum(r["packets_lost_delta"] for r in p_rows),
            "total_retransmitted_packets": sum(r["retransmitted_packets_delta"] for r in p_rows),
            "total_nack_events": sum(r["nack_delta"] for r in p_rows),
            "total_pli_events": sum(r["pli_delta"] for r in p_rows),
            "total_frames_dropped": sum(r["frames_dropped_delta"] for r in p_rows),
            "w3c_native_freeze_rate_pct": round(sum(r["label_w3c_native_freeze"] for r in p_rows) / len(p_rows) * 100.0, 2),
            "overall_freeze_rate_pct": round(sum(r["playback_freeze_label"] for r in p_rows) / len(p_rows) * 100.0, 2),
        }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"✅ 5 分鐘 (300 秒) ngrok 實測完成！共採集 {len(all_rows)} 筆樣本。")
    print(f"📁 CSV 資料集: {OUTPUT_CSV}")
    print(f"📁 統計摘要 JSON: {OUTPUT_SUMMARY_JSON}")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
