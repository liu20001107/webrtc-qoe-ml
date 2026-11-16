#!/usr/bin/env python3
"""
Transport-Aware QoE Telemetry Collector for WebRTC (Real Linux Kernel tc-netem Edition)
Based on: "Transport-Aware QoE Modeling for WebRTC: Predicting Playback Freezes with Machine Learning"

Uses Linux kernel `tc qdisc netem` on WebRTC media ports (2048..2303) to inject:
- Real random + bursty UDP/TCP packet loss (triggering real RTCP NACKs, RTX retransmissions, PLI/FIR, and decoder frame drops)
- Real WAN propagation delay and normal-distributed network jitter
- Real bandwidth rate shaping (queueing delay / bufferbloat)
"""

import os
import sys
import time
import json
import csv
import signal
import subprocess
import urllib.request
import ssl
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = "/home/user/文件/mediasoup-sfu/tests"
CHROME_BIN = os.path.join(BASE_DIR, "chrome/chrome-linux64/chrome")
DRIVER_BIN = os.path.join(BASE_DIR, "chrome/chromedriver-linux64/chromedriver")
VIDEO_PATH = os.path.join(BASE_DIR, "test_video.y4m")
TC_SCRIPT = os.path.join(BASE_DIR, "tc_netem_ctl.sh")
TARGET_URL = "https://140.118.107.178:3000"

OUTPUT_CSV = os.path.join(BASE_DIR, "transport_qoe_dataset.csv")
OUTPUT_SUMMARY_JSON = os.path.join(BASE_DIR, "transport_qoe_summary.json")

drivers = []
ssl_ctx = ssl._create_unverified_context()


def clear_tc_netem():
    try:
        subprocess.run(["sudo", "-n", TC_SCRIPT, "clear"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def apply_tc_netem(delay_ms: int, jitter_ms: int, loss_pct: float, rate_kbit: int):
    try:
        subprocess.run(
            ["sudo", "-n", TC_SCRIPT, "set", str(delay_ms), str(jitter_ms), str(loss_pct), str(rate_kbit)],
            check=True,
        )
    except Exception as e:
        print(f"[-] 套用 tc netem 失敗: {e}")


def cleanup(sig=None, frame=None):
    clear_tc_netem()
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


def set_server_transport_mode(mode: str):
    url = f"{TARGET_URL}/api/transport-mode?mode={mode}"
    try:
        with urllib.request.urlopen(url, context=ssl_ctx, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[-] 設定伺服器傳輸模式失敗: {e}")
        return None


def create_virtual_user(user_idx: int, room_id: str, transport_mode: str):
    options = webdriver.ChromeOptions()
    if os.path.exists(CHROME_BIN):
        options.binary_location = CHROME_BIN
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--use-fake-ui-for-media-stream")
    options.add_argument("--use-fake-device-for-media-stream")
    if os.path.exists(VIDEO_PATH):
        options.add_argument(f"--use-file-for-fake-video-capture={VIDEO_PATH}")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--blink-settings=imagesEnabled=false")

    if transport_mode == "QUIC_Enabled":
        options.add_argument("--enable-quic")
        options.add_argument("--enable-features=WebRtcQuicTransport,Quic")
    else:
        options.add_argument("--disable-quic")

    if os.path.exists(DRIVER_BIN):
        service = Service(executable_path=DRIVER_BIN)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    driver.get(TARGET_URL)
    wait = WebDriverWait(driver, 15)

    room_input = wait.until(EC.presence_of_element_located((By.ID, "roomInput")))
    room_input.clear()
    room_input.send_keys(room_id)

    name_input = wait.until(EC.presence_of_element_located((By.ID, "nameInput")))
    name_input.clear()
    name_input.send_keys(f"QoEUser_{user_idx}")

    cam_checkbox = wait.until(EC.presence_of_element_located((By.ID, "checkCam")))
    if not cam_checkbox.is_selected():
        driver.execute_script("arguments[0].click();", cam_checkbox)

    join_btn = wait.until(EC.element_to_be_clickable((By.ID, "btnJoin")))
    join_btn.click()
    return driver


def disable_server_fake_bottleneck(driver):
    """Ensure server-side consumer.pause() simulation is OFF so all impairments come purely from Linux tc-netem."""
    try:
        js = """
        const btnAdap = document.getElementById('btnToggleAdaptive');
        if (btnAdap && btnAdap.innerText.includes('開啟')) {
            btnAdap.click();
        }
        const btnBot = document.getElementById('btnToggleBottleneck');
        if (btnBot && btnBot.innerText.includes('惡劣擁塞')) {
            btnBot.click();
        }
        """
        driver.execute_script(js)
    except Exception:
        pass


def sample_raw_webrtc_stats(driver):
    stats_js = """
    return (async () => {
        let out = {
            candidateProtocol: 'udp',
            rttMs: 0,
            availableOutBitrateKbps: 0,
            availableInBitrateKbps: 0,
            packetsSent: 0,
            bytesSent: 0,
            retransmittedPacketsSent: 0,
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
            totalSquaredInterFrameDelayS2: 0,
            nackCount: 0,
            pliCount: 0,
            firCount: 0,
            qpSum: 0,
            freezeCount: 0,
            totalFreezesDurationS: 0,
            pauseCount: 0,
            totalPausesDurationS: 0,
            uiFreezeCount: 0,
            uiFreezeRatioPct: 0.0
        };

        const sendPc = window.peerConnection;
        if (sendPc) {
            try {
                const stats = await sendPc.getStats();
                stats.forEach(r => {
                    if (r.type === 'outbound-rtp' && r.kind === 'video') {
                        out.packetsSent += (r.packetsSent || 0);
                        out.bytesSent += (r.bytesSent || 0);
                        out.retransmittedPacketsSent += (r.retransmittedPacketsSent || 0);
                    }
                    if (r.type === 'remote-inbound-rtp' && r.kind === 'video') {
                        if (r.roundTripTime !== undefined) {
                            out.rttMs = Math.max(out.rttMs, r.roundTripTime * 1000);
                        }
                    }
                    if (r.type === 'candidate-pair' && r.state === 'succeeded') {
                        if (r.currentRoundTripTime !== undefined) {
                            out.rttMs = Math.max(out.rttMs, r.currentRoundTripTime * 1000);
                        }
                        if (r.availableOutgoingBitrate !== undefined) {
                            out.availableOutBitrateKbps = Math.round(r.availableOutgoingBitrate / 1000);
                        }
                    }
                });
            } catch (e) {}
        }

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
                        if (r.availableIncomingBitrate !== undefined) {
                            out.availableInBitrateKbps = Math.round(r.availableIncomingBitrate / 1000);
                        }
                        if (r.localCandidateId) localCandidateIds.add(r.localCandidateId);
                    }
                });
                stats.forEach(r => {
                    if (r.type === 'local-candidate' && localCandidateIds.has(r.id)) {
                        if (r.protocol) out.candidateProtocol = r.protocol;
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
                        out.totalSquaredInterFrameDelayS2 += (r.totalSquaredInterFrameDelay || 0);
                        out.nackCount += (r.nackCount || 0);
                        out.pliCount += (r.pliCount || 0);
                        out.firCount += (r.firCount || 0);
                        out.qpSum += (r.qpSum || 0);
                        out.freezeCount += (r.freezeCount || 0);
                        out.totalFreezesDurationS += (r.totalFreezesDuration || 0);
                        out.pauseCount += (r.pauseCount || 0);
                        out.totalPausesDurationS += (r.totalPausesDuration || 0);
                    }
                });
                if (videoStreamCount > 1) {
                    out.jitterMs = out.jitterMs / videoStreamCount;
                    out.fps = out.fps / videoStreamCount;
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


def run_collection_for_transport(transport_mode: str, server_mode: str, num_users: int = 3, samples_per_phase: int = 7):
    global drivers
    drivers = []
    rows = []

    print("\n" + "=" * 80)
    print(f"🚀 開始採集傳輸模式: 【{transport_mode}】 (Server Mode: {server_mode.upper()}, 虛擬用戶數: {num_users})")
    print("=" * 80)

    clear_tc_netem()
    set_server_transport_mode(server_mode)
    room_id = f"qoe_{server_mode}_{int(time.time())}"

    for i in range(1, num_users + 1):
        d = create_virtual_user(i, room_id=room_id, transport_mode=transport_mode)
        drivers.append(d)
        print(f"  [+] 虛擬客戶端 QoEUser_{i} 已連線並開始接收串流")
        time.sleep(1.0)

    print("  [~] 等待 WebRTC 媒體通道穩定 (5 秒)...")
    time.sleep(5.0)
    if drivers:
        disable_server_fake_bottleneck(drivers[0])

    prev_state = {}
    # 4 realistic Linux kernel tc-netem network regimes:
    # (phase_name, one_way_delay_ms, jitter_ms, packet_loss_pct, rate_kbit)
    phases = [
        ("good_wan", 12, 3, 0.0, 12000),
        ("moderate_loss_jitter", 30, 12, 3.5, 3000),
        ("congested_bottleneck", 55, 22, 7.5, 850),
        ("severe_impairment", 85, 35, 14.0, 400),
    ]

    for phase_name, delay_ms, jitter_ms, loss_pct, rate_kbit in phases:
        print(f"  [*] 切換真實核心網路階段: [{phase_name}]")
        apply_tc_netem(delay_ms=delay_ms, jitter_ms=jitter_ms, loss_pct=loss_pct, rate_kbit=rate_kbit)

        for step in range(samples_per_phase):
            time.sleep(1.5)
            now = time.time()

            for idx, d in enumerate(drivers):
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
                total_expected_pkts = d_pkts_rx + d_pkts_lost
                pkt_loss_rate_pct = (d_pkts_lost / total_expected_pkts * 100.0) if total_expected_pkts > 0 else 0.0
                retx_ratio_pct = (d_retx_rx / d_pkts_rx * 100.0) if d_pkts_rx > 0 else 0.0

                jb_delay_ms = (d_jb_delay_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0
                jb_target_delay_ms = (d_jb_target_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0
                jb_min_delay_ms = (d_jb_min_s / d_jb_emitted * 1000.0) if d_jb_emitted > 0 else 0.0
                jb_headroom_ms = jb_delay_ms - jb_min_delay_ms

                avg_decode_ms = (d_decode_time_s / d_frames_dec * 1000.0) if d_frames_dec > 0 else 0.0
                avg_inter_frame_ms = (d_inter_frame_s / d_frames_dec * 1000.0) if d_frames_dec > 0 else (1000.0 / max(1.0, raw["fps"]))
                avg_qp = (d_qp_sum / d_frames_dec) if d_frames_dec > 0 else 0.0

                fps = raw["fps"] if raw["fps"] > 0 else (d_frames_dec / dt)

                is_freeze = 1 if (
                    d_freeze_cnt > 0
                    or d_freeze_dur_ms > 40.0
                    or d_ui_freeze_cnt > 0
                    or (avg_inter_frame_ms > 90.0 and fps < 14.0)
                ) else 0

                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transport_mode": transport_mode,
                    "candidate_protocol": raw["candidateProtocol"],
                    "network_phase": phase_name,
                    "user_id": f"QoEUser_{idx + 1}",
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
                    "jitter_buffer_headroom_ms": round(jb_headroom_ms, 3),
                    "fps": round(fps, 2),
                    "frames_received_delta": d_frames_rx,
                    "frames_decoded_delta": d_frames_dec,
                    "frames_dropped_delta": d_frames_drop,
                    "key_frames_decoded_delta": d_keyframes,
                    "avg_decode_time_ms": round(avg_decode_ms, 3),
                    "avg_inter_frame_delay_ms": round(avg_inter_frame_ms, 3),
                    "avg_qp": round(avg_qp, 2),
                    "nack_count_Total": raw["nackCount"],
                    "nack_delta": d_nack,
                    "pli_count_Total": raw["pliCount"],
                    "pli_delta": d_pli,
                    "fir_count_Total": raw["firCount"],
                    "fir_delta": d_fir,
                    "freeze_count_total": max(raw["freezeCount"], raw["uiFreezeCount"]),
                    "freeze_count_delta": max(d_freeze_cnt, d_ui_freeze_cnt),
                    "freeze_duration_delta_ms": round(d_freeze_dur_ms, 2),
                    "ui_freeze_ratio_pct": round(raw["uiFreezeRatioPct"], 2),
                    "playback_freeze_label": is_freeze,
                }
                rows.append(row)

    clear_tc_netem()
    cleanup()
    print(f"  [✓] 傳輸模式 【{transport_mode}】 採集完成，共收集 {len(rows)} 筆時間視窗樣本。")
    return rows


def main():
    print("================================================================================")
    print("  📊 Transport-Aware WebRTC QoE 指標自動採集系統 (Linux Kernel tc-netem 真實損傷版)")
    print("  論文對應: Transport-Aware QoE Modeling for WebRTC: Predicting Playback Freezes")
    print("================================================================================")

    all_rows = []
    configs = [
        ("UDP_Standard", "udp"),
        ("QUIC_Enabled", "quic"),
        ("TCP_Fallback", "tcp"),
    ]

    try:
        for transport_label, srv_mode in configs:
            rows = run_collection_for_transport(
                transport_mode=transport_label,
                server_mode=srv_mode,
                num_users=3,
                samples_per_phase=7,
            )
            all_rows.extend(rows)
            time.sleep(2.0)
    finally:
        clear_tc_netem()
        set_server_transport_mode("udp")

    if not all_rows:
        print("[-] 未收集到任何數據，請確認伺服器狀態。")
        sys.exit(1)

    fieldnames = list(all_rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {}
    for mode, _ in configs:
        m_rows = [r for r in all_rows if r["transport_mode"] == mode]
        if not m_rows:
            continue
        freeze_samples = sum(r["playback_freeze_label"] for r in m_rows)
        summary[mode] = {
            "total_windows": len(m_rows),
            "candidate_protocol_observed": m_rows[-1]["candidate_protocol"],
            "freeze_windows": freeze_samples,
            "freeze_incidence_pct": round(freeze_samples / len(m_rows) * 100.0, 2),
            "avg_fps": round(sum(r["fps"] for r in m_rows) / len(m_rows), 2),
            "avg_rtt_ms": round(sum(r["rtt_ms"] for r in m_rows) / len(m_rows), 2),
            "avg_jitter_ms": round(sum(r["jitter_ms"] for r in m_rows) / len(m_rows), 3),
            "avg_jitter_buffer_delay_ms": round(sum(r["jitter_buffer_delay_ms"] for r in m_rows) / len(m_rows), 2),
            "avg_inter_frame_delay_ms": round(sum(r["avg_inter_frame_delay_ms"] for r in m_rows) / len(m_rows), 2),
            "avg_rx_bitrate_kbps": round(sum(r["rx_bitrate_kbps"] for r in m_rows) / len(m_rows), 2),
            "total_packets_lost": sum(r["packets_lost_delta"] for r in m_rows),
            "total_retransmitted_packets": sum(r["retransmitted_packets_delta"] for r in m_rows),
            "total_frames_dropped": sum(r["frames_dropped_delta"] for r in m_rows),
            "total_nack_events": sum(r["nack_delta"] for r in m_rows),
            "total_pli_events": sum(r["pli_delta"] for r in m_rows),
        }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"✅ 全部指標採集完成！總樣本數: {len(all_rows)} 筆")
    print(f"📁 完整特徵資料集 (CSV): {OUTPUT_CSV}")
    print(f"📁 各傳輸模式統計摘要 (JSON): {OUTPUT_SUMMARY_JSON}")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
