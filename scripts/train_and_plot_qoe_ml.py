#!/usr/bin/env python3
"""
Transport-Aware WebRTC QoE Machine Learning & Time-Series Visualization Suite
Based on: "Transport-Aware QoE Modeling for WebRTC: Predicting Playback Freezes with Machine Learning"

1. Generates 5-Minute (300s, 5-Cycle) Multi-Metric Time-Series Plot from `transport_qoe_5min_ngrok.csv`
2. Trains Random Forest (RF) & Multi-Layer Perceptron (MLP) models (both Current-Window & Next-Window t->t+1 Forecasting)
3. Analyzes Transport-Aware Feature Importance Shifts across UDP, QUIC, and TCP from `transport_qoe_dataset.csv`
4. Saves high-resolution (300 DPI) charts to `output_charts/` and `/home/user/圖片/`
"""

import os
import json
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance

# Setup Traditional Chinese CJK font
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
CSV_5MIN = os.path.join(BASE_DIR, "transport_qoe_5min_ngrok.csv")
CSV_TRANSPORT = os.path.join(BASE_DIR, "transport_qoe_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_charts")
PICTURES_DIR = "/home/user/圖片"
DESKTOP_DIR = "/home/user/桌面"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)

# Strictly transport & media-buffer telemetry features (NO fps or inter_frame_delay to prevent target leakage!)
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

FEATURE_LABELS_ZH = {
    "rtt_ms": "往返延遲 (RTT ms)",
    "rx_bitrate_kbps": "接收碼率 (Bitrate Kbps)",
    "packets_lost_delta": "單窗掉包數 (Packets Lost)",
    "packet_loss_rate_pct": "掉包率 (Loss Rate %)",
    "retransmitted_packets_delta": "RTX 重傳封包數",
    "jitter_ms": "網路抖動 (Jitter ms)",
    "jitter_buffer_delay_ms": "緩衝區延遲 (JitterBuffer ms)",
    "nack_delta": "NACK 重傳請求數",
    "pli_delta": "PLI 關鍵幀請求數",
    "frames_dropped_delta": "解碼器掉幀數 (Frames Dropped)",
}


def plot_5min_timeseries(df_5min: pd.DataFrame):
    """Plot 5-minute (300s) multi-metric time-series across the 5 network impairment cycles."""
    # Aggregate across the 3 concurrent receivers at each timestamp step
    df_5min["time_bin"] = (df_5min["elapsed_sec"] // 1.5) * 1.5
    grp = df_5min.groupby("time_bin").agg({
        "rx_bitrate_kbps": "mean",
        "fps": "mean",
        "packets_lost_delta": "sum",
        "nack_delta": "sum",
        "retransmitted_packets_delta": "sum",
        "jitter_ms": "mean",
        "jitter_buffer_delay_ms": "mean",
        "rtt_ms": "mean",
        "freeze_duration_delta_ms": "mean",
        "playback_freeze_label": "mean",
    }).reset_index()

    t = grp["time_bin"].values

    fig, axs = plt.subplots(4, 1, figsize=(16, 13), dpi=300, sharex=True)
    fig.patch.set_facecolor("#ffffff")

    stage_colors = [
        (0, 15, "#d1fae5", "正常網路 (0% Loss)"),
        (15, 30, "#fef3c7", "中度掉包 (3.5% Loss)"),
        (30, 45, "#ffedd5", "擁塞瓶頸 (7.5% / 850k)"),
        (45, 60, "#fee2e2", "重度損傷 (14% / 400k)"),
    ]

    for ax in axs:
        for cycle in range(5):
            base_t = cycle * 60
            for s_start, s_end, color, lbl in stage_colors:
                ax.axvspan(
                    base_t + s_start,
                    base_t + s_end,
                    color=color,
                    alpha=0.55,
                    label=lbl if (cycle == 0 and ax == axs[0]) else None,
                )
        ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 1: Bitrate & FPS
    ax1 = axs[0]
    ax1_r = ax1.twinx()
    l1 = ax1.plot(t, grp["rx_bitrate_kbps"], color="#1d4ed8", linewidth=2.0, label="接收碼率 (Bitrate Kbps)")
    l2 = ax1_r.plot(t, grp["fps"], color="#047857", linewidth=2.0, linestyle="-.", label="畫面幀率 (FPS)")
    ax1.set_ylabel("接收碼率 (Kbps)", color="#1d4ed8", fontweight="bold")
    ax1_r.set_ylabel("幀率 (FPS)", color="#047857", fontweight="bold")
    ax1.set_title("WebRTC 5 分鐘 (300 秒 / 5 輪循環) 真實 Linux tc-netem 網路損傷與凍結時序分析", fontsize=15, fontweight="bold", pad=12)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", ncol=3, fontsize=9, framealpha=0.9)

    # Panel 2: Packets Lost & NACK requests
    ax2 = axs[1]
    ax2_r = ax2.twinx()
    ax2.bar(t, grp["packets_lost_delta"], width=1.2, color="#dc2626", alpha=0.75, label="單窗真實掉包數 (Packets Lost)")
    ax2_r.plot(t, grp["nack_delta"], color="#7c3aed", linewidth=1.8, label="單窗 NACK 重傳請求數")
    ax2.set_ylabel("掉包數 (Packets)", color="#dc2626", fontweight="bold")
    ax2_r.set_ylabel("NACK 請求數", color="#7c3aed", fontweight="bold")
    h1, lb1 = ax2.get_legend_handles_labels()
    h2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, lb1 + lb2, loc="upper right", fontsize=9, framealpha=0.9)

    # Panel 3: Jitter & Jitter Buffer Delay
    ax3 = axs[2]
    ax3.plot(t, grp["jitter_ms"], color="#d97706", linewidth=2.0, label="網路抖動 (Network Jitter ms)")
    ax3.plot(t, grp["jitter_buffer_delay_ms"], color="#0284c7", linewidth=2.0, label="緩衝區延遲 (Jitter Buffer Delay ms)")
    ax3.set_ylabel("延遲與抖動 (ms)", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Panel 4: Freeze Duration & Freeze Probability
    ax4 = axs[3]
    ax4_r = ax4.twinx()
    ax4.fill_between(t, 0, grp["freeze_duration_delta_ms"], color="#ef4444", alpha=0.45, label="W3C 單窗畫面凍結時長 (ms)")
    ax4_r.step(t, grp["playback_freeze_label"] * 100.0, where="mid", color="#991b1b", linewidth=2.0, label="凍結發生率 (%)")
    ax4.set_ylabel("凍結時長 (ms)", color="#b91c1c", fontweight="bold")
    ax4_r.set_ylabel("凍結發生率 (%)", color="#991b1b", fontweight="bold")
    ax4_r.set_ylim(-5, 108)
    ax4.set_xlabel("測試時間 (秒) — 每 60 秒完成一輪四階段網路損傷循環 (共 300 秒)", fontsize=12, fontweight="bold")
    h1, lb1 = ax4.get_legend_handles_labels()
    h2, lb2 = ax4_r.get_legend_handles_labels()
    ax4.legend(h1 + h2, lb1 + lb2, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "qoe_5min_timeseries_curves.png")
    plt.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(PICTURES_DIR, "qoe_5min_timeseries_curves.png"))
    shutil.copy(out_path, os.path.join(DESKTOP_DIR, "WebRTC_5分鐘時序分析圖.png"))
    plt.close()
    print(f"[✓] 已產出 5 分鐘時序曲線圖: {out_path}")
    return out_path


def evaluate_models_on_5min(df_5min: pd.DataFrame):
    """
    Trains Random Forest (RF) and Multi-Layer Perceptron (MLP) on the 5-minute dataset:
      Task 1: Current-Window Freeze Prediction (using strictly transport & buffer features, no FPS leakage)
      Task 2: Next-Window (t -> t+1) Freeze Forecasting (predicting next 1.5s window freeze from current telemetry)
    """
    df = df_5min.copy()
    # Create Next-Window (t+1) target per user stream
    df["next_window_freeze_label"] = df.groupby("user_id")["playback_freeze_label"].shift(-1)
    df_next = df.dropna(subset=["next_window_freeze_label"]).copy()
    df_next["next_window_freeze_label"] = df_next["next_window_freeze_label"].astype(int)

    X_curr = df[FEATURE_COLS].fillna(0).values
    y_curr = df["playback_freeze_label"].astype(int).values

    X_next = df_next[FEATURE_COLS].fillna(0).values
    y_next = df_next["next_window_freeze_label"].astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42, class_weight="balanced")
    mlp_clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=600, random_state=42),
    )

    results = {}
    roc_data = {}

    for task_name, X, y in [
        ("Current_Window_Freeze", X_curr, y_curr),
        ("Next_Window_Forecast_t+1", X_next, y_next),
    ]:
        results[task_name] = {}
        roc_data[task_name] = {}
        for model_name, clf in [("Random Forest (RF)", rf_clf), ("MLP (Neural Net)", mlp_clf)]:
            y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
            y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            auc = roc_auc_score(y, y_prob)
            cm = confusion_matrix(y, y_pred).tolist()
            fpr, tpr, _ = roc_curve(y, y_prob)

            results[task_name][model_name] = {
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "roc_auc": round(auc * 100, 2),
                "confusion_matrix": cm,
            }
            roc_data[task_name][model_name] = (fpr, tpr, auc, y, y_pred)

    # Fit RF on full 5-min dataset to extract feature importances
    rf_clf.fit(X_curr, y_curr)
    imp_curr = rf_clf.feature_importances_
    rf_clf.fit(X_next, y_next)
    imp_next = rf_clf.feature_importances_

    # Plot Model Comparison Figure (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # (A) Bar chart: RF vs MLP metrics on Current Window & Next Window Forecast
    ax_a = axs[0, 0]
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x_pos = np.arange(len(metric_keys))
    w = 0.2

    rf_c = [results["Current_Window_Freeze"]["Random Forest (RF)"][k] for k in metric_keys]
    mlp_c = [results["Current_Window_Freeze"]["MLP (Neural Net)"][k] for k in metric_keys]
    rf_n = [results["Next_Window_Forecast_t+1"]["Random Forest (RF)"][k] for k in metric_keys]
    mlp_n = [results["Next_Window_Forecast_t+1"]["MLP (Neural Net)"][k] for k in metric_keys]

    ax_a.bar(x_pos - 1.5 * w, rf_c, width=w, color="#1d4ed8", label="RF (當窗凍結判定)")
    ax_a.bar(x_pos - 0.5 * w, mlp_c, width=w, color="#60a5fa", label="MLP (當窗凍結判定)")
    ax_a.bar(x_pos + 0.5 * w, rf_n, width=w, color="#059669", label="RF (下一窗 t+1 預測)")
    ax_a.bar(x_pos + 1.5 * w, mlp_n, width=w, color="#34d399", label="MLP (下一窗 t+1 預測)")
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(metric_labels, fontweight="bold")
    ax_a.set_ylim(50, 105)
    ax_a.set_ylabel("百分比 (%)", fontweight="bold")
    ax_a.set_title("(A) Random Forest vs. MLP 凍結預測效能指標 (5-Fold CV)", fontsize=12, fontweight="bold")
    ax_a.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_a.legend(fontsize=8.5, loc="lower right")

    # (B) ROC Curves
    ax_b = axs[0, 1]
    for task_label, task_key, ls in [("當窗判定 (t)", "Current_Window_Freeze", "-"), ("下一窗預測 (t+1)", "Next_Window_Forecast_t+1", "--")]:
        for m_name, color in [("Random Forest (RF)", "#1d4ed8" if ls == "-" else "#059669"), ("MLP (Neural Net)", "#dc2626" if ls == "-" else "#d97706")]:
            fpr, tpr, auc_val, _, _ = roc_data[task_key][m_name]
            ax_b.plot(fpr, tpr, linestyle=ls, linewidth=2.2, color=color, label=f"{m_name} [{task_label}] (AUC = {auc_val:.3f})")
    ax_b.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax_b.set_xlabel("False Positive Rate (誤報率)", fontweight="bold")
    ax_b.set_ylabel("True Positive Rate (召回率)", fontweight="bold")
    ax_b.set_title("(B) Playback Freeze 預測模型 ROC 曲線對比", fontsize=12, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4)
    ax_b.legend(fontsize=8.5, loc="lower right")

    # (C) Confusion Matrix (Random Forest Current Window)
    ax_c = axs[1, 0]
    cm = np.array(results["Current_Window_Freeze"]["Random Forest (RF)"]["confusion_matrix"])
    im = ax_c.imshow(cm, cmap="Blues")
    ax_c.set_xticks([0, 1])
    ax_c.set_yticks([0, 1])
    ax_c.set_xticklabels(["預測: 順暢 (0)", "預測: 凍結 (1)"], fontweight="bold")
    ax_c.set_yticklabels(["實際: 順暢 (0)", "實際: 凍結 (1)"], fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm.sum() * 100
            ax_c.text(j, i, f"{val} 筆\n({pct:.1f}%)", ha="center", va="center",
                      color="white" if val > cm.max() / 2 else "#0f172a", fontsize=12, fontweight="bold")
    ax_c.set_title("(C) Random Forest 混淆矩陣 (570 筆 5 分鐘實測樣本)", fontsize=12, fontweight="bold")

    # (D) Feature Importance (Current vs Next Window Forecast)
    ax_d = axs[1, 1]
    order = np.argsort(imp_curr)
    y_pos = np.arange(len(FEATURE_COLS))
    labels_ord = [FEATURE_LABELS_ZH[FEATURE_COLS[i]] for i in order]
    ax_d.barh(y_pos - 0.18, imp_curr[order] * 100, height=0.35, color="#2563eb", label="當窗凍結判定 (t)")
    ax_d.barh(y_pos + 0.18, imp_next[order] * 100, height=0.35, color="#10b981", label="下一窗凍結預警 (t+1)")
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(labels_ord, fontsize=9.5, fontweight="bold")
    ax_d.set_xlabel("特徵重要度 (Feature Importance %)", fontweight="bold")
    ax_d.set_title("(D) 預測畫面凍結之關鍵 WebRTC 指標重要度排序", fontsize=12, fontweight="bold")
    ax_d.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax_d.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "qoe_ml_model_comparison_roc_cm.png")
    plt.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(PICTURES_DIR, "qoe_ml_model_comparison_roc_cm.png"))
    shutil.copy(out_path, os.path.join(DESKTOP_DIR, "WebRTC_機器學習凍結預測評估圖.png"))
    plt.close()
    print(f"[✓] 已產出機器學習模型評估圖: {out_path}")
    return results


def analyze_transport_aware_shift(df_trans: pd.DataFrame):
    """
    Analyzes how Feature Importance shifts across transport protocols (UDP_Standard, QUIC_Enabled, TCP_Fallback)
    and demonstrates why Transport-Aware modeling outperforms Transport-Agnostic loss-only modeling.
    """
    modes = ["UDP_Standard", "QUIC_Enabled", "TCP_Fallback"]
    mode_importances = {}

    for m in modes:
        sub = df_trans[df_trans["transport_mode"] == m].copy()
        X_m = sub[FEATURE_COLS].fillna(0).values
        y_m = sub["playback_freeze_label"].astype(int).values
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
        rf.fit(X_m, y_m)
        perm = permutation_importance(rf, X_m, y_m, n_repeats=15, random_state=42, scoring="f1")
        imp = np.maximum(0, rf.feature_importances_ * 0.5 + np.maximum(0, perm.importances_mean) * 0.5)
        if imp.sum() > 0:
            imp = imp / imp.sum() * 100.0
        mode_importances[m] = {FEATURE_COLS[i]: round(float(imp[i]), 2) for i in range(len(FEATURE_COLS))}

    # Cross-transport experiment:
    # Compare Model 1 (Loss/Retransmission-only: packets_lost, loss_rate, retx, nack, frames_dropped) trained on UDP -> tested on TCP
    # vs Model 2 (Transport-Aware Full Buffer+Delay+Transport features)
    loss_only_cols = ["packets_lost_delta", "packet_loss_rate_pct", "retransmitted_packets_delta", "nack_delta", "frames_dropped_delta"]
    udp_df = df_trans[df_trans["transport_mode"] == "UDP_Standard"]
    tcp_df = df_trans[df_trans["transport_mode"] == "TCP_Fallback"]

    rf_loss_only = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_loss_only.fit(udp_df[loss_only_cols].fillna(0), udp_df["playback_freeze_label"].astype(int))
    pred_tcp_loss_only = rf_loss_only.predict(tcp_df[loss_only_cols].fillna(0))
    f1_tcp_loss_only = f1_score(tcp_df["playback_freeze_label"].astype(int), pred_tcp_loss_only, zero_division=0) * 100

    # Transport-Aware model (with transport_mode encoded + media buffer & RTT dynamics)
    df_all = df_trans.copy()
    df_all["is_tcp"] = (df_all["candidate_protocol"] == "tcp").astype(int)
    ta_cols = FEATURE_COLS + ["is_tcp"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_ta = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    pred_all_ta = cross_val_predict(rf_ta, df_all[ta_cols].fillna(0), df_all["playback_freeze_label"].astype(int), cv=cv)
    tcp_mask = (df_all["transport_mode"] == "TCP_Fallback").values
    f1_tcp_ta = f1_score(df_all.loc[tcp_mask, "playback_freeze_label"].astype(int), pred_all_ta[tcp_mask], zero_division=0) * 100

    # Plot Transport-Aware Feature Shift Figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300, gridspec_kw={"width_ratios": [1.8, 1.0]})
    fig.patch.set_facecolor("#ffffff")

    ax1 = axs[0]
    y_pos = np.arange(len(FEATURE_COLS))
    h = 0.26
    u_vals = [mode_importances["UDP_Standard"][c] for c in FEATURE_COLS]
    q_vals = [mode_importances["QUIC_Enabled"][c] for c in FEATURE_COLS]
    t_vals = [mode_importances["TCP_Fallback"][c] for c in FEATURE_COLS]

    ax1.barh(y_pos - h, u_vals, height=h, color="#2563eb", label="UDP_Standard (標準不可靠傳輸)")
    ax1.barh(y_pos, q_vals, height=h, color="#059669", label="QUIC_Enabled (多工低抖動模式)")
    ax1.barh(y_pos + h, t_vals, height=h, color="#dc2626", label="TCP_Fallback (可靠重傳 / 隊頭阻塞)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([FEATURE_LABELS_ZH[c] for c in FEATURE_COLS], fontsize=10, fontweight="bold")
    ax1.set_xlabel("正規化特徵重要度 (%)", fontweight="bold")
    ax1.set_title("(A) 不同傳輸層協定下之凍結預測特徵重要度位移 (Transport-Aware Shift)", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9.5, loc="lower right")

    ax2 = axs[1]
    bars = ax2.bar(
        ["傳統僅看掉包/重傳模型\n(Loss/NACK-Only on TCP)", "論文 Transport-Aware 模型\n(含 JitterBuffer & RTT 感知)"],
        [f1_tcp_loss_only, f1_tcp_ta],
        color=["#94a3b8", "#16a34a"],
        width=0.48,
    )
    for b in bars:
        val = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, val + 1.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("畫面凍結預測 F1-Score (%)", fontweight="bold")
    ax2.set_title("(B) 跨傳輸層凍結預測 F1-Score 對比\n(證明為何需要 Transport-Aware QoE 模型)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "qoe_transport_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    shutil.copy(out_path, os.path.join(PICTURES_DIR, "qoe_transport_feature_importance.png"))
    shutil.copy(out_path, os.path.join(DESKTOP_DIR, "WebRTC_跨傳輸層特徵重要度對比圖.png"))
    plt.close()
    print(f"[✓] 已產出跨傳輸層特徵重要度對比圖: {out_path}")

    return {
        "mode_feature_importances_pct": mode_importances,
        "tcp_f1_loss_only_model_pct": round(f1_tcp_loss_only, 2),
        "tcp_f1_transport_aware_model_pct": round(f1_tcp_ta, 2),
    }


def main():
    df_5min = pd.read_csv(CSV_5MIN)
    df_trans = pd.read_csv(CSV_TRANSPORT)

    plot_5min_timeseries(df_5min)
    ml_5min_res = evaluate_models_on_5min(df_5min)
    ta_shift_res = analyze_transport_aware_shift(df_trans)

    report = {
        "five_minute_ml_evaluation": ml_5min_res,
        "transport_aware_analysis": ta_shift_res,
    }
    report_path = os.path.join(BASE_DIR, "transport_qoe_ml_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"✅ 機器學習訓練與視覺化圖表全部完成！報告已存至: {report_path}")
    print("=" * 80)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
