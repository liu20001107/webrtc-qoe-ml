#!/usr/bin/env bash
set -e

ACTION="${1:-clear}"

if [ "$ACTION" = "clear" ]; then
    tc qdisc del dev lo root 2>/dev/null || true
    echo "[tc netem] 已清除 lo 網卡的所有流量控制規則 (恢復預設)"
    exit 0
fi

if [ "$ACTION" = "set" ]; then
    DELAY_MS="${2:-20}"
    JITTER_MS="${3:-5}"
    LOSS_PCT="${4:-0}"
    RATE_KBIT="${5:-10000}"

    tc qdisc del dev lo root 2>/dev/null || true

    # 建立 4 個佇列通道 (prio bands 4)，預設所有一般流量 (priomap 全 0) 走 1:1 (完全不延遲、不掉包)
    tc qdisc add dev lo root handle 1: prio bands 4 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

    # 在通道 1:4 掛載 netem，模擬真實廣域網延遲、抖動、隨機掉包與頻寬上限
    if [ "$LOSS_PCT" = "0" ] || [ "$LOSS_PCT" = "0.0" ]; then
        tc qdisc add dev lo parent 1:4 handle 40: netem delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution normal rate "${RATE_KBIT}kbit"
    else
        tc qdisc add dev lo parent 1:4 handle 40: netem delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution normal loss "${LOSS_PCT}%" 25% rate "${RATE_KBIT}kbit"
    fi

    # 僅將 mediasoup WebRTC 媒體連接埠 (2048..2303, 即 0x0800/0xff00) 導向 1:4 netem 通道
    tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip sport 0x0800 0xff00 flowid 1:4
    tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip dport 0x0800 0xff00 flowid 1:4

    echo "[tc netem] 已套用真實網路損傷 -> 單向延遲:${DELAY_MS}ms±${JITTER_MS}ms | 掉包率:${LOSS_PCT}% | 頻寬限制:${RATE_KBIT}kbps"
    exit 0
fi
