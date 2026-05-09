// SPDX-License-Identifier: GPL-2.0
// xdp_prog.c — Phase 3 XDP/eBPF DDoS Early Warning Kernel Program
//
// Design: "Keep eBPF dumb, keep Go smart"
//   - All counters are uint64 integers (no floating point in eBPF)
//   - Go controller reads these maps and computes rate features for ONNX
//   - Byte counts use bpf_ntohs() to handle network byte order

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>

// ──────────────────────────────────────────────────────────────────────
// Struct: flow_stats
// Raw counters accumulated per source IP. Must match Go struct exactly.
// 12 fields × 8 bytes = 96 bytes (well within 512-byte eBPF stack limit)
// ──────────────────────────────────────────────────────────────────────
struct flow_stats {
    __u64 first_seen_ns;      // Timestamp of first packet (bpf_ktime_get_ns)
    __u64 last_seen_ns;       // Timestamp of most recent packet
    __u64 fwd_packet_count;   // Total forward (ingress) packets
    __u64 bwd_packet_count;   // Total backward packets (placeholder, always 0 in XDP ingress)
    __u64 fwd_byte_count;     // Total forward bytes (via bpf_ntohs(ip->tot_len))
    __u64 bwd_byte_count;     // Total backward bytes (placeholder, always 0)
    __u64 sum_iat_ns;         // Sum of inter-arrival times in nanoseconds

    // TCP Flag Counters
    __u64 syn_flag_count;
    __u64 ack_flag_count;
    __u64 rst_flag_count;
    __u64 psh_flag_count;
    __u64 urg_flag_count;
};

// ──────────────────────────────────────────────────────────────────────
// Map 1: flow_map — Per-source-IP flow statistics
// LRU hash auto-evicts stale IPs during DDoS (IP spoofing resilience)
// ──────────────────────────────────────────────────────────────────────
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);                // Source IPv4 address
    __type(value, struct flow_stats);
} flow_map SEC(".maps");

// ──────────────────────────────────────────────────────────────────────
// Map 2: drop_map — Blacklist populated by Go controller
// When Go detects attack score > 0.5, it inserts the IP here.
// XDP checks this map on every packet for fast-path dropping.
// ──────────────────────────────────────────────────────────────────────
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, __u32);   // Source IPv4 address
    __type(value, __u8);  // Drop flag (1 = drop)
} drop_map SEC(".maps");

// ──────────────────────────────────────────────────────────────────────
// Helper: Update TCP flag counters from the TCP header
// ──────────────────────────────────────────────────────────────────────
static __always_inline void update_tcp_flags(struct flow_stats *stats,
                                              struct tcphdr *tcp)
{
    if (tcp->syn) stats->syn_flag_count++;
    if (tcp->ack) stats->ack_flag_count++;
    if (tcp->rst) stats->rst_flag_count++;
    if (tcp->psh) stats->psh_flag_count++;
    if (tcp->urg) stats->urg_flag_count++;
}

// ──────────────────────────────────────────────────────────────────────
// XDP Entry Point: ddos_early_warning
// Processes every incoming packet at the NIC level.
// ──────────────────────────────────────────────────────────────────────
SEC("xdp")
int ddos_early_warning(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data     = (void *)(long)ctx->data;

    // ── Step 1: Parse Ethernet Header ──────────────────────────────
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    // Only process IPv4 packets
    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    // ── Step 2: Parse IPv4 Header ──────────────────────────────────
    struct iphdr *ip = (struct iphdr *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    __u32 src_ip = ip->saddr;

    // ── Step 3: Fast-Path Drop Check (Blacklist) ───────────────────
    // This is the mitigation path — runs at line rate in kernel space
    __u8 *drop_flag = bpf_map_lookup_elem(&drop_map, &src_ip);
    if (drop_flag && *drop_flag == 1) {
        return XDP_DROP;  // 🚫 Packet dropped at NIC speed!
    }

    // ── Step 4: Compute packet length (endianness-safe) ────────────
    // CRITICAL: bpf_ntohs converts network byte order → host byte order
    // Without this, 100 bytes would be misread as 25600 bytes!
    __u16 pkt_len = bpf_ntohs(ip->tot_len);

    // ── Step 5: Get current timestamp ──────────────────────────────
    __u64 now = bpf_ktime_get_ns();

    // ── Step 6: Parse TCP header for flag extraction ───────────────
    struct tcphdr *tcp = NULL;
    if (ip->protocol == IPPROTO_TCP) {
        tcp = (struct tcphdr *)((void *)ip + (ip->ihl * 4));
        if ((void *)(tcp + 1) > data_end)
            tcp = NULL;  // Bounds check failed, skip TCP parsing
    }

    // ── Step 7: Update or Create Flow Statistics ───────────────────
    struct flow_stats *stats = bpf_map_lookup_elem(&flow_map, &src_ip);

    if (stats) {
        // Existing flow: update counters
        __u64 iat = now - stats->last_seen_ns;
        stats->sum_iat_ns += iat;
        stats->last_seen_ns = now;
        stats->fwd_packet_count++;
        stats->fwd_byte_count += pkt_len;

        // Update TCP flags if this is a TCP packet
        if (tcp)
            update_tcp_flags(stats, tcp);
    } else {
        // New flow: initialize all counters
        struct flow_stats new_stats = {0};
        new_stats.first_seen_ns   = now;
        new_stats.last_seen_ns    = now;
        new_stats.fwd_packet_count = 1;
        new_stats.fwd_byte_count  = pkt_len;

        // Set initial TCP flags for the first packet
        if (tcp) {
            if (tcp->syn) new_stats.syn_flag_count = 1;
            if (tcp->ack) new_stats.ack_flag_count = 1;
            if (tcp->rst) new_stats.rst_flag_count = 1;
            if (tcp->psh) new_stats.psh_flag_count = 1;
            if (tcp->urg) new_stats.urg_flag_count = 1;
        }

        bpf_map_update_elem(&flow_map, &src_ip, &new_stats, BPF_ANY);
    }

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
