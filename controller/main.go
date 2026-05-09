// Phase 3: Go Controller — DDoS Early Warning System
// Reads eBPF flow statistics, computes 34 FEATURES_V3 features,
// runs ONNX inference, and blacklists IPs with attack score > 0.5.
//
// Design decisions from Master Guide:
//   - uint64 kernel counters (no floats in eBPF)
//   - log1p + MinMaxScaler baked into ONNX (no transforms in Go)
//   - bpf_ntohs for endianness (done in C)
//   - 200ms polling interval
//   - 34 FEATURES_V3 feature vector
//   - Score threshold of 0.5

package main

import (
	"encoding/binary"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	ort "github.com/yalue/onnxruntime_go"
)

// ──────────────────────────────────────────────────────────────────────
// FlowStats mirrors the C struct flow_stats exactly.
// Must match C struct alignment (8-byte boundaries, 12 × uint64 = 96 bytes).
// ──────────────────────────────────────────────────────────────────────
type FlowStats struct {
	FirstSeenNs    uint64
	LastSeenNs     uint64
	FwdPacketCount uint64
	BwdPacketCount uint64
	FwdByteCount   uint64
	BwdByteCount   uint64
	SumIatNs       uint64
	SynFlagCount   uint64
	AckFlagCount   uint64
	RstFlagCount   uint64
	PshFlagCount   uint64
	UrgFlagCount   uint64
}

// Configuration constants from deploy_config.json and Master Guide
const (
	scoreThreshold  = 0.5
	pollingInterval = 200 * time.Millisecond
	numFeatures     = 34
	networkIface    = "ens160"
)

// uint32ToIP converts a uint32 (in network byte order) to a readable IP string.
func uint32ToIP(ip uint32) string {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, ip)
	return net.IP(b).String()
}

// assembleFeatures converts raw eBPF counters into the 34-element FEATURES_V3
// float32 array expected by the ONNX model.
//
// FEATURES_V3 order (from w6_onnx_export.py / w5_train_final.py):
//
//	 0: Flow Bytes/s
//	 1: Total Length of Fwd Packets
//	 2: Flow Packets/s
//	 3: Flow IAT Mean (microseconds)
//	 4: Flow Duration (microseconds)
//	 5: Total Backward Packets
//	 6: SYN Flag Count
//	 7: ACK Flag Count
//	 8: Protocol
//	 9: Destination Port
//	10: Flow IAT Std
//	11: Flow IAT Max
//	12: Total Fwd Packets
//	13: Total Length of Bwd Packets
//	14: Down/Up Ratio
//	15: Packet Length Std
//	16: Packet Length Variance
//	17: Average Packet Size
//	18: Bwd Packet Length Std
//	19: Fwd Packet Length Std
//	20: RST Flag Count
//	21: PSH Flag Count
//	22: URG Flag Count
//	23: Init_Win_bytes_forward
//	24: Init_Win_bytes_backward
//	25: Active Std
//	26: Idle Std
//	27: Active Mean
//	28: Idle Mean
//	29: Inbound
//	30: Subflow Fwd Packets
//	31: Subflow Bwd Packets
//	32: Bwd Packets/s
//	33: Fwd Packets/s
//
// Note: Features that cannot be computed from the minimal eBPF counters
// (Std, Variance, Init_Win_bytes, Active/Idle, IAT Std/Max, Inbound, Port)
// are set to 0. The ONNX model handles log1p(0)=0 gracefully.
// The model's primary decision drivers are rate-based features and flag counts.
func assembleFeatures(stats *FlowStats) [numFeatures]float32 {
	// Calculate duration in seconds (prevent division by zero)
	durationNs := float64(stats.LastSeenNs - stats.FirstSeenNs)
	durationSec := durationNs / 1e9
	if durationSec <= 0 {
		durationSec = 0.001 // 1ms minimum
	}
	durationMicros := durationNs / 1e3 // For Flow Duration feature (microseconds)

	totalFwdPkts := float64(stats.FwdPacketCount)
	totalBwdPkts := float64(stats.BwdPacketCount)
	totalPkts := totalFwdPkts + totalBwdPkts
	totalFwdBytes := float64(stats.FwdByteCount)
	totalBwdBytes := float64(stats.BwdByteCount)
	totalBytes := totalFwdBytes + totalBwdBytes

	// Rates
	flowBytesPerSec := totalBytes / durationSec
	flowPktsPerSec := totalPkts / durationSec
	fwdPktsPerSec := totalFwdPkts / durationSec
	bwdPktsPerSec := totalBwdPkts / durationSec

	// IAT Mean in microseconds
	var flowIATMean float64
	if totalPkts > 1 {
		flowIATMean = float64(stats.SumIatNs) / (totalPkts - 1) / 1e3
	}

	// Average packet size
	var avgPacketSize float64
	if totalPkts > 0 {
		avgPacketSize = totalBytes / totalPkts
	}

	// Down/Up Ratio
	var downUpRatio float64
	if totalFwdPkts > 0 {
		downUpRatio = totalBwdPkts / totalFwdPkts
	}

	var features [numFeatures]float32
	features[0] = float32(flowBytesPerSec)               // Flow Bytes/s
	features[1] = float32(totalFwdBytes)                  // Total Length of Fwd Packets
	features[2] = float32(flowPktsPerSec)                 // Flow Packets/s
	features[3] = float32(flowIATMean)                    // Flow IAT Mean (µs)
	features[4] = float32(durationMicros)                 // Flow Duration (µs)
	features[5] = float32(totalBwdPkts)                   // Total Backward Packets
	features[6] = float32(stats.SynFlagCount)             // SYN Flag Count
	features[7] = float32(stats.AckFlagCount)             // ACK Flag Count
	features[8] = 6                                       // Protocol (TCP=6, default)
	features[9] = 0                                       // Destination Port (not tracked)
	features[10] = 0                                      // Flow IAT Std (not computable)
	features[11] = 0                                      // Flow IAT Max (not computable)
	features[12] = float32(totalFwdPkts)                  // Total Fwd Packets
	features[13] = float32(totalBwdBytes)                 // Total Length of Bwd Packets
	features[14] = float32(downUpRatio)                   // Down/Up Ratio
	features[15] = 0                                      // Packet Length Std
	features[16] = 0                                      // Packet Length Variance
	features[17] = float32(avgPacketSize)                 // Average Packet Size
	features[18] = 0                                      // Bwd Packet Length Std
	features[19] = 0                                      // Fwd Packet Length Std
	features[20] = float32(stats.RstFlagCount)            // RST Flag Count
	features[21] = float32(stats.PshFlagCount)            // PSH Flag Count
	features[22] = float32(stats.UrgFlagCount)            // URG Flag Count
	features[23] = 0                                      // Init_Win_bytes_forward
	features[24] = 0                                      // Init_Win_bytes_backward
	features[25] = 0                                      // Active Std
	features[26] = 0                                      // Idle Std
	features[27] = 0                                      // Active Mean
	features[28] = 0                                      // Idle Mean
	features[29] = 0                                      // Inbound
	features[30] = float32(totalFwdPkts)                  // Subflow Fwd Packets (= Total Fwd Packets)
	features[31] = float32(totalBwdPkts)                  // Subflow Bwd Packets (= Total Bwd Packets)
	features[32] = float32(bwdPktsPerSec)                 // Bwd Packets/s
	features[33] = float32(fwdPktsPerSec)                 // Fwd Packets/s

	return features
}

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)
	log.Println("═══════════════════════════════════════════════════")
	log.Println("  DDoS Early Warning System — Phase 3 Controller  ")
	log.Println("═══════════════════════════════════════════════════")

	// ── Step 1: Resolve paths ──────────────────────────────────────
	execDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	xdpObjPath := filepath.Join(execDir, "..", "xdp", "xdp_prog.o")
	onnxModelPath := filepath.Join(execDir, "xgboost_final.onnx")
	onnxLibPath := filepath.Join(execDir, "..", "onnxruntime-linux-x64-1.18.1", "lib", "libonnxruntime.so")

	// Check that required files exist
	for _, f := range []string{xdpObjPath, onnxModelPath, onnxLibPath} {
		if _, err := os.Stat(f); os.IsNotExist(err) {
			log.Fatalf("Required file not found: %s", f)
		}
	}

	log.Printf("XDP object:  %s", xdpObjPath)
	log.Printf("ONNX model:  %s", onnxModelPath)
	log.Printf("ONNX lib:    %s", onnxLibPath)

	// ── Step 2: Initialize ONNX Runtime ────────────────────────────
	log.Println("\n[1/4] Initializing ONNX Runtime...")
	ort.SetSharedLibraryPath(onnxLibPath)
	err = ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()
	log.Printf("ONNX Runtime version: %s", ort.GetVersion())

	// Create input/output tensors for the ONNX session.
	// Input:  "float_input" — shape [1, 34]
	// Output: "label" — shape [1] (int64 class label)
	//         "probabilities" — shape [1, N] (probability per class)
	inputShape := ort.NewShape(1, int64(numFeatures))
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	// Output tensors: label (int64) and probabilities (float32)
	// For the probabilities output, we need to know the number of classes.
	// The model was trained with multi:softprob, so probabilities has shape [1, num_classes].
	// We'll use DynamicAdvancedSession to let ONNX allocate outputs automatically.

	// Use AdvancedSession with pre-allocated tensors for label output,
	// but we need probabilities output to be dynamically shaped.
	// DynamicAdvancedSession is better here since we don't know num_classes at compile time.

	// Create ONNX session
	inputNames := []string{"float_input"}
	outputNames := []string{"label", "probabilities"}

	session, err := ort.NewDynamicAdvancedSession(
		onnxModelPath,
		inputNames,
		outputNames,
		nil, // Use default session options
	)
	if err != nil {
		log.Fatalf("Failed to create ONNX session: %v", err)
	}
	defer session.Destroy()
	log.Println("ONNX session created successfully")

	// ── Step 3: Load eBPF program and attach to NIC ────────────────
	log.Println("\n[2/4] Loading eBPF program...")

	spec, err := ebpf.LoadCollectionSpec(xdpObjPath)
	if err != nil {
		log.Fatalf("Failed to load eBPF collection spec: %v", err)
	}

	coll, err := ebpf.NewCollection(spec)
	if err != nil {
		log.Fatalf("Failed to create eBPF collection: %v", err)
	}
	defer coll.Close()

	xdpProg := coll.Programs["ddos_early_warning"]
	if xdpProg == nil {
		log.Fatal("XDP program 'ddos_early_warning' not found in eBPF object")
	}

	flowMap := coll.Maps["flow_map"]
	if flowMap == nil {
		log.Fatal("eBPF map 'flow_map' not found")
	}

	dropMap := coll.Maps["drop_map"]
	if dropMap == nil {
		log.Fatal("eBPF map 'drop_map' not found")
	}

	log.Printf("eBPF maps loaded — flow_map (LRU, max=%d), drop_map (Hash, max=%d)",
		65536, 10000)

	// Attach XDP program to network interface
	log.Printf("\n[3/4] Attaching XDP to interface '%s'...", networkIface)
	iface, err := net.InterfaceByName(networkIface)
	if err != nil {
		log.Fatalf("Failed to find interface %s: %v", networkIface, err)
	}

	xdpLink, err := link.AttachXDP(link.XDPOptions{
		Program:   xdpProg,
		Interface: iface.Index,
	})
	if err != nil {
		log.Fatalf("Failed to attach XDP to %s: %v", networkIface, err)
	}
	defer xdpLink.Close()
	log.Printf("✅ XDP attached to %s (index %d)", networkIface, iface.Index)

	// ── Step 4: Polling Loop ───────────────────────────────────────
	log.Println("\n[4/4] Starting inference loop (200ms polling)...")
	log.Println("Waiting for traffic... Press Ctrl+C to stop.")
	log.Println("────────────────────────────────────────────────────")

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	ticker := time.NewTicker(pollingInterval)
	defer ticker.Stop()

	totalDetections := 0
	totalFlowsProcessed := uint64(0)

	for {
		select {
		case <-sigChan:
			log.Println("\n────────────────────────────────────────────────────")
			log.Printf("Shutting down. Total detections: %d, Total flows processed: %d",
				totalDetections, totalFlowsProcessed)
			log.Println("Detaching XDP program...")
			return

		case <-ticker.C:
			var ip uint32
			var stats FlowStats

			iter := flowMap.Iterate()
			for iter.Next(&ip, &stats) {
				totalFlowsProcessed++

				// Skip flows with too few packets (noise)
				if stats.FwdPacketCount < 5 {
					continue
				}

				// Assemble the 34-feature vector
				features := assembleFeatures(&stats)

				// Copy features into the input tensor
				inputData := inputTensor.GetData()
				copy(inputData, features[:])

				// Run ONNX inference
				// DynamicAdvancedSession: outputs are allocated by ORT
				outputs := []ort.Value{nil, nil}
				err := session.Run([]ort.Value{inputTensor}, outputs)
				if err != nil {
					log.Printf("⚠️  ONNX inference error: %v", err)
					continue
				}

				// Extract attack probability from the probabilities output
				// probabilities output is a map (ZipMap) or tensor depending on export
				// We need to handle both cases
				probOutput := outputs[1]
				if probOutput == nil {
					log.Printf("⚠️  Nil probability output")
					continue
				}

				// The output could be a Sequence of Maps (ZipMap enabled) or a Tensor
				// Try to get the attack score
				var attackScore float32

				// For ZipMap output (Sequence<Map<int64, float>>), the probability
				// output is a sequence. We need to handle this carefully.
				// Try tensor first:
				switch v := probOutput.(type) {
				case *ort.Tensor[float32]:
					// Direct tensor output — probabilities shape [1, num_classes]
					data := v.GetData()
					if len(data) >= 2 {
						// Class 0 = BENIGN, others = attack types
						// Attack score = 1 - P(BENIGN)
						// Actually for multi-class: sum of all non-benign probabilities
						attackScore = 1.0 - data[0] // data[0] is P(BENIGN) for class index 0
					} else if len(data) == 1 {
						attackScore = data[0]
					}
				default:
					// For ZipMap or other complex outputs, we need to inspect
					// Let's try to get probabilities from the label output instead
					labelOutput := outputs[0]
					if labelOutput != nil {
						if labelTensor, ok := labelOutput.(*ort.Tensor[int64]); ok {
							labelData := labelTensor.GetData()
							if len(labelData) > 0 && labelData[0] != 0 {
								// Non-benign label predicted
								attackScore = 1.0
							}
						}
					}
				}

				// Clean up dynamically allocated outputs
				for _, o := range outputs {
					if o != nil {
						o.Destroy()
					}
				}

				// Threshold enforcement
				ipStr := uint32ToIP(ip)
				if attackScore > scoreThreshold {
					totalDetections++
					log.Printf("🚨 DDoS DETECTED from IP %s | Score: %.4f | Pkts: %d | Bytes: %d",
						ipStr, attackScore, stats.FwdPacketCount, stats.FwdByteCount)

					// Insert IP into drop_map for kernel-level blocking
					dropVal := uint8(1)
					err := dropMap.Update(ip, dropVal, ebpf.UpdateAny)
					if err != nil {
						log.Printf("⚠️  Failed to update drop_map for %s: %v", ipStr, err)
					} else {
						log.Printf("🛡️  IP %s added to blacklist — packets will be dropped at line rate", ipStr)
					}
				}
			}

			if err := iter.Err(); err != nil {
				log.Printf("⚠️  Map iteration error: %v", err)
			}
		}
	}
}
