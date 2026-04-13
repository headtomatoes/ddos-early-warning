# CICIDS2019 EDA Summary: Visual Feature Discrimination

## 1. Dataset Overview & Class Distribution
The dataset exhibits **extreme class imbalance**, which is typical for DDoS traffic captures. Note that the y-axis on all histograms is plotted on a logarithmic scale. 

* **Majority Classes:** Classes such as `TFTP` (grey), `Syn` (pink), and `DrDoS_NTP` (green) frequently reach peak bin counts between 1 million and 10 million packets/flows.
* **Minority Classes:** Classes like `WebDDoS` (cyan) and `BENIGN` traffic are vastly outnumbered, often registering several orders of magnitude lower in frequency across the distributions.
* **Implication:** If trained directly on this data, a model will heavily overfit to `Syn` and `TFTP` attacks. Class weighting or undersampling of the majority attack classes is mandatory before training the ML classifier.

## 2. Top 5 Most Discriminative Features (Visual Analysis)
Based on visual separability—where specific attack classes isolate themselves along the x-axis—the following 5 features are the strongest candidates for the lightweight detection model:

1.  **`Total Length of Fwd Packets`**
    * *Why:* Excellent class isolation. `DrDoS_NTP` (green bars) entirely dominates the mid-to-high spectrum (20,000 to 70,000+ bytes) with a highly uniform distribution. Meanwhile, `TFTP` (grey) forms a massive, isolated spike near zero. 
2.  **`Flow Bytes/s`**
    * *Why:* This feature creates distinct, separate peaks for different attack vectors. `Syn` attacks spike massively at zero, while `DrDoS_SNMP` (red) and `DrDoS_NTP` show distinct, separate clusters further down the x-axis (around 1.5 billion bytes/s).
3.  **`Flow Duration` (and highly correlated IAT features)**
    * *Why:* `Syn` (pink) attacks are completely separated from the rest of the dataset. While most traffic clusters near 0, `Syn` traffic creates a massive, broad bell-curve that stretches across the entire upper duration spectrum.
4.  **`Flow Packets/s`**
    * *Why:* Demonstrates strong multi-class separation. `TFTP` is clustered tightly at the lower end, while `DrDoS_NTP` and `DrDoS_NetBIOS` form distinct, identifiable spikes at higher packet rates (e.g., 1.0M to 1.75M packets/s).
5.  **`Total Fwd Packets`**
    * *Why:* Effectively separates high-volume forward-packet attacks from standard traffic. `DrDoS_NTP` consistently populates the long right tail (60 to 160+ packets), separating it cleanly from the bulk of traffic which remains under 20 packets.

## 3. Features to Discard
* **`Total Backward Packets` & `Total Length of Bwd Packets`:** These features show almost no visual variance. The data is highly concentrated in just 2 or 3 bins, with total overlap among classes. Tracking these in eBPF would waste CPU resources without improving detection accuracy.

