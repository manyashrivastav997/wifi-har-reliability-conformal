# WiCaP: Reliable WiFi-Based Human Activity Recognition under Environment Shift

## 📌 Overview

WiFi-based Human Activity Recognition (HAR) uses Channel State Information (CSI) to detect human activities without cameras or wearables.
However, models trained in one environment often fail in real-world settings due to **environmental shifts**.

This project proposes **WiCaP**, a framework to improve **prediction reliability** and reduce **overconfident errors**.

---

## ❗ Problem Statement

* Models assume training and test data follow the same distribution
* Real-world environments change (rooms, devices, users)
* Model still gives **high-confidence wrong predictions**

👉 This leads to **silent failures**, which are dangerous in real applications

---

## ⚙️ Experimental Setup

### 📊 Dataset

* **UT-HAR Dataset**
* WiFi CSI-based human activity data
* Includes multiple indoor activities (walking, sitting, etc.)

---

### 🔬 What I Did

* Train model on one environment
* Test on a different environment (distribution shift)
* Evaluate prediction reliability

---

### 🧠 Methods Compared

* **ERM (Empirical Risk Minimization)** → baseline deep learning
* **APS (Adaptive Prediction Sets)** → uncertainty estimation
* **RAPS (Regularized APS)** → improved conformal prediction
* **Selective Prediction** → rejects low-confidence outputs

---

## 📈 Results & Observations

* ERM achieves **~98% accuracy** in controlled settings
* Under environment shift:

  * High-confidence **wrong predictions**
  * Leads to **silent failures**

### ✅ Conformal Methods (APS, RAPS)

* Provide uncertainty-aware predictions
* Maintain coverage guarantees
* Reduce incorrect confident outputs

### ✅ Selective Prediction

* Rejects uncertain predictions
* Improves safety in critical scenarios

---

## 📉 Silent Failure Rate (SFR)

We introduce a new metric:

> **SFR = fraction of high-confidence incorrect predictions**

### 🔍 Insight

* ERM → High SFR ❌ (dangerous)
* WiCaP → Low SFR ✅ (reliable)

---

## 🧾 Conclusion

* Accuracy alone is **not sufficient**
* Models must be **reliability-aware**
* Conformal prediction helps handle uncertainty
* Selective prediction avoids risky decisions

👉 **Final Takeaway:**
WiCaP significantly improves reliability of WiFi HAR systems under real-world conditions

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py
python test.py
```

---

## 🌍 Applications

* Smart Homes
* Healthcare Monitoring
* Elderly Assistance Systems
* Wireless Sensing

---

## 📄 Research Paper

👉 Published Version (DOI):
https://doi.org/10.5281/zenodo.20621329

👉 Direct PDF (Google Drive):
https://drive.google.com/file/d/1tqv7knt7PcOj797Ylm6rIUAfvVc4f74I/view?usp=sharing

---

