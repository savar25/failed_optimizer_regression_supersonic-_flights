# ✈️ Failed Optimizer Regression for Supersonic Flights

## Overview
Trajectory optimization is a cornerstone of aerospace engineering, especially for supersonic flight applications. However, traditional optimization methods often suffer from **high computational cost** and **poor convergence**, particularly when dealing with failed or unstable cases.

This project introduces a **data-driven framework** that leverages failed optimization cases to **improve convergence and efficiency**. By combining feature extraction, clustering, and predictive modeling, the system generates **high-quality initial guesses** for trajectory optimizers.

---

##  Key Idea
Instead of discarding failed optimization runs, this project **learns from them**.

We:
1. Extract meaningful features from trajectory data  
2. Cluster similar trajectory behaviors  
3. Predict improved initial profiles for optimization  

This leads to faster, more reliable trajectory optimization.

---

##  Methodology

### 1. Feature Extraction
We extract structured information from trajectory data at selected time steps:
- Mean values  
- Variance  
- Absolute sum of changes  

These features provide a compact statistical representation of both **successful and failed trajectories**.

---

### 2. Clustering (K-Means)
We apply **K-Means clustering** to group trajectory profiles based on similarity.

- Clustering is based on deviation from optimized altitude  
- Helps identify patterns in failed cases  
- Finds trajectories closest to the **best-performing solutions**

---

### 3. Profile Prediction
Using clustered data:
- Select trajectories similar to optimal cases  
- Construct a **low-computation approximation** of an optimal trajectory  
- Use this as an **initial guess** for the optimizer  

---

## Benefits

- Reduced computational cost  
- Improved optimizer convergence  
- Better use of failed simulation data  
- Scalable for Multi-Disciplinary Analysis (MDA) workflows  

---

## Applications

- Supersonic flight trajectory design  
- Aerospace mission planning  
- Multi-disciplinary optimization (MDO/MDA)  
- Simulation acceleration  

---

## Usage 

### Step 1: Prepare Data
python feature_extraction/extract_features.py

### Step 2: Run Clustering
python clustering/kmeans.py

### Step 3: Predict Initial Profiles
python prediction/predict_profile.py

### Step 4: Run Optimization
python optimization/run_optimizer.py
