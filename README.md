# Neighborhood Business Longevity Bot

> **An Interactive Venture Decision-Support System for Chicago’s Commercial Landscape.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![UIUC MCS](https://img.shields.io/badge/UIUC-MCS-blue.svg)](https://cs.illinois.edu/)
[![City Scholars](https://img.shields.io/badge/Program-City_Scholars-orange.svg)](https://cityscholars.illinois.edu/)

The **Neighborhood Business Longevity Bot** is an end-to-end predictive platform designed to help entrepreneurs mitigate market-entry risk. By fusing 20 years of Chicago business registries with socio-economic census data, the bot forecasts the survival probability of commercial ventures using advanced survival analysis.

Website link: https://stay332.github.io/Chicago-Bussiness-Longevity/

---

## 🚀 Key Features

*   **10-Year Survival Forecasting**: Generates dynamic probability curves based on industry type and localized economic "oxygen" (Rent/Income).
*   **Geospatial Interaction**: A live map of Chicago's active commerce with a **Double-Click Integration**—double-clicking any location on the map instantly feeds spatial metadata into the chatbot for real-time inference.
*   **Competitive Benchmarking**: Automatically identifies and lists nearby active and closed competitors to provide a "reality check" of the local competitive intensity.
*   **Predictive AI Consultant**: A conversational interface that translates complex survival coefficients into actionable strategic advice.

---

## 🛠️ Data Engineering (The "City Scholar" Standard)

A significant portion of this project focused on the rigorous data engineering standards observed during **TE 440** industry visits to Chicago tech leaders like DRW and Adyen.

### 1. Lifecycle Synthesis (The 2-Year Cycle)
The raw dataset contains fragmented license records. I discovered a **bi-annual license cycle** in Chicago, where businesses typically renew every two years. 
*   **Temporal Aggregation**: I developed a pipeline to merge consecutive/overlapping records for the same entity into a single "Business Lifecycle." This prevents the model from misinterpreting a standard renewal as a business closure.

### 2. Socio-Spatial Fusion
*   **Exogenous Factors**: Fused the Chicago Business License dataset with **US Census (ACS) data**.
*   **Feature Selection**: Integrated **Median Rent** (Operational Overhead) and **Median Income** (Purchasing Power) to contextualize raw spatial coordinates.

---

## 🧠 Modeling & Results

I performed a diagnostic comparison between tree-based and regularized linear models:

| Model Type | Features | Train C-index | Test C-index | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| Cox (Baseline) | Spatial Only | 0.5300 | 0.5288 | Near-Random |
| **Cox (Final)** | **Spatial + Industry + Economic** | **0.6009** | **0.6023** | **Optimal** |
| RSF (Initial) | Spatial + Industry | 0.6523 | 0.5377 | Overfitted |

### The "Coordinate Memorization" Trap
The **Random Survival Forest (RSF)** initially seemed promising but suffered from severe overfitting by "memorizing" specific successful or failed coordinates. By selecting a **Regularized Cox Proportional Hazards (CPH) model**, the bot achieves superior generalization across Chicago's diverse grid.

---

## 💻 Tech Stack

*   **Backend**: Python, `lifelines` (Survival Analysis), `pandas`, `pickle`.
*   **Frontend**: React, `Leaflet`/`CARTO` (Geospatial Rendering).
*   **Data Sources**: Chicago Data Portal, US Census Bureau.

---

## 🎓 Academic Context (TE 440)

Developed as a capstone project for the **UIUC Master of Computer Science** program and the **TE 440 (High Tech Venture Management)** City Scholars course. The project applies industry-level risk management principles to urban entrepreneurship, providing a prototype for a PropTech decision-support tool.

---

## ⚠️ Limitations

The model acknowledges a **Predictive Ceiling** (Unobserved Heterogeneity). It cannot account for idiosyncratic factors like personal owner bankruptcy, internal management disputes, or successful businesses that relocate (appearing as "closures" in administrative data).

---


**Developed by Chenxin Yan**  
*Master of Computer Science (MCS) Student*  
*University of Illinois Urbana-Champaign*
