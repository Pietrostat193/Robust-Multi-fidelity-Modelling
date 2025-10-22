# Figures Folder

This directory contains summary statistics and time series visualizations of PM2.5 air quality data for the year **2021**. The figures and tables here are organized to provide both aggregated summaries and station-specific insights.

---

## 📁 Folder Structure

figures/
│
├── summary_by_station_id.csv
├── summary_by_fidelity.csv
└── TimeSeriesByStation/
├── Station_#1.png
├── Station_#2.png
├── ...
└── Station_#N.png


---

## 📊 Summary Files

### **1. summary_by_station_id.csv**

Contains summary statistics of PM2.5 measurements **grouped by station ID**.

| Column Name | Description |
|--------------|-------------|
| `ID` | Unique station identifier |
| `Count` | Number of observations for the station |
| `Min_PM25` | Minimum recorded PM2.5 concentration |
| `Max_PM25` | Maximum recorded PM2.5 concentration |
| `Mean_PM25` | Mean (average) PM2.5 concentration |
| `SE_PM25` | Standard error of the mean PM2.5 |
| `LOWER_95CI_PM25` | Lower bound of the 95% confidence interval for the mean |
| `UPPER_95CI_PM25` | Upper bound of the 95% confidence interval for the mean |

---

### **2. summary_by_fidelity.csv**

Contains summary statistics of PM2.5 measurements **grouped by data fidelity class** (e.g., low, medium, high).

| Column Name | Description |
|--------------|-------------|
| `ID` | Fidelity category identifier |
| `Count` | Number of observations within the fidelity category |
| `Min_PM25` | Minimum recorded PM2.5 concentration |
| `Max_PM25` | Maximum recorded PM2.5 concentration |
| `Mean_PM25` | Mean (average) PM2.5 concentration |
| `SE_PM25` | Standard error of the mean PM2.5 |
| `LOWER_95CI_PM25` | Lower bound of the 95% confidence interval for the mean |
| `UPPER_95CI_PM25` | Upper bound of the 95% confidence interval for the mean |

---

## 📈 Time Series Visualizations

### **Folder: `TimeSeriesByStation/`**

Contains PNG plots showing **daily (or hourly) PM2.5 concentrations during 2021** for each monitoring station.

Each file is named according to its station identifier:

TimeSeriesByStation/
├── Station_#1.png # Time series of PM2.5 for Station #1
├── Station_#2.png # Time series of PM2.5 for Station #2
└── ...

Each plot includes:
- X-axis: Date (from 2021-01-01 to 2021-12-31)
- Y-axis: PM2.5 concentration (µg/m³)
- Title: Station ID and type of sensor





