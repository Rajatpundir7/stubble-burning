# Punjab Stubble Burning Analysis (2018-2021)

This repository contains data and code for analyzing stubble burning events in Punjab, India, from 2018 to 2021 using satellite sensor data.

## Project Structure

* **`stubble_burning_analysis.ipynb`**: The main Jupyter Notebook containing the interactive analysis, visual storytelling, and detailed commentary.
* **`run_analysis.py`**: A standalone Python script that executes the complete 5-stage analysis and outputs the plots directly to your workspace.
* **`Punjab Stubble data 2018-21_Clean and Processed - Raw Data_18.csv`**: The clean and processed dataset containing stubble burning coordinates, fire power, detection dates, times, and satellite info.
* **Plots (`*.png`)**: Visualizations saved from the analysis:
  * `01_missing_value_heatmap.png`: Heatmap of missing values across fields.
  * `02_fire_power_distribution.png`: Fire power distribution per satellite sensor.
  * `03_temporal_analysis.png`: Yearly event trends, monthly seasonality, and weekday/weekend distributions.
  * `04_geographic_analysis.png`: Geospatial distribution of events by district and block.
  * `04b_spatial_map.png`: Scatter plot mapping latitude and longitude of fires colored by intensity.
  * `05_intensity_storytelling.png`: Combined panel plot summarizing the intensity metrics, density curve shifts, and year-over-year fire power distribution.

## Requirements

Ensure you have the required packages installed:
```bash
pip install pandas numpy matplotlib seaborn
```

## How to Run

### Option 1: Standalone Script
Run the script to perform the data profiling, cleaning, analysis, and plot output:
```bash
python run_analysis.py
```

### Option 2: Jupyter Notebook
Open the notebook in your environment and run all cells:
```bash
jupyter notebook stubble_burning_analysis.ipynb
```
