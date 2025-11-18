# mlfs-book
O'Reilly book - Building Machine Learning Systems with a feature store: batch, real-time, and LLMs


## Lab Description

This lab implements a parameterized multi-sensor air-quality forecasting pipeline using the Hopsworks Feature Store. The system ingests air-quality and weather data daily, trains per-sensor models, produces 7-day forecasts, and publishes dashboards through GitHub Pages.

### Data and Feature Engineering

- Historical weather and air-quality data are processed in `1_air_quality_feature_backfill.ipynb`, which prepares backfill features and per-sensor configuration.
- Daily ingestion in `2_air_quality_feature_pipeline.ipynb` updates two Feature Groups in Hopsworks:
  - **air_quality** (including lag features pm25_lag_1, pm25_lag_2, pm25_lag_3)
  - **weather**
- Predictions from batch inference are stored in the `aq_predictions` Feature Group and later used for hindcast evaluation.

### Modeling

- `3_air_quality_training_pipeline.ipynb` trains a dedicated XGBoost model for each sensor based on a Feature View that joins weather data with the PM2.5 lagged values.
- A weather-only baseline model is also trained for comparison. We report MSE and R² scores and register each per-sensor model in the Hopsworks Model Registry.

### Inference and Dashboards

- `4_air_quality_batch_inference.ipynb` loads the relevant sensor model and generates a 7-day PM2.5 forecast.
- If lagged PM2.5 features are used, forecasting is done recursively, rolling predictions forward step by step from D+1 to D+7.
- Forecast and hindcast plots are exported as PNG images and automatically published to GitHub Pages. The hindcast view compares predictions with actual observed values when available.

### Orchestration

- `.github/workflows/air-quality-daily.yml` executes the full pipeline every day using a GitHub Actions job matrix with `SENSOR_SLUG` for per-sensor runs.
- All generated artifacts and dashboard images are stored under sensor-specific paths to avoid conflicts.

**Dashboard:**  
https://jeppcode.github.io/mlfs-book/

## ML System Examples


[Link to our dashboard:](https://jeppcode.github.io/mlfs-book/)

# Run Air Quality Tutorial

See [tutorial instructions here](https://docs.google.com/document/d/1YXfM1_rpo1-jM-lYyb1HpbV9EJPN6i1u6h2rhdPduNE/edit?usp=sharing)
    # Create a conda or virtual environment for your project
    conda create -n book 
    conda activate book

    # Install 'uv' and 'invoke'
    pip install invoke dotenv

    # 'invoke install' installs python dependencies using uv and requirements.txt
    invoke install


## PyInvoke

    invoke aq-backfill
    invoke aq-features
    invoke aq-train
    invoke aq-inference
    invoke aq-clean

## Feldera


pip install feldera ipython-secrets
sudo apt-get install python3-secretstorage
sudo apt-get install gnome-keyring 

mkdir -p /tmp/c.app.hopsworks.ai
ln -s  /tmp/c.app.hopsworks.ai ~/hopsworks
docker run -p 8080:8080 \
  -v ~/hopsworks:/tmp/c.app.hopsworks.ai \
  --tty --rm -it ghcr.io/feldera/pipeline-manager:latest


