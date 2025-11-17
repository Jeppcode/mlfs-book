# Air Quality Dashboard

<link rel="stylesheet" href="./assets/css/air-quality.css">

<div class="aq-header">
  <p>
    This page presents PM2.5 forecasts and hindcasts for five sensors located on Gotland and in Kalmar.
    The models predict air quality based on historical measurements and are updated daily.
  </p>
</div>

<div class="aq-dashboard">
  <div class="aq-card">
    <h3>Gotland: Visby – Östra Tvärgatan</h3>
    <img src="./assets/img/pm25_forecast_visby_ostra_tvargatan.png" alt="Forecast Visby – Östra Tvärgatan"/>
    <img src="./assets/img/pm25_hindcast_1day_visby_ostra_tvargatan.png" alt="Hindcast Visby – Östra Tvärgatan"/>
  </div>
  <div class="aq-card">
    <h3>Gotland: Visby – Österväg 17</h3>
    <img src="./assets/img/pm25_forecast_visby_ostervag_17.png" alt="Forecast Visby – Österväg 17"/>
    <img src="./assets/img/pm25_hindcast_1day_visby_ostervag_17.png" alt="Hindcast Visby – Österväg 17"/>
  </div>
  <div class="aq-card">
    <h3>Gotland: Visby – Brömsebroväg 8</h3>
    <img src="./assets/img/pm25_forecast_visby_bromsebrovag_8.png" alt="Forecast Visby – Brömsebroväg 8"/>
    <img src="./assets/img/pm25_hindcast_1day_visby_bromsebrovag_8.png" alt="Hindcast Visby – Brömsebroväg 8"/>
  </div>
  <div class="aq-card">
    <h3>Gotland: Ljugarn – Storvägen</h3>
    <img src="./assets/img/pm25_forecast_ljugarn_storvagen.png" alt="Forecast Ljugarn – Storvägen"/>
    <img src="./assets/img/pm25_hindcast_1day_ljugarn_storvagen.png" alt="Hindcast Ljugarn – Storvägen"/>
  </div>
    <div class="aq-card">
    <h3>Kalmar: Öland – Norra Långgatan</h3>
    <img src="./assets/img/pm25_forecast_kalmar_borgholm.png" alt="Forecast Kalmar – Norra Långgatan"/>
    <img src="./assets/img/pm25_hindcast_1day_kalmar_borgholm.png" alt="Hindcast Kalmar – Norra Långgatan"/>
    <div class="aq-alert aq-alert-warning">
    <p>
        The sensor for Norra Långgatan, Borgholm (Kalmar County) has not been in use since
        <strong>2025-08-21</strong>. We have therefore only used data up to this date for training.
        The hindcast plot does not work for this station because no new daily outcome data is being published.
    </p>
    <p>
        <em>This air quality monitoring station has been automatically disabled due to abnormal data readings.</em><br/>
        For more information, check the data validation page for this sensor:
        <a href="https://aqicn.org/station/validation/@71104" target="_blank" rel="noopener">aqicn.org/station/validation/@71104</a>
    </p>
    </div>

  </div>
</div>

