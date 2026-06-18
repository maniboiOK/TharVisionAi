# THAR Vision AI

Streamlit dashboard for Rajasthan climate resilience planning.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud deployment

1. Upload this project to GitHub.
2. In Streamlit Cloud, choose `streamlit_app.py` as the app file.
3. Add this secret in Streamlit Cloud settings:

```toml
OPENWEATHER_API_KEY = "your_openweathermap_api_key"
```

4. Deploy the app.

The app also works in manual/demo mode if the weather API key is not configured.
