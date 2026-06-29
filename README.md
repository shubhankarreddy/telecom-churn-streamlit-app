# Telecom Churn Streamlit App

Chatbot-style SQL analytics + churn prediction in one Streamlit app.

## Features

- Ask business questions in natural language (or SQL).
- Run read-only SQL analytics on telecom churn data.
- View chart outputs and download visualizations.
- Run single-customer churn prediction.
- Batch-score analytics output when model features are present.
- Robust dataset loading for local and Streamlit Cloud.

## Project Structure

- `app.py` - main Streamlit app
- `requirements.txt` - Python dependencies
- `telecom_churn_clean.csv` - dataset (recommended location)
- `frontend/` - HTML/CSS UI prototype

## Dataset Loading Order

The app tries these paths in order:

1. `telecom_churn_clean.csv` (same folder as `app.py`)
2. `data/telecom_churn_clean.csv`
3. `data/clean/telecom_churn_clean.csv`
4. `../telecom_churn_clean.csv`

If not found, the app shows an error and allows manual upload using a file uploader.

## Local Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

3. Open the local URL shown in terminal.

## Streamlit Cloud Deploy

1. Push repository to GitHub.
2. In Streamlit Community Cloud, create a new app:
   - Repository: `shubhankarreddy/telecom-churn-streamlit-app`
   - Branch: `master`
   - Main file path: `app.py`
3. Deploy.

## Redeploy / Reboot

When code or data changes:

1. Push latest commit to GitHub.
2. Open Streamlit Cloud app dashboard.
3. Click **Reboot app** (or **Restart** / **Deploy latest**).
4. Check logs to confirm dataset load path and successful startup.

## Notes

- Keep `telecom_churn_clean.csv` in the project root for the most reliable cloud behavior.
- SQL execution is restricted to read-only queries.
