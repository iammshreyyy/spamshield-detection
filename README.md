# SpamShield — Email & SMS Spam Detection

Starter project: scikit-learn text classifier + Flask API + static frontend deployable to Netlify.

## Architecture

- `backend/train_model.py`: trains a TF-IDF + Logistic Regression classifier.
- `backend/app.py`: exposes `POST /api/predict` and `GET /api/health`.
- `frontend/index.html`: user interface.
- Netlify hosts the static frontend. The Python/Flask API should run on a Python-capable host (the included `render.yaml` is a starting point). Netlify Functions are not a native Python runtime, so don't try to deploy this Flask app as a Netlify Function.

## 1. Prepare data and train

Use a properly licensed, labeled SMS/email dataset. Make a CSV at `backend/spam.csv` with exactly these columns:

```csv
text,label
"Congratulations, you won a prize. Claim now","spam"
"Hi, are we still meeting at 5?","ham"
```

These two rows only show the format; they are not enough to train. Use a substantial labeled dataset and check its license. If combining email and SMS datasets, normalize their labels to `spam` and `ham`.

In a terminal:

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

The model file `spam_model.joblib` is created locally. For deployment, make sure it is available to the API service. For a simple demo, commit the trained artifact only if your repository policy allows it; for production, use a controlled artifact store or build step.

## 2. Run the frontend

Open `frontend/index.html` in a browser. By default it calls `http://127.0.0.1:5000`. For a deployed API, change `API_BASE` in the script to the HTTPS URL of your backend. Set `FRONTEND_ORIGIN` on the backend to your exact Netlify site origin, then tighten CORS to that origin.

## 3. Deploy

1. Push this folder to a Git repository.
2. Deploy the repository to Netlify with publish directory `frontend`. The included `netlify.toml` sets this.
3. Deploy the Python API separately using `render.yaml` or another Python host.
4. Set the frontend API base URL to the deployed API URL and configure the backend's `FRONTEND_ORIGIN`.
5. Test `/api/health`, then classify a message in the UI.

## 4. Real-time email integration is a separate step

This starter does **not** connect to or modify a real mailbox. For Gmail, implement Google OAuth 2.0 and use the Gmail API. For near-real-time notifications, Gmail's `users.watch` uses Google Cloud Pub/Sub and needs renewal at least every seven days (daily renewal is recommended). A backend then fetches changed messages and sends subject/body to `/api/predict`.

Start with read-only access, explicit user consent, and minimal scopes. Never ask users to paste their Gmail password or store OAuth tokens in the frontend. Do not automatically delete or move messages based only on an ML prediction; display a review queue first.

For real inbound SMS, a provider such as Twilio can call a public HTTPS webhook. Validate the provider's webhook signature before trusting incoming requests. A normal web form only classifies pasted SMS; it cannot read the user's phone SMS inbox.

## Limitations / production checklist

- The model is a baseline, not a guaranteed phishing detector. Train/test with representative, separate data and report precision, recall, F1 and confusion matrix.
- `confidence` is the model's predicted class probability, not a guarantee. Calibrate it if it will be shown to users.
- Email header authentication, sender reputation, malicious URLs, attachments and spoofing require additional analysis beyond text classification.
- Add authentication, rate limits, request-size limits, logs that avoid storing message contents, and HTTPS before public use.
- Do not put API keys, OAuth secrets or tokens in frontend JavaScript or a public Git repository.
