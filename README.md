# AI-Based Early Warning and Landslide Risk Monitoring System

A Flask web application that predicts landslide risk from seven sensor and terrain readings using a Random Forest model. It classifies risk as Low, Medium, High or Critical, plays an alarm, sends a Telegram alert and keeps a history of every prediction.

## Features

- Random Forest landslide probability prediction
- Four risk levels: Low, Medium, High, Critical
- Animated web interface with four pages: Home, About, Check risk (input) and Dashboard
- Soft beep for Low and Medium risk, loud siren for High and Critical risk
- Telegram Bot alerts
- Optional site photo upload (saved as a record only)
- Dashboard with model metrics, risk distribution, probability trend, feature importance and prediction history
- CSV log download

## Project structure

```
landslide_flask/
├── app.py                                            Flask backend
├── Landslide_Risk_Optimized_RandomForest_Model.pkl   Trained model (add this yourself)
├── requirements.txt                                  Python packages
├── .env                                              Telegram keys (create this yourself)
├── landslide_prediction_log.csv                      Created after the first prediction
├── uploaded_site_images/                             Created automatically
├── static/
│   └── style.css                                     Design and animations
└── templates/
    ├── base.html                                     Shared navigation bar and footer
    ├── index.html                                    Home page
    ├── about.html                                    About page
    ├── predict.html                                  Input page
    └── dashboard.html                                Dashboard page
```

The folders must be named `templates` and `static`, because Flask looks for those names.

## Requirements

- Python 3.9 or newer
- The trained model file `Landslide_Risk_Optimized_RandomForest_Model.pkl` in the same folder as `app.py`

Create a file named `requirements.txt` with this content:

```
flask
joblib
numpy
pandas
scikit-learn
requests
python-dotenv
```

## Installation

1. Open a terminal in the `landslide_flask` folder.
2. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # Mac / Linux
   ```
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```

The scikit-learn version you install should match the version used to train the model, or joblib may show a warning or fail to load the file.

## Telegram setup

1. In Telegram, open **@BotFather**, send `/newbot` and copy the bot token.
2. Open your new bot, press **Start** and send any message.
3. Open `https://api.telegram.org/botYOUR_TOKEN/getUpdates` in a browser and copy the number after `"chat":{"id":`. Group IDs start with a minus sign.
4. Create a file named `.env` next to `app.py`:
   ```
   TELEGRAM_BOT_TOKEN=123456789:ABCdefYourRealToken
   TELEGRAM_CHAT_ID=123456789
   ```
5. Make sure `app.py` has these two lines just below the Flask import:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

Never share your token or upload the `.env` file to GitHub. If you use Git, add `.env` to `.gitignore`.

## Run the app

```
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## How to use

1. **Home** shows an overview of the system.
2. **About** explains how the system works.
3. **Check risk**: enter the seven readings, optionally upload a site photo, choose whether to send a Telegram alert and play the alarm, then press **Detect landslide risk**.
4. **Dashboard** shows model performance, charts and the full prediction history, with a CSV download button.

## Model inputs and output

| Input | Unit |
|---|---|
| Rainfall intensity | mm |
| Soil moisture | % |
| Slope angle | degrees |
| Ground movement | mm/day |
| Temperature | °C |
| Groundwater level | m |
| Historical landslide | Yes or No |

The output is a landslide probability (0 to 100%) and a risk level.

Default risk thresholds (a model file can override them):

| Probability | Risk level |
|---|---|
| Below 20% | Low |
| 20% to 45% | Medium |
| 45% to 70% | High |
| 70% and above | Critical |

The site photo is stored as a record. The model predicts from the seven numeric readings only, not from the image.

## Routes

| URL | Method | Purpose |
|---|---|---|
| `/` | GET | Home page |
| `/about` | GET | About page |
| `/predict` | GET | Input page |
| `/dashboard` | GET | Dashboard page |
| `/api/predict` | POST | Returns probability and risk level |
| `/api/dashboard` | GET | Metrics, counts and log data for the dashboard |
| `/alarm/soft.wav`, `/alarm/loud.wav` | GET | Alarm sounds |
| `/download/log` | GET | Downloads the CSV log |

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the model | Put the `.pkl` file in the same folder as `app.py` and check the file name |
| `Telegram not configured` | Check the `.env` file, or that the variables are set in the same terminal, then restart the app |
| `HTTP 401` from Telegram | The bot token is wrong. Copy it again from BotFather |
| `chat not found` | The chat ID is wrong, or you have not sent a message to the bot yet |
| No alarm sound | Browsers may block audio until you interact with the page. Press the button again and check that the volume is on |
| Charts do not load | The dashboard loads Chart.js from the internet, so check your connection |

## Technology

Python, Flask, scikit-learn (Random Forest), pandas, NumPy, Requests, Telegram Bot API, Chart.js, HTML, CSS and JavaScript.

## Limitations

- The model is only as good as its training data and should be validated with real local data before it is used for real decisions.
- This system supports decision making. It does not replace expert geotechnical assessment or official government warnings.
- The Flask development server (`debug=True`) is for local use only. Use a production server such as Gunicorn or Waitress if you deploy it.
