# Projekt Inżynierski

# Setup

Install venv `pip install virtualenv` <br>
Create venv in root folder `python3 -m venv .venv` <br>
Activate venv `cd .venv/bin && source activate` <br>
Go back to root directory and to install requirements enter `pip install -r requirements.txt ` <br>

# Download NASDAQ data
```
n = NasdaqData("2018-01-01","2019-01-01") # Choose time period
n.download_nasdaq_data(10)
n.merge_csv()
```