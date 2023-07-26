# Projekt Inżynierski

# Setup

Install venv `pip install virtualenv`
Create venv in root folder `python3 -m venv .venv`
Activate venv `cd .venv/bin && source activate`
Go back to root directory and to install requirements enter `pip install -r requirements.txt `

# NASDAQ script usage

**python3 data_api.py** `[-h]` `[--start START]` `[--end END]` `[--company COMPANY]` `[--trades]` `[--quotas]` `[--download]` `[--merge]`
            `[--aggregate]` `[--visualize]`

**options:** <br>
  - `-h, --help` <br>
  - `--start START, -s START`  <br>
    * Starting year which data will be worked on.
  - `--end END, -e END`  <br>
    * Ending year which data will be worked on.
  - `--company COMPANY, -c` <br>
    * Company name, ex: *(BP,BAMXF,VLKAF)*
  - `--trades, -t` <br>
    * Choose trades data.
  - `--quotes, -q` <br>
    * Choose quotes data.
  - `--download, -d` <br>
    * Only download.
  - `--merge, -m` <br>
    * Only merge.
  - `--aggregate, -a` <br>
    * Only aggregate.
