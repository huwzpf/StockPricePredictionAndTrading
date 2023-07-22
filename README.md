# Projekt Inżynierski

# Setup

To install requirements enter `pip install -r requirements.txt `

# NASDAQ script usage

**python3 data_api.py** `[-h]` `[--start START]` `[--end END]` `[--company COMPANY]` `[--trades]` `[--quotas]` `[--download]` `[--merge]`
            `[--aggregate]` `[--visualize]`

**options:** <br>
  - `-h, --help` <br>
  - `--start START, -s START`  <br>
    * Starting year which data will be downloaded for.
  - `--end END, -e END`  <br>
    * Ending year which data will be downloaded for.
  - `--company COMPANY, -c` <br>
    * Company name, ex: *(BP,BAMXF,VLKAF)*
  - `--trades, -t` <br>
    * Choose trades.
  - `--quotas, -q` <br>
    * Choose quotas.
  - `--download, -d` <br>
    * Only download.
  - `--merge, -m` <br>
    * Only merge.
  - `--aggregate, -a` <br>
    * Only aggregate.
  - `--visualize, -v` <br>
    * Only visualize.
