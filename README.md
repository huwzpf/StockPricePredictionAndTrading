# Projekt Inżynierski

# Setup

To install requirements enter `pip install -r requirements.txt `

# NASDAQ script usage

usage: data_api.py [-h] [--start START] [--end END] [--company COMPANY] [--trades] [--quotas] [--download] [--merge]
                   [--aggregate] [--visualize]

options:
  -h, --help            show this help message and exit
  --start START, -s START
                        Starting year which data will be downloaded for.
  --end END, -e END     Ending year which data will be downloaded for.
  --company COMPANY, -c COMPANY
                        Company name, ex: (BP,BAMXF,VLKAF)
  --trades, -t          Choose trades.
  --quotas, -q          Choose quotas.
  --download, -d        Only download.
  --merge, -m           Only merge.
  --aggregate, -a       Only aggregate.
  --visualize, -v       Only visualize.
