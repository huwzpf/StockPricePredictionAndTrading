# Imports
import boto3
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Set credentials
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Configuration variables
STARTING_YEAR = '2008'
ENDING_YEAR = '2021'
COMPANY = 'VLKAF' # 'BAMXF' 'BP'

# # Create list of years which data will be downloaded for
YEARS = []
YEAR = int(STARTING_YEAR)
END_YEAR = int(ENDING_YEAR)
while (YEAR <= END_YEAR):
    YEARS.append('{0}'.format(YEAR))
    YEAR+=1
print(YEARS)

# Create appropariate folder structure
os.mkdir('datasets/{0}'.format(COMPANY))
for YEAR in YEARS:
    os.mkdir('datasets/{0}/{0}{1}'.format(COMPANY, YEAR))

# Set up AWS connection and create bucket
s3 = boto3.resource('s3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

bucket = s3.Bucket('ncpgisahub-pro-dod-etl')

# Download data for chosen company
for year in YEARS:
    print('Downloading trades for year {0}...'.format(year))
    
    for obj in bucket.objects.filter(Prefix='trades/symbol={0}/date={1}'.format(COMPANY, year)):
        path, filename = os.path.split(obj.key)
        destination_folder = os.path.abspath('datasets/{0}/{1}{2}{3}'.format(COMPANY,COMPANY, year, os.sep))
        bucket.download_file(obj.key, os.path.join(destination_folder, filename))
        print(os.path.join(destination_folder, filename))
        print('Trades: {0}:{1}'.format(bucket.name, obj.key))

    print('\n')
    print('---------------------------------------------------------------------------------------------')
    print('All trades for year {0} downloaded!!'.format(year))
    print('---------------------------------------------------------------------------------------------')
    print('\n')

    # TODO: Download quotes

# Merge data
for year in YEARS:
    first = True
    counter = 0
    global df
    for file in os.listdir('datasets/{0}/{0}{1}'.format(COMPANY, year)):     
        if file.endswith('.parquet'):
            filename = file
            source = pd.read_parquet('datasets/{0}/{0}{1}/'.format(COMPANY, year) + filename)
            if first:
                df = source
                df.to_csv('datasets/{0}/{0}{1}.csv'.format(COMPANY, year))
                print('First parquet file from year {0} converted to csv!'.format(year))
                counter += 1
                first = False
            else:
                df = source
                df.to_csv('datasets/{0}/{0}{1}.csv'.format(COMPANY, year), mode='a', header=False)
                counter += 1 
                if ((counter % 100) == 0):
                    print('-------------------------------------------------')
                    print('Counter: ' + str(counter))
                    print('Next 100 parquet files appended to csv file!')  
                    print('-------------------------------------------------')

# TODO: Merge quotes

# Aggregate data
for year in YEARS: 
    df = pd.read_csv('datasets/{0}/{0}{1}.csv'.format(COMPANY, year), index_col='timestamp', usecols=['timestamp', 'price'], dtype={"price": "float64"})
    df.index = df.index.sort_values()
    df.index = pd.to_datetime(df.index)
    df = df.groupby([df.index.year.values, df.index.month.values]).apply(pd.Series.tail,1)
    df.to_csv('datasets/{0}/{0}{1}aggregated.csv'.format(COMPANY, year))

# TODO: Aggregate quotes

# Load aggregated data
HEADER = ['year', 'month', 'timestamp', 'price']

first = True
for year in YEARS:
    df = pd.read_csv('datasets/{0}/{0}{1}aggregated.csv'.format(COMPANY, year))
    if first: 
        df.to_csv('datasets/{0}/{0}monthly.csv'.format(COMPANY), header=HEADER, index = False)
        first = False
    else:
        df.to_csv('datasets/{0}/{0}monthly.csv'.format(COMPANY), mode='a', header=False, index = False)

# TODO: Load aggregated quotes

# # Create charts

df = pd.read_csv('datasets/{0}/{0}monthly.csv'.format(COMPANY))
df.set_index('timestamp', inplace=True)
df.index = pd.to_datetime(df.index)

# Visualize

plt.figure(figsize=(14, 7))
plt.plot_date(df.index, df['price'], linewidth=3, linestyle='solid')
plt.title('{0} stock monthly closing prices {1} - {2}'.format(COMPANY, STARTING_YEAR, ENDING_YEAR))
plt.xlabel('Date')
plt.ylabel('Monthly closing price values')
plt.legend(['{0}'.format(COMPANY)])
plt.savefig('datasets/{0}/{0}_monthly_closing_prices.png'.format(COMPANY));
plt.show()
