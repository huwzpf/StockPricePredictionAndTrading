import boto3
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

class NasdaqData:
    def __init__(self, start : str, end : str, company : str):
        load_dotenv()
        self._AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        self._AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        self._STARTING_YEAR = start
        self._ENDING_YEAR = end
        self._COMPANY = company
        self._YEARS = []
        self._TRADES_DF = None
        YEAR = int(self._STARTING_YEAR)
        END_YEAR = int(self._ENDING_YEAR)
        while (YEAR <= END_YEAR):
            self._YEARS.append('{0}'.format(YEAR))
            YEAR+=1
        # Create folder structure
        if not os.path.exists('datasets'):
            os.mkdir('datasets/')
            os.mkdir('datasets/trades/')
            os.mkdir('datasets/quotes/')
        if not os.path.exists('datasets/trades/{0}'.format(self._COMPANY)):
            os.mkdir('datasets/trades/{0}'.format(self._COMPANY))
        if not os.path.exists('datasets/quotes/{0}'.format(self._COMPANY)):
            os.mkdir('datasets/quotes/{0}'.format(self._COMPANY))
        for YEAR in self._YEARS:
            if not os.path.exists('datasets/trades/{0}/{0}{1}'.format(self._COMPANY, YEAR)):
                os.mkdir('datasets/trades/{0}/{0}{1}'.format(self._COMPANY, YEAR))
            if not os.path.exists('datasets/quotes/{0}/{0}{1}'.format(self._COMPANY, YEAR)):
                os.mkdir('datasets/quotes/{0}/{0}{1}'.format(self._COMPANY, YEAR))
        # Connect to AWS
        s3 = boto3.resource('s3',
            aws_access_key_id = self._AWS_ACCESS_KEY_ID,
            aws_secret_access_key = self._AWS_SECRET_ACCESS_KEY)
        
        self._BUCKET = s3.Bucket('ncpgisahub-pro-dod-etl')

        print("Initialization complete!\n")

    def download_trades(self):
        for year in self._YEARS:
            print('Downloading trades for year {0}...'.format(year))
            
            for obj in self._BUCKET.objects.filter(Prefix='trades/symbol={0}/date={1}'.format(self._COMPANY, year)):
                path, filename = os.path.split(obj.key)
                destination_folder = os.path.abspath('datasets/trades/{0}/{0}{1}{2}'.format(self._COMPANY, year, os.sep))
                self._BUCKET.download_file(obj.key, os.path.join(destination_folder, filename))
                print(os.path.join(destination_folder, filename))
                print('Trades: {0}:{1}'.format(self._BUCKET.name, obj.key))

            print('\nAll trades for year {0} downloaded!\n'.format(year))

    def download_quotes(self):
        raise NotImplementedError("Download quotes not implemented!")
        #TODO: Use nasdaq website instead

    def merge_trades(self):
        for year in self._YEARS:
            first = True
            counter = 0
            for file in os.listdir('datasets/trades/{0}/{0}{1}'.format(self._COMPANY, year)):     
                if file.endswith('.parquet'):
                    filename = file
                    source = pd.read_parquet('datasets/trades/{0}/{0}{1}/'.format(self._COMPANY, year) + filename)
                    if first:
                        df = source
                        df.to_csv('datasets/trades/{0}/{0}{1}.csv'.format(self._COMPANY, year))
                        print('First parquet file from year {0} converted to csv!'.format(year))
                        counter += 1
                        first = False
                    else:
                        df = source
                        df.to_csv('datasets/trades/{0}/{0}{1}.csv'.format(self._COMPANY, year), mode='a', header=False)
                        counter += 1 
                        if ((counter % 100) == 0):
                            print('\nCounter: ' + str(counter))
                            print('Next 100 parquet files appended to csv file!\n')  

        #TODO: Drop unnecessary columns, trim data

    def aggregate_trades(self):
        for year in self._YEARS: 
            df = pd.read_csv('datasets/trades/{0}/{0}{1}.csv'.format(self._COMPANY, year), index_col='timestamp', usecols=['timestamp', 'price'], dtype={"price": "float64"})
            df.index = df.index.sort_values()
            df.index = pd.to_datetime(df.index)
            df = df.groupby([df.index.year.values, df.index.month.values]).apply(pd.Series.tail,1)
            df.to_csv('datasets/trades/{0}/{0}{1}aggregated.csv'.format(self._COMPANY, year))

        HEADER = ['year', 'month', 'timestamp', 'price']

        first = True
        for year in self._YEARS:
            df = pd.read_csv('datasets/trades/{0}/{0}{1}aggregated.csv'.format(self._COMPANY, year))
            if first: 
                df.to_csv('datasets/trades/{0}/{0}monthly.csv'.format(self._COMPANY), header=HEADER, index = False)
                first = False
            else:
                df.to_csv('datasets/trades/{0}/{0}monthly.csv'.format(self._COMPANY), mode='a', header=False, index = False)

        self._TRADES_DF = pd.read_csv('datasets/trades/{0}/{0}monthly.csv'.format(self._COMPANY))
        self._TRADES_DF.set_index('timestamp', inplace=True)
        self._TRADES_DF.index = pd.to_datetime(self._TRADES_DF.index)

    def visualize_quotas():
        raise NotImplementedError("Visualization of quotas not implemented!")

if __name__ == "__main__":
    print("Warning: This script should not be run directly.")
    print("Please run `data_api.py` instead.")
