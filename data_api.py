import argparse
from data_operations import NasdaqData

def main():

    parser = argparse.ArgumentParser(description="API for NASDAQ data script.")

    parser.add_argument("--start", "-s", help="Starting year which data will be downloaded for.")
    parser.add_argument("--end", "-e", help="Ending year which data will be downloaded for.")
    parser.add_argument("--company", "-c", help="Company name, ex: (BP,BAMXF,VLKAF)")

    parser.add_argument("--trades", "-t", action="store_true", help="Choose trades.")
    parser.add_argument("--quotes", "-q", action="store_true", help="Choose quotes.")

    parser.add_argument("--download", "-d", action="store_true", help="Only download.")
    parser.add_argument("--merge", "-m", action="store_true", help="Only merge.")
    parser.add_argument("--aggregate", "-a", action="store_true", help="Only aggregate.")

    args = parser.parse_args()

    start = 2008
    end = 2021

    if args.company:
        company = args.company
        if args.start:
            start = args.start
        if args.end:
            end = args.end

        nasdaq = NasdaqData(str(start), str(end), str(company))

        if args.download:
            if args.trades:
                nasdaq.download_trades()
                print(type(args.company))
            else:
                nasdaq.download_quotes()

        if args.merge:
            if args.trades:
                nasdaq.merge_trades()
            else:
                print("No merge function for quotes yet!")

        if args.aggregate:
            if args.trades:
                nasdaq.aggregate_trades()
            else:
                print("No aggregate function for quotes yet!")

        if args.visualize:
            print("No visualize function yet!")

if __name__ == "__main__":
    main()
