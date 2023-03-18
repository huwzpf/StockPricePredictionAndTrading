import requests
import pandas as pd


def download_csv(url: str, delimiter: str = ',') -> pd.DataFrame:
    assert url.endswith('.csv')
    with requests.Session() as s:
        download = s.get(url)

    decoded_content = download.content.decode('utf-8')
    newline = '\r\n' if '\r' in decoded_content else '\n'
    content_split = [row.split(delimiter) for row in decoded_content.split(newline)]
    columns = content_split[0]
    data = content_split[1:]

    return pd.DataFrame(data, columns=columns)
