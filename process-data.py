
import requests
import zipfile
import io
from pathlib import Path

def generateApiLink(tickerv : str, date : str) -> str:
    return f"https://historical-data.kucoin.com/data/spot/daily/trades/{tickerv}USDT/{tickerv}USDT-trades-{date}.zip"

tickers = ["BTC","ETH","XMR","XRP","SOL","LTC","ZEC","SUI","HYPE","DN","PEPE","ESIM","BREV","TAO","ADA","DOGE","ZTC","TRX","NEAR","BNB"]
yesterday = "2026-01-"
output_dir = Path("src/main/java/data")
output_dir.mkdir(exist_ok=True)
for i in range(13,10,-1):
    for ticker in tickers:
        url = generateApiLink(ticker, yesterday + str(i))
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            z.extract(csv_name, output_dir)

#Hardcode a list of 20 most traded tickets
#Iterate through the list and autogenerate a backlink for yesterday's trades
#Add to a data folder
#Use unzipping library to unzip the data into csv