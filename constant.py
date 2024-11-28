import pandas as pd

# Fetch the S&P 500 company list from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
sp500_companies = tables[0]

# Display the ticker symbols
sp500_ticker_symbol = sp500_companies['Symbol'].tolist()




