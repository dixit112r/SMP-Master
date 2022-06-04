import yfinance as yf
import sqlite3 as sq

from SMP.settings import BASE_DIR

def GetDataFromWeb(ticker, table):
    start_date = '2004-08-19'
    end_date = '2020-12-18'

    data = yf.download(ticker, start_date, end_date)

    cols = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    data = data[cols]
    data.columns = data.columns.str.replace(" ", "_")

    # Data to the SQL databse
    sql_data = BASE_DIR / 'db.sqlite3'
    conn = sq.connect(sql_data)
    data.to_sql(table, conn, if_exists='replace')
    conn.close()
