import operator

def read_stock_prices(records):
    with open('data/stock_prices.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]),
                      float(fields[4]))
            records.append(record)


def read_market_analysis(records):
    with open('data/market_analysis.txt') as f:
        for line in f:
            fields = line.split()
            record = (fields[0],
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]))
            records.append(record)


def read_market_segments(records):
    with open('data/market_segments.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      fields[1])
            records.append(record)


def read_info(records):
    with open('data/info.txt') as f:
        for line in f:
            fields = line.split()
            record = (int(fields[0]),
                      int(fields[1]),
                      int(fields[2]),
                      int(fields[3]),
                      int(fields[4]),
                      int(fields[5]),
                      int(fields[6]),
                      float(fields[7]),
                      float(fields[8]),
                      float(fields[9]),
                      int(fields[10]))
            records.append(record)

'''
the find function take as input 3 parameters:
  -) :data: it refers to market_analysis list (defined at row 75)
  -) :key_func: it refers to key_market_analysis, that is the sorting order of 
                market_analysis list (row 80)
  -) :key: key tuple defined at row 134.
'''
def find(data, key_func, key):
    for i in range(len(data)):
        if key_func(data[i]) == key:
            return i
    raise ValueError


def simulate(start_year, start_day, pd):

    # read elements from stock_prices.txt file
    stock_prices = []
    read_stock_prices(stock_prices)
    # order element just extracted per year(col 1), day(col 2), 
    # and finally Company(col 0)
    stock_prices.sort(key=operator.itemgetter(1, 2, 0))

    # read elements from market_analysis.txt file
    market_analysis = []
    read_market_analysis(market_analysis)
    # order element just extracted per year(col 1), quarter(col 2), 
    # and finally segment(col 0)
    key_market_analysis = operator.itemgetter(1, 2, 0)
    market_analysis.sort(key=key_market_analysis)

    # read elements from market_segments.txt file
    market_segments = []
    read_market_segments(market_segments)
    # order element just extracted per company(col 0)
    market_segments.sort(key=operator.itemgetter(0))

    # read elements from info.txt file
    info_daily = []
    read_info(info_daily)
    # order element just extracted per year(col 1), day(col 2), company(col 0)
    info_daily.sort(key=operator.itemgetter(1, 2, 0))

    # unique companes and sectors
    companies, sectors = set(), set()
    for s in market_segments:
        companies.add(s[0])
        sectors.add(s[1])

    # find the total number of companies (3 in this case)
    num_companies = len(companies)
    
    total, correct = 0, 0
    previous_stock_prices = None
    # loop for all daily info, 3 by 3
    for i in range(0, len(info_daily), num_companies):

        # list of tuples, composed by 3 tuples. 
        # Each tuple is one field of info_daily.
        daily = info_daily[i:i+num_companies]

        # initialization and filling list of stock prices for all the stock 
        # prices provided from the stock_prices.txt file 
        current_stock_prices = []
        for c in range(num_companies):
            current_stock_prices.append(stock_prices[i + c][4])
        
        # check if the first (out of 3) tuple's year is >= the start year given 
        # as parameter, and if the first (out of 3) tuple's day is >= the start 
        # day given as parameter
        if daily[0][1] >= start_year and daily[0][2] >= start_day:
            quarterly = []
            # for loop between two values: "IT" and "BIO"
            for s in sectors:

                # key is a tuple of 3 elements:
                #   -) year of the first (out of 3) tuple that is looping (row 102)
                #   -) quarter of the first (out of 3) tuple that is looping (row 102)
                #   -) sector s which is looping (row 120)
                key = (daily[0][1], daily[0][3], s)

                idx = find(market_analysis, key_market_analysis, key)
                quarterly.append(market_analysis[idx])
            if previous_stock_prices is not None:
                increased = []
                for c in range(num_companies):
                    increased.append(current_stock_prices[c] > previous_stock_prices[c])
                y = pd.predict(market_segments, quarterly, daily, current_stock_prices)
                print(market_segments, quarterly, daily, current_stock_prices)
                print("Predictions (year, day):", daily[0][1], daily[0][2], y, "Target:", increased)
                for c in range(num_companies):
                    if y[c] == increased[c]:
                        correct += 1
                    total += 1
        previous_stock_prices = current_stock_prices
    print("Accuracy(%) = ", 100*correct / total)
