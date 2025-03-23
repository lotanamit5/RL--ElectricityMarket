import numpy as np


DEFAULT_BATTERY_CAPACITY = 50 # kWh, Data taken from https://kedmasolar.com/storage/

HOURS_A_YEAR = 8760

# std calculated from _demand_default_fn data
DEMAND_STD = 0.15
PRICE_STD = 0.31

@staticmethod
def _days_in_month(month):
    assert 1 <= month <= 12
    _days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return _days_per_month[month - 1]

@staticmethod
def _timestamp_to_date(t):
    """
    Convert an environment timestep (integer) to hour, day, and month.
    Assumes a 'regular' year starting on January 1st at 00:00.
    Parameters:
    - t (int): timestep.

    Outputs:
    - hour (int): Hour of the day, [0-23].
    - day (int): Day of the month, [1-31].
    - month (int): Month of the year, [1-12].
    """
    
    hour = t % 24
    total_days = t // 24
    
    day = total_days + 1  # Days are 1-based
    month = 1
    for month in range(1, 13):
        days = _days_in_month(month)
        if day > days:
            day -= days
        else:
            break
    
    return hour, day, month

@staticmethod
def demand_default_fn(t):
    return _demand_default_fn(*_timestamp_to_date(t))

@staticmethod
def _demand_default_fn(hour, day, month):
    # Data taken from https://fs.knesset.gov.il/globaldocs/MMM/5f43840b-bf74-ed11-8150-005056aac6c3/2_5f43840b-bf74-ed11-8150-005056aac6c3_11_19876.pdf
    # And scaled to my personal average monthly consumption, averaged over 2023-2024.
    assert 0 <= hour <= 23
    assert 1 <= day <= 31
    assert 1 <= month <= 12
    
    _IL_avg_monthly_demand_per_month = [
        291.8669571 , 241.28077231, 244.95976757, 198.05257803,
        210.77576997, 249.09863723, 353.64341914, 373.41801864,
        305.81648079, 219.20680076, 184.86951169, 267.34032205
    ]
    day_average = _IL_avg_monthly_demand_per_month[month - 1] / _days_in_month(month)
    demand_coeff = 2 if 17 <= hour <= 22 else 1 # sum to 29
    demand = (demand_coeff * day_average) / 29
    return demand

@staticmethod
def price_default_fn(t):
    return _price_default_fn(*_timestamp_to_date(t))

@staticmethod
def _price_default_fn(hour, day, month):
    # Data taken from- https://kedmasolar.com/storage/
    _IL_avg_prices_per_month = {
        # prices are in ILS / kWh
        # 17:00-22:00
        'high': [.981, .981, .3917, .3917, .3917, 1.4131,  1.4131,  1.4131,  1.4131, .3917, .3917, .981],
        # 00:00-17:00, 22:00-24:00
        'low': [.3576, .3576, .3491, .3491, .3491, .4115,  .4115,  .4115,  .4115, .3491, .3491, .3576],
    }
    
    high_low = 'high' if 17 <= hour <= 22 else 'low'
    price = _IL_avg_prices_per_month[high_low][month - 1]
    return price

if __name__ == "__main__":
    demands = [demand_default_fn(t) for t in range(HOURS_A_YEAR)]
    prices = [price_default_fn(t) for t in range(HOURS_A_YEAR)]
    print(f"Demand stats: {np.mean(demands):.2f} +/- {np.std(demands):.2f}")
    print(f"Price stats: {np.mean(prices):.2f} +/- {np.std(prices):.2f}")