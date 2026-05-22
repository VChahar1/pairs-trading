from src.pairs.data import download_prices, compute_log_returns, check_data_quality

prices = download_prices()
print('Prices shape:', prices.shape)
print('Date range:', prices.index.min(), 'to', prices.index.max())

returns = compute_log_returns(prices)
print('Returns shape:', returns.shape)

quality = check_data_quality(prices)
for k, v in quality.items():
    print(f'  {k}: {v}')