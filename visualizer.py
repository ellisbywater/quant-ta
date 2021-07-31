import mplfinance as mpf


def squeeze_plot(df):
    df = df[-100:]
    ohlc = df[['open', 'high', 'close', 'low']]

    # add colors for the value bar
    colors = []
    for ind, val in enumerate(df['value']):
        if val >= 0:
            color = 'green'
            if val > df['value'][ind - 1]:
                color = 'lime'
        else:
            color = 'maroon'
            if val < df['value'][ind - 1]:
                color = 'red'
        colors.append(color)

    # add 2 subplots: 1.bars, 2.crosses
    apds = [mpf.make_addplot(df['value'], panel=1, type='bar', color=colors, alpha=0.8, secondary_y=False),
            mpf.make_addplot([0] * len(df), panel=1, type='scatter', marker='x', markersize=50,
                             color=['gray' if s else 'black' for s in df['sqeeze_off']], secondary_y=False)]

    # plot subplots
    fig, axes = mpf.plot(ohlc, volume_panel=2, figratio=(2, 1), figscale=1, type='candle', addplot=apds, returnfig=True)
