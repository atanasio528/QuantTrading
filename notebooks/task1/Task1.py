import csp
from csp import ts
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

class TwoWayPrice(csp.Struct):  # Quote 대체
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float

class Order(csp.Struct):
    symbol: str
    order_side: str  # BUY / SELL
    price: float
    qty: float
    order_type: str  # market, limit
    market: str

class Trade(csp.Struct):
    position: str
    price: float
    qty: float
    latency: float

@csp.node
def my_market_simulator(mu: float = 0, sigma: float = 0.2, sigma_spread: float = 0.1, model: str = "GBM") -> ts[TwoWayPrice]:
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        s_bid = 100.01
        s_ask = 99.99
        s_mid = (s_bid + s_ask) / 2
        s_bid_qty = 10
        s_ask_qty = 10
        mu_second = mu / (252 * 24 * 60 * 60)
        sigma_second = sigma / np.sqrt(252 * 24 * 60 * 60)
        sigma_spread_second = sigma_spread / np.sqrt(252 * 24 * 60 * 60)
        dt = 1

    with csp.start():
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)

    if csp.ticked(alarm):
        csp.schedule_alarm(alarm, timedelta(seconds=1), True)

        if model == "GBM":
            s_mid *= np.exp((mu_second - 0.5 * sigma_second ** 2) * dt + sigma_second * np.sqrt(dt) * np.random.randn())
            spread = min(abs(3 * sigma_spread_second * np.random.randn() * s_mid), 3 * sigma_spread_second * s_mid)
            s_bid = s_mid + 0.5 * spread
            s_ask = s_mid - 0.5 * spread
            s_bid_qty = np.random.randint(1, 100)
            s_ask_qty = np.random.randint(1, 100)
            market = TwoWayPrice(bid=s_bid, ask=s_ask, bid_qty=s_bid_qty, ask_qty=s_ask_qty)
            return market


@csp.node
def my_weighted_mid(market: ts[TwoWayPrice]) -> ts[float]:
    w_bid = market.bid_qty / (market.bid_qty + market.ask_qty)
    w_ask = market.ask_qty / (market.bid_qty + market.ask_qty)
    return w_bid * market.bid + w_ask * market.ask


@csp.node
def my_moving_average(counting_policy: str, window_size: int, mid_prices: ts[float]) -> ts[float]:
    with csp.start():
        assert window_size > 0
        csp.set_buffering_policy(mid_prices, tick_history=timedelta(seconds=window_size))

    if counting_policy == "Seconds":
        if csp.ticked(mid_prices):
            buffer = csp.values_at(mid_prices,
                                   timedelta(seconds=-window_size),
                                   timedelta(seconds=0),
                                   csp.TimeIndexPolicy.INCLUSIVE,
                                   csp.TimeIndexPolicy.INCLUSIVE)
            if len(buffer) > 0:
                return np.mean(buffer)
    elif counting_policy == "Ticks":
        raise NotImplementedError("Tick-based moving average not implemented yet.")


@csp.node
def golden_cross(market: ts[TwoWayPrice], ma5: ts[float], ma10: ts[float]) -> ts[Order]:
    with csp.state():
        s_position_opened = False
        s_sell_trigerred = False
        s_remainings = 0.0

    if csp.ticked(market, ma5, ma10) and csp.valid(market, ma5, ma10):
        if ma5 > ma10 and not s_position_opened:
            s_position_opened = True
            s_remainings += market.bid_qty
            return Order(symbol="AAPL", order_side="BUY", price=market.bid, qty=market.bid_qty, order_type="market", market="STOCK")
        if ma5 < ma10 and s_position_opened:
            s_position_opened = False
            s_sell_trigerred = True
            selling_qty = min(s_remainings, market.ask_qty)
            s_remainings -= selling_qty
            return Order(symbol="AAPL", order_side="SELL", price=market.ask, qty=selling_qty, order_type="market", market="STOCK")

    if csp.ticked(market) and s_sell_trigerred:
        if s_remainings == 0:
            s_sell_trigerred = False
        if s_remainings > 0:
            selling_qty = min(s_remainings, market.ask_qty)
            s_remainings -= selling_qty
            return Order(symbol="AAPL", order_side="SELL", price=market.ask, qty=selling_qty, order_type="market", market="STOCK")


@csp.node
def gateway(orders: ts[Order]) -> ts[Trade]:
    if csp.ticked(orders):
        return Trade(position=orders.order_side, price=orders.price, qty=orders.qty, latency=0.01)


@csp.node
def calc_pnl(mid: ts[float], trades: ts[Trade]) -> ts[float]:
    with csp.state():
        s_position_qty = 0.0
        s_pnl = 0.0
        s_buying_price = 0.0

    if csp.ticked(trades):
        if trades.position == "BUY":
            s_position_qty += trades.qty
            s_buying_price = trades.price
        else:
            s_position_qty -= trades.qty

    if csp.ticked(mid):
        s_pnl += (mid - s_buying_price) * s_position_qty
        return s_pnl


@csp.graph
def my_graph():
    market = my_market_simulator()
    mid = my_weighted_mid(market)
    ma5 = my_moving_average(counting_policy="Seconds", window_size=5, mid_prices=mid)
    ma10 = my_moving_average(counting_policy="Seconds", window_size=10, mid_prices=mid)
    signals = golden_cross(market, ma5, ma10)
    trades = gateway(signals)
    pnl = calc_pnl(mid, trades)

    csp.add_graph_output("market", market)
    csp.add_graph_output("ma5", ma5)
    csp.add_graph_output("ma10", ma10)
    csp.add_graph_output("signals", signals)
    csp.add_graph_output("pnl", pnl)


def main():
    np.random.seed(0)
    start = datetime(2025, 1, 1, 9, 30, 0)
    end = datetime(2025, 1, 1, 9, 35, 0)
    results = csp.run(my_graph, starttime=start, endtime=end, realtime=False)

    df = pd.DataFrame()
    for i in range(len(results['market'])):
        idx = results['market'][i][0]
        quotes = results['market'][i][1]
        ma5 = results['ma5'][i][1]
        ma10 = results['ma10'][i][1]
        pnl = results['pnl'][i][1]
        df.loc[idx, 'bid'] = quotes.bid
        df.loc[idx, 'ask'] = quotes.ask
        df.loc[idx, 'ma5'] = ma5
        df.loc[idx, 'ma10'] = ma10
        df.loc[idx, 'pnl'] = pnl

    for i in range(len(results['signals'])):
        idx = results['signals'][i][0]
        order = results['signals'][i][1]
        df.loc[idx, 'signals'] = order.order_side

    # Plot 1: bid/ask
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['bid'], label='Bid')
    plt.plot(df.index, df['ask'], label='Ask')
    plt.title('Bid and Ask Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/bid_ask_prices.png")  # 🔽 파일로 저장
    plt.close()

    # Plot 2: Moving averages + triangle signals
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['ma5'], label='MA5')
    plt.plot(df.index, df['ma10'], label='MA10')

    buy_signals = df[df['signals'] == 'BUY']
    sell_signals = df[df['signals'] == 'SELL']
    plt.scatter(buy_signals.index, df.loc[buy_signals.index, 'ma5'], marker='^', color='green', label='BUY', s=100)
    plt.scatter(sell_signals.index, df.loc[sell_signals.index, 'ma10'], marker='v', color='red', label='SELL', s=100)

    plt.title('Golden Cross Strategy with BUY/SELL Signals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/golden_cross_signals.png")  # 🔽 파일로 저장
    plt.close()


def main2():
    np.random.seed(2025)
    start = datetime(2025, 1, 1, 9, 30, 0)
    end = datetime(2025, 1, 1, 16, 00, 0)  # market closed
    results = csp.run(my_graph, starttime=start, endtime=end, realtime=False)

    df = pd.DataFrame()
    for i in range(len(results['market'])):
        idx = results['market'][i][0]
        quotes = results['market'][i][1]
        ma5 = results['ma5'][i][1]
        ma10 = results['ma10'][i][1]
        pnl = results['pnl'][i][1]
        df.loc[idx, 'bid'] = quotes.bid
        df.loc[idx, 'ask'] = quotes.ask
        df.loc[idx, 'ma5'] = ma5
        df.loc[idx, 'ma10'] = ma10
        df.loc[idx, 'pnl'] = pnl

    for i in range(len(results['signals'])):
        idx = results['signals'][i][0]
        order = results['signals'][i][1]
        df.loc[idx, 'signals'] = order.order_side

    # Plot 4: PnL
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['pnl'], label='PnL', color='purple')
    plt.title('PnL Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/pnl.png")  # 🔽 파일로 저장
    plt.close()

if __name__ == "__main__":
    main()
    main2()
