import pandas as pd

from backtester.strategy import Strategy


class IchimokuStrategy(Strategy):
    """Ichimoku Cloud strategy.

    Uses Tenkan-sen/Kijun-sen crossovers with cloud (Kumo) confirmation.
    Long: Tenkan > Kijun AND price above cloud.
    Short: Tenkan < Kijun AND price below cloud.
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(self.tenkan).max() + low.rolling(self.tenkan).min()) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(self.kijun).max() + low.rolling(self.kijun).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(self.kijun)

        # Senkou Span B (Leading Span B)
        senkou_b_val = ((high.rolling(self.senkou_b).max() + low.rolling(self.senkou_b).min()) / 2).shift(self.kijun)

        # Cloud top and bottom
        cloud_top = pd.concat([senkou_a, senkou_b_val], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b_val], axis=1).min(axis=1)

        signals = pd.Series(0, index=df.index, dtype="int8")

        # Long: TK cross up + price above cloud
        long_cond = (tenkan_sen > kijun_sen) & (close > cloud_top)
        short_cond = (tenkan_sen < kijun_sen) & (close < cloud_bottom)

        signals[long_cond] = 1
        signals[short_cond] = -1

        # Forward-fill to hold
        signals = signals.replace(0, pd.NA).ffill().fillna(0).astype("int8")
        return signals

    def param_grid(self) -> dict[str, list]:
        return {
            "tenkan": [7, 9, 12, 15],
            "kijun": [20, 26, 30, 40],
            "senkou_b": [40, 52, 60, 80],
        }
