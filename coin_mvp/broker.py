from __future__ import annotations

from .models import Fill, Position, Side, utc_now


class PaperBroker:
    def __init__(self, market: str, starting_cash: float, fee_rate: float, slippage_bps: float) -> None:
        self.market = market
        self.cash = starting_cash
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.position = Position()
        self.realized_pnl = 0.0

    def equity(self, mark_price: float) -> float:
        return self.cash + self.position.qty * mark_price

    def buy(self, price: float, cash_to_use: float, reason: str) -> Fill | None:
        cash_to_use = min(cash_to_use, self.cash)
        if cash_to_use <= 0:
            return None

        fill_price = self._apply_slippage(price, Side.BUY)
        fee = cash_to_use * self.fee_rate
        notional = cash_to_use - fee
        qty = notional / fill_price
        total_cost = notional + fee

        previous_qty = self.position.qty
        new_qty = previous_qty + qty
        if new_qty <= 0:
            return None

        self.position.avg_price = (
            (previous_qty * self.position.avg_price) + (qty * fill_price)
        ) / new_qty
        self.position.qty = new_qty
        self.position.peak_price = max(self.position.peak_price, fill_price)
        self.position.partial_exit_taken = False
        self.cash -= total_cost

        return Fill(
            timestamp=utc_now(),
            market=self.market,
            side=Side.BUY,
            price=fill_price,
            qty=qty,
            fee=fee,
            cash_after=self.cash,
            position_qty_after=self.position.qty,
            realized_pnl=0.0,
            reason=reason,
        )

    def sell_fraction(self, price: float, fraction: float, reason: str) -> Fill | None:
        if not self.position.is_open:
            return None
        fraction = max(0.0, min(1.0, fraction))
        if fraction <= 0:
            return None

        qty = self.position.qty * fraction
        if qty <= 0:
            return None
        fill_price = self._apply_slippage(price, Side.SELL)
        gross = qty * fill_price
        fee = gross * self.fee_rate
        proceeds = gross - fee
        pnl = proceeds - (qty * self.position.avg_price)

        self.cash += proceeds
        self.realized_pnl += pnl
        remaining_qty = max(0.0, self.position.qty - qty)
        position_qty_after = remaining_qty
        if remaining_qty <= 1e-12:
            self.position = Position()
            position_qty_after = 0.0
        else:
            self.position.qty = remaining_qty
            self.position.partial_exit_taken = self.position.partial_exit_taken or fraction < 1.0

        return Fill(
            timestamp=utc_now(),
            market=self.market,
            side=Side.SELL,
            price=fill_price,
            qty=qty,
            fee=fee,
            cash_after=self.cash,
            position_qty_after=position_qty_after,
            realized_pnl=pnl,
            reason=reason,
        )

    def sell_all(self, price: float, reason: str) -> Fill | None:
        return self.sell_fraction(price, 1.0, reason)

    def mark_peak(self, price: float) -> None:
        if not self.position.is_open:
            return
        self.position.peak_price = max(self.position.peak_price, price)

    def _apply_slippage(self, price: float, side: Side) -> float:
        multiplier = self.slippage_bps / 10_000.0
        if side == Side.BUY:
            return price * (1.0 + multiplier)
        return price * (1.0 - multiplier)
