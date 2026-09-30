import math

import src.currency as currency


def test_currency_round_trip_and_formatting():
    rate = 0.004
    assert math.isclose(currency.pkr_to_usd(2500, rate), 10.0)
    assert math.isclose(currency.usd_to_pkr(10.0, rate), 2500.0)
    assert currency.format_usd(2500, rate) == "$10.00"
    assert currency.format_usd(-2500, rate) == "-$10.00"


def test_live_rate_primary_provider(monkeypatch):
    monkeypatch.setattr(
        currency,
        "_fetch_json",
        lambda url, timeout: {"date": "2026-09-30", "base": "PKR", "quote": "USD", "rate": 0.00361},
    )
    fx = currency.fetch_pkr_to_usd_rate()
    assert fx["is_live"] is True
    assert fx["rate"] == 0.00361
    assert "State Bank of Pakistan" in fx["source"]


def test_live_rate_falls_back_when_network_is_unavailable(monkeypatch):
    def fail(url, timeout):
        raise OSError("offline")

    monkeypatch.setattr(currency, "_fetch_json", fail)
    fx = currency.fetch_pkr_to_usd_rate()
    assert fx["is_live"] is False
    assert fx["rate"] == currency.FALLBACK_RATE
    assert fx["source"] == "Packaged fallback rate"
