import pandas as pd

from shelfcast.data import build_panel, clean_transactions, FIRST_WEEK, VALIDATION_START


def raw():
    rows = []
    for i, week in enumerate(pd.date_range(FIRST_WEEK, periods=104, freq="W-MON")):
        for code, units in (("10001", 20), ("10002", 10)):
            rows.append({"Invoice": str(100000 + i), "StockCode": code, "Description": code,
                         "Quantity": units, "InvoiceDate": week, "Price": 2, "Country": "United Kingdom"})
    return pd.DataFrame(rows)


def test_cohort_selection_ignores_validation_and_test_sales():
    r = raw()
    first, _, _ = build_panel(r, top_k=1)
    changed = r.copy()
    changed.loc[(changed.InvoiceDate >= VALIDATION_START) & changed.StockCode.eq("10002"), "Quantity"] = 1000000
    second, _, _ = build_panel(changed, top_k=1)
    assert list(first.sku.unique()) == list(second.sku.unique()) == ["10001"]


def test_filter_accounting_and_no_sales_weeks():
    r = raw()
    bad = r.iloc[:1].copy().assign(Invoice="C99999", Quantity=-20)
    r = pd.concat([r, r.iloc[:1], bad], ignore_index=True)
    clean, audit = clean_transactions(r)
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["cancellation_or_nonpositive_sale_removed"] == 1
    assert len(clean) == 208
    assert audit["raw_rows"] == audit["clean_rows"] + sum(v for k, v in audit.items() if k.endswith("_removed"))
    r = r.loc[~((r.StockCode == "10001") & (r.InvoiceDate == FIRST_WEEK))]
    p, _, _ = build_panel(r, 1)
    assert len(p) == 104 and p.iloc[0].units == 0
