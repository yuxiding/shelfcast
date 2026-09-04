# Data provenance

**Source:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D

The source contains 1,067,371 transaction rows from a UK online retailer,
dated 2009-12-01 through 2011-12-09. Products include gifts and many customers
are wholesalers. Source license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

`weekly_sales.csv`, `products.json`, and `audit.json` are transformed derivatives
of that dataset, shared under the same CC BY 4.0 license. UCI and the dataset
author do not endorse this project. No customer identifiers are redistributed.

## Transformation

1. Combine the two original workbook sheets, treating exact duplicate records
   as repeated entries. This assumption may also remove legitimate identical lines.
2. Keep UK rows with finite positive quantities/prices, excluding cancellation
   invoices and non-product codes. Product codes match five digits plus up to two letters.
3. Retain complete Monday–Sunday weeks from 2009-12-07 through 2011-12-04.
4. Use only weeks before 2011-05-02 to select the 50 largest products by units,
   requiring sales in at least 30 training weeks. Ties are resolved by product code.
5. Sum positive recorded units by product and week. Fill weeks without a
   transaction with zero. Do not remove genuine large positive orders.
6. Obtain product descriptions from the training-period mode only.

The committed panel has 5,200 product-weeks across 104 weeks.
`audit.json` contains mutually sequential filter counts that reconcile to the raw row count.
Excluded cancellations are not matched to prior sales: the target is **gross
positive recorded sales**, not net revenue or latent demand. Missing customer IDs
do not cause a sale to be excluded.

Original workbook SHA-256:
`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`

Panel SHA-256:
`ed59498aa8c70f789b7910be2cc5714d34c879bfbf4ea29324e7e59b0e55dc49`

Rebuild from the original file with `shelfcast reproduce --from-source`.
The downloader checks the source hash and reads only the expected workbook member.
Raw files and a checksum-guarded CSV cache stay in the ignored `data/raw/` directory.
