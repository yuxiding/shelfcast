# Serving the research model

Run the experiment before starting the service. `SHELFCAST_MODEL` optionally
selects a trusted model artifact; the default is `outputs/model.joblib`.
The service offers `/health`, `/products`, `/forecast`, and `/docs`.

Validation rejects unknown products, negative or nonfinite units, duplicate/gapped
history, non-Monday weeks, stale history, negative inventory, and unexpected input
fields. At least eight complete weeks are needed, and 52 enable the annual lag.
The forecast week must be later than the artifact's calibration block.

History contains recorded weekly sales supplied by the caller. An explicit zero
means no recorded units, not proof that the product was in stock. Missing weeks
must be resolved by the data pipeline before calling the API.

## Optional container

```bash
docker build -t shelfcast .
docker run --rm -p 8000:8000 \
  -v "$PWD/outputs:/app/outputs:ro" shelfcast
```

The process runs as a non-root user and reads the mounted model. This container
configuration is provided for local use; Docker execution was unavailable in the
authoring environment. The package, API, and saved-model inference were tested directly.

The service returns suggestions only. Authentication, atomic stock accounting,
purchase execution, autoscaling, deployment monitoring, and modern retailer data
would be separate work before production use.
