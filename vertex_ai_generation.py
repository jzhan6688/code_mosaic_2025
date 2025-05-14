#!/usr/bin/env python3
"""
bulk_review_summarizer_vertex_batch.py
--------------------------------------
• INPUT  : a newline‑delimited JSON file in GCS
           gs://<bucket>/<object>.jsonl
           Each line must contain either
             { "listing_id": "...", "reviews": [...] }
           or
             { "input": { "listing_id": "...", "reviews": [...] }, ... }

• OUTPUT : Vertex AI Batch Prediction job results
           (requests are staged in BigQuery table
            ${PROJECT}.${DATASET}.${INPUT_TAB})


conda create -n vertexai_env \
  python=3.11 \
  pandas \
  tqdm \
  google-cloud-storage \
  google-cloud-bigquery \
  google-cloud-aiplatform \
  -c conda-forge

conda activate vertexai_env
pip install vertexai
pip install pandas tqdm google-cloud-storage google-cloud-bigquery google-cloud-aiplatform vertexai db-dtypes pyarrow
"""

import argparse
import json
import re
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from google.cloud import storage, bigquery, aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.batch_prediction import BatchPredictionJob


# ---------- CONFIG -------------------------------------------------------
PROJECT   = "etsy-inventory-ml-dev"
REGION    = "us-central1"
DATASET   = "iml_development"
INPUT_TAB = "james_code_mosaic_2025"
MODEL_ID  = "gemini-2.0-flash-001"

SYSTEM_PROMPT = (
    "Your task as an e‑commerce and consumer product expert is to analyze a set "
    "of Etsy listing reviews. Identify common themes, key attributes, and overall "
    "sentiments expressed by buyers, focusing on Etsy's distinctive marketplace "
    "and products. Given the following buyer reviews, summarize the reviews in 1 sentence."
)

USER_TEMPLATE = """
Here are strict guidelines to follow when summarizing the listing reviews:
- Only include information in the review summary if the information is found in multiple reviews.
- Do not include any personally identifiable information. This includes any information that could be used to identify the Buyer or Seller, including the use of first and last names.
- Do not provide a final sentence in the summary conveying what the overall sentiment of the reviews were.
- Please provide the response in plain text only.
- Do not include personal attacks against the seller. For example, if a review says 'the seller is a jerk' do not include this type of content in your review summary.

Buyer Reviews: {reviews}

Your final output should be "[your summary]" and no other text or extra newlines. If there is not enough information to provide an insightful summary, your final output should be 'Not enough data'.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reviews": {"type": "STRING"},
    },
    "required": ["reviews"],
}

# ---------- HELPERS ------------------------------------------------------
def _parse_gcs_uri(uri: str):
    m = re.match(r"^gs://([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return m.group(1), m.group(2)


def read_jsonl_from_gcs(gcs_uri: str) -> pd.DataFrame:
    """Stream NDJSON from GCS into a flattened DataFrame."""
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    blob = storage.Client(project=PROJECT).bucket(bucket_name).blob(blob_name)
    raw = blob.download_as_bytes()

    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    # flatten nested `input.*` keys if present
    return pd.json_normalize(records, sep=".")


def extract_reviews(row: pd.Series) -> List[str]:
    return (
        row.get("reviews")
        or row.get("input.reviews")
        or []
    )


def extract_listing_id(row: pd.Series) -> str:
    return (
        str(row.get("listing_id") or row.get("input.listing_id") or "")
    )


def build_request(row: pd.Series) -> str:
    prompt = USER_TEMPLATE.format(reviews=extract_reviews(row) or "None")

    req = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generation_config": {
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "temperature": 0.2,
        },
    }
    return json.dumps(req, ensure_ascii=False)


# ---------- MAIN ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Batch summarizer (GCS jsonl input only)"
    )
    parser.add_argument("--gcs-uri", required=True,
                        help="gs://bucket/path/to/listings.jsonl")
    args = parser.parse_args()

    # init clients
    vertexai.init(project=PROJECT, location=REGION)
    bq = bigquery.Client(project=PROJECT, location=REGION)
    aiplatform.init(project=PROJECT, location=REGION)

    # load data
    df = read_jsonl_from_gcs(args.gcs_uri)

    # stage requests in BigQuery
    table_id = f"{PROJECT}.{DATASET}.{INPUT_TAB}"
    schema = [
        bigquery.SchemaField("request", "STRING"),
        bigquery.SchemaField("listing_id", "STRING"),
    ]
    bq.delete_table(table_id, not_found_ok=True)
    bq.create_table(bigquery.Table(table_id, schema=schema))
    print(f"Created table {table_id}")

    rows = [
        {
            "request": build_request(row),
            "listing_id": extract_listing_id(row),
        }
        for _, row in df.iterrows()
    ]

    CHUNK = 50_000
    for start in tqdm(range(0, len(rows), CHUNK), desc="Uploading"):
        errors = bq.insert_rows_json(table_id, rows[start : start + CHUNK])
        if errors:
            raise RuntimeError(f"BQ insert errors: {errors}")

    print("Upload complete. Submitting Vertex AI batch prediction job…")

    # launch batch job
    model = GenerativeModel(model_name=MODEL_ID)
    job: BatchPredictionJob = BatchPredictionJob.submit(
        model,
        input_dataset=f"bq://{table_id}",
        output_uri_prefix=f"bq://{PROJECT}.{DATASET}",
    )

    print(f"Job submitted: {job.resource_name}")
    print("Tip: gcloud ai batch-predictions describe", job.name)


if __name__ == "__main__":
    main()
