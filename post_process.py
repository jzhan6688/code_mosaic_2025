#!/usr/bin/env python3
"""
Export listing-level review summaries from a Vertex AI Batch Prediction
results table in BigQuery.

The script expects the predictions table to contain at least these fields:
  • `listing_id` – string
  • `request`    – JSON string sent to Gemini
  • `response`   – JSON string returned by Gemini

It produces a CSV with columns
  listing_id, buyer_reviews, summary


Example
-------
python bq_to_csv.py \
    --table etsy-inventory-ml-dev.iml_development.predictions_2025-05-13-22-03-26-9bd08 \
    --output summaries.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BUYER_REVIEWS_RE = re.compile(r"Buyer Reviews:\s*(\[[^]]*\])", re.S)

def _parse_buyer_reviews(req: str) -> List[str]:
    """Return the list of buyer reviews embedded in the request payload."""
    try:
        req_obj = json.loads(req)
        prompt_text: str = (
            req_obj["contents"][0]["parts"][0]["text"]  # type: ignore[index]
        )
    except Exception:
        return []

    m = BUYER_REVIEWS_RE.search(prompt_text)
    if not m:
        return []

    list_literal = m.group(1)
    try:
        # The list is represented with Python single quotes, so ast.literal_eval
        # is more forgiving than json.loads.
        return ast.literal_eval(list_literal)
    except Exception:
        return []


def _parse_summary(resp: str) -> Optional[str]:
    """Extract the model-generated summary from the response payload."""
    try:
        resp_obj = json.loads(resp)
        text_block: str = resp_obj["candidates"][0]["content"]["parts"][0][
            "text"
        ]  # type: ignore[index]
    except Exception:
        return None

    # The text block itself is a JSON string like { "reviews": "..." }
    try:
        inner = json.loads(text_block)
        return inner.get("reviews")
    except Exception:
        # If that fails, just return raw text.
        return text_block.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export Gemini review summaries");
    parser.add_argument("--table", required=True, help="Fully-qualified BigQuery table, e.g. project.dataset.table")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging")
    args = parser.parse_args(argv)

    client = bigquery.Client()

    sql = f"SELECT listing_id, request, response FROM `{args.table}`"
    if args.limit:
        sql += f" LIMIT {args.limit}"

    print(f"Running query on {args.table} …", file=sys.stderr)
    df = client.query(sql).to_dataframe()
    print(f"Fetched {len(df)} rows", file=sys.stderr)

    out_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        listing_id: str = str(row["listing_id"])
        buyer_reviews = _parse_buyer_reviews(row["request"])
        summary = _parse_summary(row["response"])
        out_records.append(
            {
                "listing_id": listing_id,
                "buyer_reviews": " | ".join(buyer_reviews),  # join for CSV readability
                "summary": summary,
            }
        )

    out_df = pd.DataFrame(out_records)
    output_path = Path(args.output)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote CSV to {output_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
