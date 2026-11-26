"""
Read the claim frames in the output format into string rows.
Used for both the gold output and the system output for Task 1.
"""

from __future__ import annotations

import csv
from crux.domain.output import OUTPUT_COLUMNS, OutputClaimFrameFields
from crux.ingestion.issues import IngestionIssue
from pathlib import Path


def read_output_claim_frames(path: Path) -> tuple[tuple[OutputClaimFrameFields, ...], list[IngestionIssue]]:
    """
    Read either the 35-column LDC claim frame annotation file and map it to the 12-column system output format,
    or read a 12-column system output file directly.

    Args:
        path: Path to a gold or system output file for Task 1.

    Returns:
        A tuple of (rows, issues). Each row is a tuple of 12 strings in OUTPUT_COLUMNS order.
    """
    rows: list[OutputClaimFrameFields] = []
    issues: list[IngestionIssue] = []
    first_line: dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None:
            return (), [IngestionIssue.error(path, "file", "empty file")]

        header = [h.strip() for h in header]
        # if len(header) != len(OUTPUT_COLUMNS):
        #     return (), [IngestionIssue.error(path, "header", f"Header has {len(header)} columns, expected {len(OUTPUT_COLUMNS)}")]
        if len(set(header)) != len(header):
            return (), [IngestionIssue.error(path, "header", f"duplicate column names: {header}")]

        # Check missing columns
        missing = [c for c in OUTPUT_COLUMNS if c not in header]
        if missing:
            return (), [IngestionIssue.error(path, "header", f"missing columns: {missing}")]
        
        # Check extra columns
        extra = [c for c in header if c not in OUTPUT_COLUMNS]
        if extra:
            issues.append(IngestionIssue.warning(path, "header", f"unexpected columns ignored: {extra}"))
        
        # Get the order of columns
        order = [header.index(c) for c in OUTPUT_COLUMNS]

        for fields in reader:
            where = f"line {reader.line_num}"
            if not any(v.strip() for v in fields):
                issues.append(IngestionIssue.warning(path, where, "empty row"))
                continue
            if len(fields) != len(header):
                issues.append(IngestionIssue.error(path, where, f"row has {len(fields)} columns, expected {len(header)}"))
                continue
            
            row = OutputClaimFrameFields(*[fields[i] for i in order])
            
            empty = [c for c, v in zip(OUTPUT_COLUMNS, row) if not v.strip()]
            if empty:
                issues.append(IngestionIssue.warning(path, where, f"empty fields: {empty}"))

            claim_id = row.claim_id
            if claim_id in first_line:
                issues.append(IngestionIssue.error(path, where, f"duplicate claim_id: {claim_id} (first at line {first_line[claim_id]}), skipped"))
                continue
            first_line[claim_id] = reader.line_num
            rows.append(row)
    
    return tuple(rows), issues