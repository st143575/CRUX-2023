"""
Read the LDC claim frame annotation into AnnotatedClaimFrame objects.
"""


from __future__ import annotations

import csv
from collections.abc import Iterable
from crux.domain.claim import AnnotatedClaimFrame
from crux.domain.ldc_time import ClaimDatetime
from crux.domain.topic import ClaimTemplate
from crux.ingestion.issues import IngestionIssue, summarize_validation_error
from crux.ingestion.read_topic_list import index_by_text
from pathlib import Path
from pydantic import ValidationError


AFFILIATION_SUFFIXES: tuple[str, ...] = ("", "_1", "_2")

# 35 columns
REQUIRED_COLUMNS: tuple[str, ...] = (
    "root_uid", "claim_id", "topic", "subtopic", "claim_template", 
    "x_variable", "qnode_x_variable_identity", "qnode_x_variable_type",
    "description", "claim_semantics", 
    "claimer", "qnode_claimer_identity", "qnode_claimer_type", "claimer_provenance",
    "claimer_affiliation", "claimer_affiliation_1", "claimer_affiliation_2",
    "qnode_claimer_affiliation_identity", "qnode_claimer_affiliation_identity_1", "qnode_claimer_affiliation_identity_2",
    "qnode_claimer_affiliation_type", "qnode_claimer_affiliation_type_1", "qnode_claimer_affiliation_type_2",
    "epistemic_status", "sentiment_status", "claim_datetime", 
    "claim_location", "qnode_claim_location_identity", "qnode_claim_location_type", "claim_location_provenance",
    "claim_medium", "qnode_claim_medium_identity", "qnode_claim_medium_type", "claim_medium_provenance",
    "associated_kes",
)


def read_claim_frames(path: Path, claim_templates: Iterable[ClaimTemplate]) -> tuple[tuple[AnnotatedClaimFrame, ...], list[IngestionIssue]]:
    """
    Read the LDC claim frame annotation into AnnotatedClaimFrame objects.
    
    Args:
        path: Path to the claim frame annotation file.
        claim_templates: Iterable of ClaimTemplate objects.
        
    Returns:
        A tuple of (claim frames, issues).
    """
    claim_frames: list[AnnotatedClaimFrame] = []
    issues: list[IngestionIssue] = []
    first_line: dict[str, int] = {}
    claim_frame_idx = index_by_text(claim_templates)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

        # Check for missing columns
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            return (), [IngestionIssue.error(path, "header", f"missing columns: {missing}")]

        for row in reader:
            where = f"line {reader.line_num}"
            if None in row or any(v is None for v in row.values()):
                issues.append(IngestionIssue.error(path, where, "missing required column value"))
                continue
            
            claim_id = row["claim_id"].strip()
            if claim_id in first_line:
                issues.append(IngestionIssue.error(path, where, f"duplicate claim_id: {claim_id} (first at line {first_line[claim_id]}), skipped"))
                continue
            first_line[claim_id] = reader.line_num

            claim_template = claim_frame_idx.get((row["topic"].strip(), row["claim_template"].strip()))
            if claim_template is None:
                issues.append(IngestionIssue.error(path, where, f"{claim_id}: template not in topic list: {row['topic']!r} / {row['claim_template']!r}"))
                continue
            if row["subtopic"].strip() != claim_template.subtopic:
                issues.append(IngestionIssue.error(path, where, f"{claim_id}: subtopic mismatch: {row['subtopic']!r} != {claim_template.subtopic!r}"))
                continue

            try:
                claim_datetime = ClaimDatetime.parse(row["claim_datetime"])
            except ValueError as e:
                issues.append(IngestionIssue.error(path, where, f"{claim_id}: invalid claim_datetime: {e}"))
                continue

            affiliations = []
            for suffix in AFFILIATION_SUFFIXES:
                name = _empty_na_to_none(row[f"claimer_affiliation{suffix}"])
                if name is None:
                    if _empty_na_to_none(row[f"qnode_claimer_affiliation_identity{suffix}"]) or _empty_na_to_none(row[f"qnode_claimer_affiliation_type{suffix}"]):
                        issues.append(IngestionIssue.error(path, where, f"{claim_id}: missing claimer_affiliation but qnode exists"))
                    continue
                affiliations.append({
                    "name": name,
                    "qnode_identity_id": _empty_na_to_none(row[f"qnode_claimer_affiliation_identity{suffix}"]),
                    "qnode_type_ids": _empty_na_to_empty_tuple(row[f"qnode_claimer_affiliation_type{suffix}"]),
                })

            try:
                claim_frames.append(AnnotatedClaimFrame.model_validate({
                    "root_uid": row["root_uid"].strip(),
                    "claim_id": claim_id,
                    "claim_template": claim_template,
                    "x_variable": {
                        "name": _empty_na_to_none(row["x_variable"]),
                        "qnode_identity_id": _empty_na_to_none(row["qnode_x_variable_identity"]),
                        "qnode_type_ids": _empty_na_to_empty_tuple(row["qnode_x_variable_type"]),
                    },
                    "description": _empty_na_to_none(row["description"]),
                    "claim_semantics": _process_ke_mention_ids(row["claim_semantics"]),
                    "claimer": {
                        "name": _empty_na_to_none(row["claimer"]),
                        "qnode_identity_id": _empty_na_to_none(row["qnode_claimer_identity"]),
                        "qnode_type_ids": _empty_na_to_empty_tuple(row["qnode_claimer_type"]),
                        "provenance": _empty_na_to_none(row["claimer_provenance"]),
                        "affiliations": affiliations,
                    },
                    "epistemic_status": row["epistemic_status"].strip(),
                    "sentiment_status": row["sentiment_status"].strip(),
                    "claim_datetime": claim_datetime,
                    "claim_location": _process_claim_location_or_medium(row, "claim_location"),
                    "claim_medium": _process_claim_location_or_medium(row, "claim_medium"),
                    "associated_kes": _process_ke_mention_ids(row["associated_kes"]),
                }))
            except ValidationError as e:
                issues.append(IngestionIssue.error(path, where, f"{claim_id}: {summarize_validation_error(e)}"))

    if not claim_frames and not issues:
        issues.append(IngestionIssue.error(path, "file", "no claim frames found"))

    return tuple(claim_frames), issues



def _empty_na_to_none(value: str) -> str | None:
    """
    Strip a cell and map "EMPTY_NA" and "" to None.
    """
    value = value.strip()
    return None if value in ("", "EMPTY_NA") else value

def _empty_na_to_empty_tuple(value: str) -> tuple[str, ...]:
    """
    Strip a cell and map "EMPTY_NA" and "" to empty tuple.
    """
    type_qnodes = _empty_na_to_none(value)
    return () if type_qnodes is None else tuple(qnode.strip() for qnode in type_qnodes.split("|"))

def _process_ke_mention_ids(value: str) -> tuple[str, ...]:
    """
    Split the KE mention IDs by comma.
    If the value is EMPTY_NA, return an empty tuple.
    """
    mention_ids = _empty_na_to_none(value)
    return () if mention_ids is None else tuple(mention_id.strip() for mention_id in mention_ids.split(","))

def _process_claim_location_or_medium(row: dict[str, str], name: str) -> dict[str, object] | None:
    """
    Process a ClaimLocation or ClaimMedium object.
    """
    value = _empty_na_to_none(row[name])
    if value is None:
        return None
    return {
        "name": value,
        "qnode_identity_id": _empty_na_to_none(row[f"qnode_{name}_identity"]),
        "qnode_type_ids": _empty_na_to_empty_tuple(row[f"qnode_{name}_type"]),
        "provenance": _empty_na_to_none(row[f"{name}_provenance"]),
    }
    
    