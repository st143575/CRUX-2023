"""
Command-line interface of the crux package.

Run:
    crux score task1 PATH_TO_SYSTEM_OUTPUT PATH_TO_GOLD [--keep-empty-na] [--raw] [--lenient] [--docs FILE] [--json FILE] [-v]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from crux import __version__
from crux.eval.cfe_scorer import (
    CFEReport,
    CFEScorer,
    ScoringInputError,
    normalize_row,
    score_claim_frame_files,
)
from crux.ingestion.issues import IngestionIssue, IssueType


# Exit codes
EXIT_OK = 0
EXIT_INPUT_ERROR = 1        # argparse itself uses 2 for usage errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


###############################################################################
#                                   Parser                                    #
###############################################################################

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crux", description=__doc__.strip().splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="score system output against gold annotations")
    tasks = score.add_subparsers(dest="task", required=True)

    task1 = tasks.add_parser(
        "task1",
        help="Task 1: claim frame extraction",
        description="Score a Task 1 system output file against a gold file "
                    "(TAC 2023 CRUX Evaluation Plan, Sec. 2.1).",
    )
    task1.add_argument("pred", type=Path, help="system output file (.tab)")
    task1.add_argument("gold", type=Path, help="gold file (.tab, 12-column output format or LDC annotation format)")
    task1.add_argument("--keep-empty-na", action="store_true",
                       help="count EMPTY_NA as a value in the per-field (secondary) metric")
    task1.add_argument("--raw", action="store_true",
                       help="compare raw strings, without normalization")
    task1.add_argument("--lenient", action="store_true",
                       help="count unreadable system rows as false positives instead of failing")
    task1.add_argument("--docs", type=Path, metavar="FILE",
                       help="score only the documents listed in FILE (one root_uid per line)")
    task1.add_argument("--json", type=Path, metavar="FILE", dest="json_path",
                       help="also write the full report as JSON to FILE ('-' for stdout)")
    task1.add_argument("-v", "--verbose", action="store_true",
                       help="list every ingestion warning, not only their number")
    task1.set_defaults(func=_score_task1)

    return parser


###############################################################################
#                                  Commands                                   #
###############################################################################

def _score_task1(args: argparse.Namespace) -> int:
    scorer = CFEScorer(
        normalize=None if args.raw else normalize_row,
        ignore_empty_na=not args.keep_empty_na,
    )
    try:
        root_uids = _read_doc_list(args.docs) if args.docs else None
        report, issues = score_claim_frame_files(
            args.pred, args.gold, scorer, strict=not args.lenient, root_uids=root_uids,
        )
    except ScoringInputError as e:
        print(f"error: {e}", file=sys.stderr)
        _print_issues(e.issues, verbose=True)
        return EXIT_INPUT_ERROR
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    _print_issues(issues, verbose=args.verbose)

    if args.json_path is not None:
        payload = {
            "pred": str(args.pred),
            "gold": str(args.gold),
            "config": {
                "normalize": not args.raw,
                "ignore_empty_na": scorer.ignore_empty_na,
                "strict": not args.lenient,
                "docs": str(args.docs) if args.docs else None,
            },
            "report": report.to_dict(),
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if str(args.json_path) == "-":
            print(text)
            return EXIT_OK
        args.json_path.write_text(text + "\n", encoding="utf-8")

    print(format_cfe_report(report, ignore_empty_na=scorer.ignore_empty_na))
    return EXIT_OK


###############################################################################
#                                   Output                                    #
###############################################################################

def format_cfe_report(report: CFEReport, *, ignore_empty_na: bool) -> str:
    c = report.claims
    lines = [
        "Task 1: claim frame extraction",
        f"  Precision  {c.precision:.4f}",
        f"  Recall     {c.recall:.4f}",
        f"  F1         {c.f1:.4f}",
        f"  TP {c.tp}   FP {c.fp}   FN {c.fn}"
        + (f"   (incl. {report.n_pred_unparsed} unreadable system rows as FP)" if report.n_pred_unparsed else ""),
        f"  Weighted accuracy  {report.weighted_accuracy:.4f}",
        "",
        f"Per-field P/R/F1 (EMPTY_NA {'ignored' if ignore_empty_na else 'counted'})",
        f"  {'field':<20} {'P':>7} {'R':>7} {'F1':>7}",
    ]
    for name, prf in report.fields.items():
        if prf.n_pred == prf.n_gold == 0:       # nothing to score, e.g. only EMPTY_NA values
            lines.append(f"  {name:<20} {'-':>7} {'-':>7} {'-':>7}")
        else:
            lines.append(f"  {name:<20} {prf.precision:>7.4f} {prf.recall:>7.4f} {prf.f1:>7.4f}")
    return "\n".join(lines)


def _print_issues(issues: Sequence[IngestionIssue], *, verbose: bool) -> None:
    if not issues:
        return
    if verbose:
        for i in issues:
            print(f"{i.issue_type}: {i.source}: {i.location}: {i.message}", file=sys.stderr)
        return
    n_err = sum(1 for i in issues if i.issue_type == IssueType.ERROR)
    n_warn = len(issues) - n_err
    print(f"{n_err} error(s), {n_warn} warning(s) while reading the input files (use -v to list them)",
          file=sys.stderr)


def _read_doc_list(path: Path) -> list[str]:
    """
    One root_uid per line; blank lines and lines starting with '#' are skipped.
    """
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


if __name__ == "__main__":
    sys.exit(main())
