"""CLI: `python -m mxd.cli scan <events.jsonl>`."""
from __future__ import annotations

import argparse
import json
import sys

from .analyst import LLMExtractionAnalyst
from .detector import ExtractionDetector
from .events import QueryEvent
from .pipeline import DetectionPipeline


def _read_events(path: str) -> list[QueryEvent]:
    out: list[QueryEvent] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(QueryEvent(
                actor_id=d["actor_id"],
                timestamp=float(d["timestamp"]),
                query_text=d.get("query_text", ""),
                response_text=d.get("response_text", ""),
                response_token_count=int(d.get("response_token_count", 0)),
                query_token_count=int(d.get("query_token_count", 0)),
                embedding=d.get("embedding"),
                endpoint=d.get("endpoint", "chat"),
                ip=d.get("ip", ""),
                user_agent=d.get("user_agent", ""),
                request_id=d.get("request_id", ""),
            ))
    return out


def _cmd_scan(args: argparse.Namespace) -> int:
    events = _read_events(args.events)
    pipe = DetectionPipeline()
    verdicts = pipe.evaluate_batch(events)
    out = {actor_id: v.to_dict() for actor_id, v in verdicts.items()}
    if args.analyse:
        a = LLMExtractionAnalyst()
        for actor_id, vdict in out.items():
            out[actor_id]["incident"] = a.analyse(vdict).to_dict()
    txt = json.dumps(out, indent=2)
    if args.output == "-":
        print(txt)
    else:
        with open(args.output, "w") as fh:
            fh.write(txt)
    suspicious = sum(1 for v in verdicts.values() if v.is_suspicious)
    print(f"actors={len(verdicts)} suspicious={suspicious}", file=sys.stderr)
    return 1 if args.fail_on_suspicious and suspicious else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("mxd")
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("events", help="path to JSONL of QueryEvents")
    scan.add_argument("-o", "--output", default="-")
    scan.add_argument("--analyse", action="store_true")
    scan.add_argument("--fail-on-suspicious", action="store_true")
    scan.set_defaults(fn=_cmd_scan)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
