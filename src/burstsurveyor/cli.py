"""Command-line entry point. Successful commands emit a JSON document to stdout."""

import argparse
import json
from pathlib import Path
import sys


def _parser():
    parser = argparse.ArgumentParser(
        prog="burstsurveyor",
        description="Automated fitting of radio bursts and folded pulsar profiles with FitBurst.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    fit = sub.add_parser("fit", help="fit one NPZ file or a directory and export the reconstructions")
    fit.add_argument("inputs", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    fit.add_argument('--mode', choices=('bounded', 'scintillation', 'combined', 'pulsar'), default='bounded',
                     help='bounded is the default; pulsar fits folded profiles with period metadata')
    fit.add_argument(
        "--resume",
        action="store_true",
        help="continue the original run using identical inputs",
    )
    fit.add_argument(
        "--second-look",
        action="store_true",
        help="also run the experimental isolated temporal rescue",
    )
    fit.add_argument(
        "--limit",
        type=int,
        default=8,
        help="maximum additional events for --second-look",
    )
    verify = sub.add_parser("verify", help="check saved fits and hashes")
    verify.add_argument("output", type=Path)
    verify.add_argument(
        "--second-look",
        type=Path,
        metavar="DIRECTORY",
        help="also verify this alternate analysis",
    )
    verify.add_argument("--route-rescue", type=Path, metavar="DIRECTORY",
                        help="also verify an opt-in routed rescue")
    prepare = sub.add_parser(
        "prepare", help="fit the inputs and generate review packets and figures"
    )
    prepare.add_argument("inputs", type=Path)
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--no-auto-rescue", action="store_true")
    prepare.add_argument("--mode", choices=("bounded", "combined", "scintillation", "pulsar"), default="bounded",
                         help="bounded is the default; pulsar fits one complete folded rotation")
    decide = sub.add_parser(
        "decide", help="record assessments and test proposed fits"
    )
    decide.add_argument("workspace", type=Path)
    decide.add_argument("inputs", type=Path)
    decide.add_argument(
        "decisions", type=Path, help="JSON object or array of decisions"
    )
    finish = sub.add_parser(
        "finish", help="verify complete reviews and write the reviewed event table"
    )
    finish.add_argument("workspace", type=Path)
    example = sub.add_parser("example", help="write a synthetic test input")
    example.add_argument("destination", type=Path)
    inspect = sub.add_parser("inspect", help="read the input grid")
    inspect.add_argument("source", type=Path)
    review = sub.add_parser("review", help="list fit flags (internal)")
    review.add_argument("run", type=Path)
    review.add_argument(
        "--sample-clean",
        type=int,
        default=0,
        metavar="N",
        help="also include N unflagged controls",
    )
    plot = sub.add_parser("plot", help="plot a fitted event and its residuals")
    plot.add_argument("run", type=Path)
    plot.add_argument("--event", required=True, dest="event_id")
    plot.add_argument("--input", required=True, type=Path)
    plot.add_argument("--out", required=True, type=Path)
    diag = sub.add_parser("diagnose", help="compute diagnostics for a fitted event")
    diag.add_argument("run", type=Path)
    diag.add_argument("--event", required=True, dest="event_id")
    diag.add_argument("--input", required=True, type=Path)
    triage = sub.add_parser("triage", help="rank a sealed fit for review without refitting")
    triage.add_argument("run", type=Path)
    triage.add_argument("inputs", type=Path)
    triage.add_argument("output", type=Path)
    triage.add_argument("--resume", action="store_true")
    triage_verify = sub.add_parser("triage-verify", help="verify a ranked review queue and its evidence")
    triage_verify.add_argument("output", type=Path)
    triage_verify.add_argument("--run", type=Path)
    rescue = sub.add_parser("rescue", help="test a temporal follow-up fit")
    challenge = sub.add_parser("challenge", help="test a proposed additional component")
    for cmd in (rescue, challenge):
        cmd.add_argument("run", type=Path)
        cmd.add_argument("inputs", type=Path)
        cmd.add_argument("output", type=Path, help="a separate output directory")
        cmd.add_argument("--resume", action="store_true")
    rescue.add_argument("--limit", type=int, default=8)
    challenge.add_argument("--proposal", required=True, type=Path)
    routed = sub.add_parser("route-rescue", help="try one targeted rescue on a sealed burst fit")
    routed.add_argument("run", type=Path)
    routed.add_argument("input", type=Path, help="one original NPZ from the sealed run")
    routed.add_argument("output", type=Path, help="a new directory for the rescue result")
    sub.add_parser(
        "doctor", help="check the numerical runtime and upstream FitBurst import"
    )
    schema = sub.add_parser("schema", help="show the results or proposal JSON schema")
    schema.add_argument("kind", choices=("results", "proposal"))
    return parser


def _brief(summary, folder):
    return dict(
        output=str(folder.resolve()),
        inputs=summary["inputs"],
        exported=summary["exported"],
        statuses=summary["statuses"],
        results=str((folder / "results.json").resolve()),
    )


def _run(args):
    if args.command in {"prepare", "decide", "finish"}:
        from . import assistant

        if args.command == "prepare":
            return assistant.prepare(
                args.inputs, args.workspace, workers=args.workers, resume=args.resume,
                auto_rescue=not args.no_auto_rescue, mode=args.mode
            )
        if args.command == "decide":
            return assistant.decide(args.workspace, args.inputs, args.decisions)
        return assistant.finish(args.workspace)
    if args.command == "triage":
        from .triage import audit
        return audit(args.run, args.inputs, args.output, resume=args.resume)
    if args.command == "triage-verify":
        from .triage import verify_queue
        return verify_queue(args.output, args.run)
    if args.command == "schema":
        from importlib.resources import files

        path = files("burstsurveyor").joinpath("data", args.kind + ".schema.json")
        return json.loads(path.read_text(encoding="utf-8"))
    if args.command == "doctor":
        from importlib.metadata import version
        from ._upstream import verified_sources

        verified_sources()
        from fitburst.analysis.model import SpectrumModeler

        return dict(
            product="Burst Surveyor",
            python=sys.version.split()[0],
            renderer=SpectrumModeler.__name__,
            packages={
                n: version(n)
                for n in ("fitburst", "numpy", "scipy", "psutil", "threadpoolctl")
            },
        )
    if args.command == "example":
        from .example import create_example

        return dict(synthetic_input=str(create_example(args.destination).resolve()))
    if args.command == "inspect":
        from .science import inspect_npz

        return inspect_npz(args.source)
    if args.command == "fit":
        from .batch import fit

        if args.mode in {'scintillation', 'combined', 'pulsar'} and args.second_look:
            raise ValueError('This selection mode uses an internal search; inspect remaining residuals rather than requesting a bounded second look.')
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        summary = _brief(
            fit(args.inputs, args.output, workers=args.workers, resume=args.resume, mode=args.mode),
            args.output,
        )
        if args.second_look:
            from .secondlook import second_look

            extra = args.output.resolve().with_name(
                args.output.resolve().name + ".second-look"
            )
            second_look(
                args.output,
                args.inputs,
                extra,
                limit=args.limit,
                resume=args.resume and extra.exists(),
            )
            summary["second_look"] = str(extra)
        return summary
    if args.command == "verify":
        from .batch import verify

        stock = verify(args.output)
        summary = _brief(stock, args.output)
        if args.second_look is not None:
            from .secondlook import verify_second_look

            summary["second_look"] = verify_second_look(args.output, args.second_look)
        if args.route_rescue is not None:
            from .route_rescue import verify_route_rescue

            summary["route_rescue"] = verify_route_rescue(args.output, args.route_rescue, stock)
        return summary
    if args.command == "route-rescue":
        from .route_rescue import route_rescue

        return route_rescue(args.run, args.input, args.output)
    if args.command == "review":
        from .review import queue

        return dict(
            schema_version=1, events=queue(args.run, sample_clean=args.sample_clean)
        )
    if args.command == "plot":
        from .review import plot_event

        return dict(
            plot=str(
                plot_event(args.run, args.event_id, args.input, args.out).resolve()
            )
        )
    if args.command == "diagnose":
        from .diagnostics import diagnose

        return diagnose(args.run, args.event_id, args.input)
    from .secondlook import second_look

    return second_look(
        args.run,
        args.inputs,
        args.output,
        proposal=getattr(args, "proposal", None),
        limit=getattr(args, "limit", 1),
        resume=args.resume,
    )


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
        print(json.dumps(result, allow_nan=False, ensure_ascii=True))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"burstsurveyor: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
