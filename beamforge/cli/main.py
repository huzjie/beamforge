"""Command-line interface: doctor / train / bench / serve / report."""
import argparse
import json
import sys


def _cmd_doctor(args):
    from ..model.engine import BeamEngine
    from ..moe.layer import MoELayer
    from ..fault.orchestrator import FaultTolerantOrchestrator
    from ..sandbox.factory import SandboxFactory
    from ..backends import list_backends
    from ..bench import list_benchmarks
    e = BeamEngine()
    moe = MoELayer(dim=16, hidden=16, n_experts=8, top_k=2)
    orch = FaultTolerantOrchestrator(n_faults=3, checkpoint_dir=".beamforge_ckpt_doctor")
    sf = SandboxFactory(capacity=8)
    print(f"[doctor] backends={list_backends()}")
    print(f"[doctor] benchmarks={list_benchmarks()}")
    print(f"[doctor] moe sparse_ratio={moe.sparse_ratio}")
    print(f"[doctor] engine backend={e.backend.name} skill={e.skill()}")
    print(f"[doctor] fault orchestrator errors={len(orch.injector.stream)}")
    print(f"[doctor] sandbox factory kinds={list(sf.kinds)}")
    print("[doctor] OK")


def _cmd_train(args):
    from ..train.factory import TrainingFactory
    res = TrainingFactory().run_all(
        pretrain_steps=args.steps, midtrain_steps=args.steps * 2, rl_prompts=args.rl)
    pre = res["pretrain"]
    rl = res["rl"]
    print(f"[train] pretrain skill {pre['curve'][0][1]} -> {pre['final_skill']}, "
          f"loss {pre['final_loss']}, acc {pre['accuracy']}")
    print(f"[train] rl skill {rl['skill_initial']} -> {rl['skill_final']}")


def _cmd_bench(args):
    from ..bench import run_all
    res = run_all(n_faults=args.faults)
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _cmd_serve(args):
    from ..serving.server import serve
    serve(port=args.port)


def _cmd_report(args):
    from ..bench import run_all
    from ..train.factory import TrainingFactory
    res = run_all(n_faults=args.faults)
    train = TrainingFactory().run_all()
    payload = {"benchmarks": res, "train": {"pretrain": train["pretrain"]["final_skill"],
                                           "rl": train["rl"]["skill_final"]}}
    out = args.output or "beamforge_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[report] wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="beamforge",
                                 description="sparse frontier-model training factory")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("doctor", help="smoke-test all components")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("train", help="run pretrain -> midtrain -> RL")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rl", type=int, default=20)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("bench", help="run all benchmarks")
    p.add_argument("--faults", type=int, default=71)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("serve", help="start HTTP server")
    p.add_argument("--port", type=int, default=8899)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("report", help="write a full JSON report")
    p.add_argument("--faults", type=int, default=71)
    p.add_argument("--output", type=str, default=None)
    p.set_defaults(func=_cmd_report)

    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
