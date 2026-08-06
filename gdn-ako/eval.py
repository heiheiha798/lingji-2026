"""
评测程序：固定性能点给出正确性与性能；随机用例只检查正确性。

  # 选手自测（baseline 当场同机测，得环境无关加速比供迭代；跨选手不保证可比）
  python eval.py --submission submission.py
  # 组织方复测：① 生成统一 baseline（此进程不 import 选手代码 ⇒ 免投毒）
  python eval.py --make-baseline baseline.json
  #             ② 逐份提交共用同一分母（可换 seed）
  python eval.py --submission A.py --baseline baseline.json
  # golden 校准（真机第一次）：python reference.py --calibrate

golden = fla fused_recurrent(fp32)；基线 = fla chunk(bf16)。
"""

import argparse
import importlib.util
import json
import math
import platform
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path

import torch

import config as C
from reference import make_inputs, reference_output, SCALE, _rel_l2
from baseline import baseline_chunk


def _load_submission(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "gdn_chunk_scan"), "submission 必须提供 gdn_chunk_scan(q,k,v,g,beta,scale)"
    return mod.gdn_chunk_scan


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _fwd_call(fn, inp):
    return fn(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], SCALE)


def _valid_output(o, B, T):
    """输出须为 CUDA bf16、形状 [B,T,Hv,V]。contiguity 不查（布局是合法可优化项）。"""
    return (torch.is_tensor(o) and o.is_cuda
            and o.dtype == C.IO_DTYPE
            and tuple(o.shape) == (B, T, C.NUM_V_HEADS, C.HEAD_V))


def _clone_inputs(inp):
    """给选手一份拷贝：即使选手 in-place 改输入，golden 也已在纯净输入上算好，骗不过。"""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}


def _fwdbwd_step(fn, inp, grad_out):
    for name in ("q", "k", "v", "beta"):
        inp[name].grad = None
    o = _fwd_call(fn, inp)
    o.backward(grad_out)


def _make_pool(B, T, dev, seed, requires_grad=False):
    """TIMING_POOL 份不同 data_ptr、不同值的独立输入，供计时轮换。"""
    return [make_inputs(B, T, device=dev, seed=seed + 5000 + j * 131, requires_grad=requires_grad)
            for j in range(C.TIMING_POOL)]


def _refill_inplace(inp, seed):
    """同 data_ptr、新随机值重填一组输入（不新分配）。"""
    fresh = make_inputs(inp["q"].shape[0], inp["q"].shape[1], device=inp["q"].device, seed=seed)
    for name in ("q", "k", "v", "g", "beta"):
        inp[name].data.copy_(fresh[name])


def _refresh_pool_inplace(pool, base_seed, round_idx):
    """每轮计时前、计时区外，用新随机值 in-place 重填整个 pool（data_ptr 不变）。
    跨轮值变 ⇒ 「按内容/ptr 缓存输出」的作弊必然 miss、被迫真算；合法 kernel 照算不亏，
    合法固定-buffer/ptr-captured graph 读到新值仍正确。详见 README 2.6。"""
    for j, inp in enumerate(pool):
        _refill_inplace(inp, base_seed + 900000 + round_idx * 7919 + j * 131)


def _time_ms(call, pool, refresh=None, warmup=C.WARMUP_ITERS, iters=C.TIMED_ITERS, rounds=C.TIMED_ROUNDS):
    """多轮取中位（抗噪声/漂移）。每轮开始前在计时区外刷新 pool 值，计时循环内再轮换不同 data_ptr——
    两者合起来堵住「重复同输入 / 缓存输出」的计时空子（详见 README 2.6）。"""
    K = len(pool)
    for i in range(warmup):
        call(pool[i % K])
    torch.cuda.synchronize()
    times = []
    for r in range(rounds):
        if refresh is not None:
            refresh(pool, r)
            torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(iters):
            call(pool[i % K])
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / iters)
    times.sort()
    return times[len(times) // 2]


def _machine_fingerprint():
    """baseline 存档指纹（GPU 型号 + 关键库版本）。跨型号/换版本即拒绝复用。
    注：只认型号+版本串，不含 GPU UUID/driver/clocks——同型号同版本的另一台会被视为同机。"""
    import triton
    try:
        import fla
        fla_v = getattr(fla, "__version__", "unknown")
    except Exception:
        fla_v = "unknown"
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "triton": triton.__version__, "fla": fla_v, "cuda": torch.version.cuda}


def _environment():
    result = _machine_fingerprint()
    result.update({
        "os": platform.platform(),
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    return result


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path, result):
    if path is None:
        return
    path.write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入：{path}")


def _grad_out(B, T, dev, seed):
    return torch.randn(B, T, C.NUM_V_HEADS, C.HEAD_V, device=dev, dtype=C.IO_DTYPE,
                       generator=torch.Generator(device=dev).manual_seed(seed + 7))


def measure_baseline(B, T, seed, dev="cuda"):
    """当场测 fla 基线时延（fwd, fwd+bwd）。时延与随机值无关，只依赖形状/dtype。"""
    grad_out = _grad_out(B, T, dev, seed)
    pool = _make_pool(B, T, dev, seed)
    pool_bw = _make_pool(B, T, dev, seed, requires_grad=True)
    ref = lambda p, r: _refresh_pool_inplace(p, seed, r)
    base_fwd = _time_ms(lambda inp: baseline_chunk(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], SCALE),
                        pool, refresh=ref)
    base_fwdbwd = _time_ms(lambda inp: _fwdbwd_step(
        lambda q, k, v, g, b, s: baseline_chunk(q, k, v, g, b, s), inp, grad_out), pool_bw, refresh=ref)
    return base_fwd, base_fwdbwd


def _shape_thresholds(B, T, seed, dev="cuda"):
    """本形状自校准容差：用本形状 fla chunk(bf16) vs golden(fp32) 的差距定阈值，
    避免用单一形状的校准跨形状套用而误杀长序列。返回 (fwd_thresh, bwd_thresh, ref_gap, o_gold, inp)。"""
    inp = make_inputs(B, T, device=dev, seed=seed)
    o_gold = reference_output(inp)
    o_base = baseline_chunk(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], SCALE).float()
    ref_gap = _rel_l2(o_base, o_gold).item()
    fwd_thresh = max(C.FWD_REL_L2_FLOOR, C.REL_TOL_MULT * ref_gap)
    bwd_thresh = max(C.BWD_REL_L2_FLOOR, C.REL_TOL_MULT * ref_gap)
    return fwd_thresh, bwd_thresh, ref_gap, o_gold, inp


def eval_shape(sub_fn, B, T, seed, baseline_cache=None):
    dev = "cuda"
    # 本形状自校准容差（顺带拿到 seed 这一份的 golden，供第 0 个正确性 seed 复用）
    fwd_thresh, bwd_thresh, _, o_gold0, inp0 = _shape_thresholds(B, T, seed, dev)

    # ---- 正确性：前向，多 seed 全过。golden 先在纯净输入上算好，选手拿 clone ⇒ 变异输入骗不过 ----
    fwd_rel, fwd_cos, fwd_ok = 0.0, 1.0, True
    for i in range(C.NUM_CORRECT_SEEDS):
        if i == 0:
            inp, o_gold = inp0, o_gold0
        else:
            inp = make_inputs(B, T, device=dev, seed=seed + i * 10007)
            o_gold = reference_output(inp)
        with torch.no_grad():
            o_raw = _fwd_call(sub_fn, _clone_inputs(inp))
        if not _valid_output(o_raw, B, T):
            fwd_ok = False
            continue
        o_sub = o_raw.float()
        rel, cos = _rel_l2(o_sub, o_gold).item(), _cosine(o_sub, o_gold)
        fwd_rel, fwd_cos = max(fwd_rel, rel), min(fwd_cos, cos)
        if not (rel < fwd_thresh and cos > C.COSINE_MIN):
            fwd_ok = False

    # ---- 正确性：反向（对 fla chunk(bf16) 的梯度）。不可微/无梯度/异常 ⇒ bwd_ok=False，不崩，仍给前向分 ----
    grad_out = _grad_out(B, T, dev, seed)
    bwd_rel, bwd_ok = float("nan"), False
    try:
        inp_g = make_inputs(B, T, device=dev, seed=seed, requires_grad=True)
        o_g = _fwd_call(sub_fn, inp_g)
        if torch.is_tensor(o_g) and o_g.requires_grad and _valid_output(o_g, B, T):
            o_g.backward(grad_out)
            if all(inp_g[n].grad is not None for n in ("q", "k", "v", "beta")):
                inp_ref = make_inputs(B, T, device=dev, seed=seed, requires_grad=True)
                baseline_chunk(inp_ref["q"], inp_ref["k"], inp_ref["v"], inp_ref["g"],
                               inp_ref["beta"], SCALE).backward(grad_out)
                bwd_rel = max(_rel_l2(inp_g[n].grad.float(), inp_ref[n].grad.float()).item()
                              for n in ("q", "k", "v", "beta"))
                bwd_ok = bwd_rel < bwd_thresh
    except Exception:
        bwd_rel, bwd_ok = float("nan"), False

    # ---- 性能：baseline 用统一存档（复测）或当场测（自测）----
    key = f"{B}x{T}"
    if baseline_cache is not None:
        base_fwd, base_fwdbwd = baseline_cache[key]["base_fwd"], baseline_cache[key]["base_fwdbwd"]
    else:
        base_fwd, base_fwdbwd = measure_baseline(B, T, seed, dev)

    ref = lambda p, r: _refresh_pool_inplace(p, seed, r)
    pool = _make_pool(B, T, dev, seed)
    sub_fwd = _time_ms(lambda inp: _fwd_call(sub_fn, inp), pool, refresh=ref)
    sub_fwdbwd = None
    if bwd_ok:
        pool_bw = _make_pool(B, T, dev, seed, requires_grad=True)
        sub_fwdbwd = _time_ms(lambda inp: _fwdbwd_step(sub_fn, inp, grad_out), pool_bw, refresh=ref)

    # ---- 新鲜度复验：同 data_ptr 填新值再验前向一次（抓内容缓存/快照类作弊）----
    fresh_rel = float("nan")
    if C.FRESHNESS_RECHECK and fwd_ok:
        fi = pool[0]
        _refill_inplace(fi, seed + 999983)
        o_gold_f = reference_output(fi)
        with torch.no_grad():
            # Preserve the timing buffer's data_ptr so pointer-keyed stale caches
            # are exercised against new values and cannot escape this check.
            o_raw = _fwd_call(sub_fn, fi)
        if _valid_output(o_raw, B, T):
            fresh_rel = _rel_l2(o_raw.float(), o_gold_f).item()
            if not (fresh_rel < fwd_thresh):
                fwd_ok = False
        else:
            fwd_ok = False

    fwd_speedup = (base_fwd / sub_fwd) if fwd_ok else 0.0
    fwdbwd_speedup = (base_fwdbwd / sub_fwdbwd) if (fwd_ok and bwd_ok and sub_fwdbwd) else 0.0
    return {
        "B": B, "T": T, "fwd_rel": fwd_rel, "fwd_cos": fwd_cos, "fwd_ok": fwd_ok,
        "fresh_rel": fresh_rel, "bwd_rel": bwd_rel, "bwd_ok": bwd_ok,
        "base_fwd": base_fwd, "sub_fwd": sub_fwd, "fwd_speedup": fwd_speedup,
        "base_fwdbwd": base_fwdbwd, "sub_fwdbwd": (sub_fwdbwd or float("nan")),
        "fwdbwd_speedup": fwdbwd_speedup,
    }


def eval_correctness_only(sub_fn, B, T, seed):
    """Check a random case without running either baseline or submission timing."""
    dev = "cuda"
    fwd_thresh, bwd_thresh, _, o_gold0, inp0 = _shape_thresholds(B, T, seed, dev)
    fwd_rel, fwd_cos, fwd_ok = 0.0, 1.0, True
    for i in range(C.NUM_CORRECT_SEEDS):
        if i == 0:
            inp, o_gold = inp0, o_gold0
        else:
            inp = make_inputs(B, T, device=dev, seed=seed + i * 10007)
            o_gold = reference_output(inp)
        with torch.no_grad():
            o_raw = _fwd_call(sub_fn, _clone_inputs(inp))
        if not _valid_output(o_raw, B, T):
            fwd_ok = False
            continue
        rel = _rel_l2(o_raw.float(), o_gold).item()
        cos = _cosine(o_raw.float(), o_gold)
        fwd_rel, fwd_cos = max(fwd_rel, rel), min(fwd_cos, cos)
        if not (rel < fwd_thresh and cos > C.COSINE_MIN):
            fwd_ok = False

    grad_out = _grad_out(B, T, dev, seed)
    bwd_rel, bwd_ok = float("nan"), False
    try:
        inp_g = make_inputs(B, T, device=dev, seed=seed, requires_grad=True)
        o_g = _fwd_call(sub_fn, inp_g)
        if torch.is_tensor(o_g) and o_g.requires_grad and _valid_output(o_g, B, T):
            o_g.backward(grad_out)
            if all(inp_g[name].grad is not None for name in ("q", "k", "v", "beta")):
                inp_ref = make_inputs(B, T, device=dev, seed=seed, requires_grad=True)
                baseline_chunk(
                    inp_ref["q"], inp_ref["k"], inp_ref["v"], inp_ref["g"],
                    inp_ref["beta"], SCALE,
                ).backward(grad_out)
                bwd_rel = max(
                    _rel_l2(inp_g[name].grad.float(), inp_ref[name].grad.float()).item()
                    for name in ("q", "k", "v", "beta")
                )
                bwd_ok = bwd_rel < bwd_thresh
    except Exception:
        bwd_rel, bwd_ok = float("nan"), False

    return {
        "B": B,
        "T": T,
        "seed": seed,
        "fwd_rel": fwd_rel,
        "fwd_cos": fwd_cos,
        "fwd_threshold": fwd_thresh,
        "fwd_ok": fwd_ok,
        "bwd_rel": bwd_rel,
        "bwd_threshold": bwd_thresh,
        "bwd_ok": bwd_ok,
    }


def do_make_baseline(path, seed):
    """只测 baseline 并写档。此进程不 import 任何选手代码 ⇒ baseline 天然免投毒。"""
    fp = _machine_fingerprint()
    print(f"[make-baseline] 生成统一 baseline（进程内无选手代码）\nGPU: {fp['gpu']}\n")
    shapes = {}
    for (B, T, w) in C.SHAPES:
        bf, bfb = measure_baseline(B, T, seed)
        shapes[f"{B}x{T}"] = {"base_fwd": bf, "base_fwdbwd": bfb}
        print(f"  {B}x{T:<6}  fwd={bf:.4f}ms  fwd+bwd={bfb:.4f}ms")
    with open(path, "w") as f:
        json.dump({"fingerprint": fp, "seed_note": "时延与随机值无关，可跨 seed 复用", "shapes": shapes}, f, indent=2)
    print(f"\n[make-baseline] 已写档 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission")
    ap.add_argument("--baseline", help="用统一 baseline 存档做分母（组织方复测，带机器指纹校验）")
    ap.add_argument("--make-baseline", metavar="PATH", help="只测 baseline 并写档，不评选手")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--json-out", type=Path, help="由评测程序生成 JSON 结果文件")
    ap.add_argument("--random-correctness", action="store_true",
                    help="运行不计性能的随机正确性测试")
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--random-count", type=int, default=5)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "需要 CUDA（单张 4090）"

    if args.random_count <= 0:
        ap.error("--random-count must be positive")

    if args.make_baseline:
        do_make_baseline(args.make_baseline, args.seed)
        return

    assert args.submission, "需要 --submission，或用 --make-baseline 生成统一 baseline"
    fp = _machine_fingerprint()
    print(f"GPU: {fp['gpu']}   seed={args.seed}")

    if args.random_correctness:
        random_seed = args.random_seed if args.random_seed is not None else secrets.randbits(63)
        rng = random.Random(random_seed)
        sub_fn = _load_submission(args.submission)
        rows = []
        print(f"[随机正确性] seed={random_seed}，不执行性能计时")
        for index in range(args.random_count):
            B = rng.choice((1, 2, 4))
            T = rng.choice((257, 511, 1024, 2048))
            case_seed = rng.randrange(1, 2**31)
            row = eval_correctness_only(sub_fn, B, T, case_seed)
            rows.append(row)
            print(
                f"  R{index + 1} {B}x{T:<5} "
                f"fwd={'PASS' if row['fwd_ok'] else 'FAIL'} "
                f"bwd={'PASS' if row['bwd_ok'] else 'FAIL'}"
            )
        passed = all(row["fwd_ok"] and row["bwd_ok"] for row in rows)
        _write_json(args.json_out, {
            "schema_version": 2,
            "problem": "gdn_ako",
            "report_type": "correctness",
            "environment": _environment(),
            "random_seed": random_seed,
            "passed": passed,
            "cases": rows,
        })
        if not passed:
            raise SystemExit(1)
        return

    baseline_cache = None
    if args.baseline:
        with open(args.baseline) as f:
            cached = json.load(f)
        if cached.get("fingerprint") != fp:
            raise SystemExit(
                "❌ baseline 存档的机器指纹与当前机器不一致，拒绝使用（跨机器 baseline 会让比值失真）。\n"
                f"   存档 = {cached.get('fingerprint')}\n   当前 = {fp}\n"
                "   请在复测机上 `--make-baseline` 重新生成。")
        # fail-closed：每个配置形状都必须在存档里且字段是数值——缺一即拒，绝不静默当场测（否则口径不一）
        miss = []
        for (B, T, w) in C.SHAPES:
            ent = cached.get("shapes", {}).get(f"{B}x{T}")
            if not ent or not all(isinstance(ent.get(k), (int, float)) for k in ("base_fwd", "base_fwdbwd")):
                miss.append(f"{B}x{T}")
        if miss:
            raise SystemExit(f"❌ baseline 存档缺形状或字段非法：{miss}；请在复测机重新 --make-baseline。")
        baseline_cache = cached["shapes"]
        print(f"[baseline] 用存档 {args.baseline}（统一分母 · 复测口径 · 主变量=选手 kernel）\n")
    else:
        print("[baseline] 当场测（选手自测口径 · 环境无关比值 · 跨选手不保证可比）\n")

    sub_fn = _load_submission(args.submission)
    print("[容差] 每个形状按「本形状 fla chunk(bf16) vs golden(fp32) 的差距」自校准，"
          f"下限 fwd={C.FWD_REL_L2_FLOOR:.0e} / bwd={C.BWD_REL_L2_FLOOR:.0e}\n")

    rows, wsum, index_numerator = [], 0.0, 0.0
    print(f"{'B':>3} {'T':>6} | {'fwd_relL2':>9} {'fresh_relL2':>11} {'bwd_relL2':>9} {'fwd?':>4} {'bwd?':>4}"
          f" | {'fwd×':>6} {'fwd+bwd×':>8}")
    print("-" * 84)
    for (B, T, w) in C.SHAPES:
        r = eval_shape(sub_fn, B, T, args.seed, baseline_cache)
        rows.append((r, w))
        shape_index = C.SCORE_W_FWD * r["fwd_speedup"] + C.SCORE_W_FWDBWD * r["fwdbwd_speedup"]
        index_numerator += w * shape_index
        wsum += w
        print(f"{B:>3} {T:>6} | {r['fwd_rel']:>9.2e} {r['fresh_rel']:>11.2e} {r['bwd_rel']:>9.2e} "
              f"{'PASS' if r['fwd_ok'] else 'FAIL':>4} {'PASS' if r['bwd_ok'] else 'FAIL':>4}"
              f" | {r['fwd_speedup']:>6.2f} {r['fwdbwd_speedup']:>8.2f}")

    performance_index = index_numerator / wsum
    all_fwd_ok = all(r["fwd_ok"] for r, _ in rows)
    print("-" * 84)
    print(f"\n【正确性】前向全部通过：{all_fwd_ok}   "
          f"（反向全部通过：{all(r['bwd_ok'] for r, _ in rows)}）")
    print(f"【性能】加权性能指数（相对 fla 基线的等效加速比）= {performance_index:.3f}")
    print("       >1.0 = 比 fla 官方 kernel 快；正确性未过的形状该项计 0。")
    _write_json(args.json_out, {
        "schema_version": 2,
        "problem": "gdn_ako",
        "report_type": "correctness_and_performance",
        "environment": _environment(),
        "seed": args.seed,
        "baseline_source": str(args.baseline) if args.baseline else "measured_in_process",
        "all_forward_passed": all_fwd_ok,
        "all_backward_passed": all(r["bwd_ok"] for r, _ in rows),
        "performance_index": performance_index,
        "cases": [{**r, "weight": w} for r, w in rows],
    })


if __name__ == "__main__":
    main()
