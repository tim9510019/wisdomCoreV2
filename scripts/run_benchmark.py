#!/usr/bin/env python3
"""
ARC-AGI-3 公平比較評測框架
==========================
對四個方向做完全相同的評測流程：
  Dir1: Standard LoRA (q+v, r=4) - Online TTT
  Dir2: Hybrid BFS+Gemma4 - LoRA TTT
  Dir3: DNA_HELIX_ULTIMATE - Online TTT
  Dir4: DNA_HELIX_ULTIMATE - Search-to-SFT TTT

規則：
  - 每個方向跑完全相同的 15 個 games
  - 每個 game 相同的 max_steps
  - 每個方向使用獨立的 CUDA device (兩台 Blackwell 各跑兩個方向)
  - 結果收集到同一個 JSON 文件，最後生成比較報告
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

# ===== 方向設定 =====
DIRECTIONS = {
    "dir1_lora":       ROOT / "agent"    / "my_agent_lora.py",
}

GPU_ASSIGNMENT = {
    # GPU 0 (first 96GB Blackwell): Dir1
    "dir1_lora":       "cuda:0",
}


def load_agent_class(agent_path: Path, direction_name: str):
    """從指定路徑動態載入 MyAgent class。"""
    spec = importlib.util.spec_from_file_location(f"agent_{direction_name}", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load agent from {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MyAgent"):
        raise RuntimeError(f"{agent_path} must define MyAgent")
    return module.MyAgent


def run_direction(direction_name: str, agent_path: Path, game_ids: list,
                  max_steps: int, cuda_device: str, results_dir: Path) -> dict:
    """執行單一方向的完整評測，回傳結果 dict。"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting Direction: {direction_name}")
    print(f"   Agent: {agent_path.name}")
    print(f"   Device: {cuda_device}")
    print(f"   Games: {game_ids}")
    print(f"   Max steps: {max_steps}")
    print(f"{'='*60}")

    # NOTE: CUDA_VISIBLE_DEVICES is set by the shell launcher (run_gpu0.sh / run_gpu1.sh).
    # Inside this process, PyTorch always sees the assigned GPU as cuda:0.
    # Do NOT override CUDA_VISIBLE_DEVICES here — it's too late after PyTorch import.

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    MyAgentCls = load_agent_class(agent_path, direction_name)
    if hasattr(MyAgentCls, "MAX_ACTIONS"):
        MyAgentCls.MAX_ACTIONS = max_steps

    direction_results = {
        "direction": direction_name,
        "agent_file": str(agent_path.name),
        "cuda_device": cuda_device,
        "max_steps": max_steps,
        "start_time": datetime.now().isoformat(),
        "games": {}
    }

    total_levels = 0
    total_actions = 0
    wins = 0

    for i, game_id in enumerate(game_ids, 1):
        print(f"\n  [{i}/{len(game_ids)}] {direction_name} → {game_id}")
        game_start = time.time()
        try:
            env = arc.make(game_id)
            if env is None:
                print(f"    ⚠️  Could not create env for {game_id}, skipping")
                direction_results["games"][game_id] = {"error": "env creation failed"}
                continue

            agent = MyAgentCls(
                card_id=f"benchmark-{direction_name}",
                game_id=game_id,
                agent_name=f"{direction_name}.{game_id}",
                ROOT_URL="http://localhost",
                record=False,
                arc_env=env,
                tags=["benchmark", direction_name],
            )
            agent.main()

            final = agent.frames[-1]
            elapsed = time.time() - game_start
            levels = final.levels_completed
            actions = agent.action_counter
            state = str(final.state)
            is_win = "WIN" in state.upper()

            total_levels += levels
            total_actions += actions
            if is_win:
                wins += 1

            game_result = {
                "levels_completed": levels,
                "actions_used": actions,
                "state": state,
                "win": is_win,
                "elapsed_sec": round(elapsed, 1)
            }
            direction_results["games"][game_id] = game_result
            print(f"    ✅ levels={levels}, actions={actions}, state={state}, time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - game_start
            print(f"    ❌ ERROR: {e}")
            direction_results["games"][game_id] = {"error": str(e), "elapsed_sec": round(elapsed, 1)}

    # Summary
    direction_results["end_time"] = datetime.now().isoformat()
    direction_results["summary"] = {
        "total_levels_completed": total_levels,
        "total_actions_used": total_actions,
        "wins": wins,
        "games_played": len(game_ids),
        "win_rate": round(wins / len(game_ids), 4) if game_ids else 0,
        "avg_levels_per_game": round(total_levels / len(game_ids), 2) if game_ids else 0,
        "avg_actions_per_game": round(total_actions / len(game_ids), 1) if game_ids else 0,
    }

    # Save intermediate result
    out_file = results_dir / f"{direction_name}_results.json"
    with open(out_file, "w") as f:
        json.dump(direction_results, f, indent=2)
    print(f"\n  💾 Saved: {out_file.name}")
    print(f"  📊 {direction_name} Summary: levels={total_levels}, wins={wins}/{len(game_ids)}")

    return direction_results


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Fair Comparison Benchmark")
    parser.add_argument("--direction", default="all",
                        help="Which direction(s) to run: all, dir1_lora, dir2_hybrid, dir3_dna_ttt, dir4_dna_search, or comma-separated")
    parser.add_argument("--games", default="all",
                        help="Which games: all, or comma-separated list e.g. ls20,vc33")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max actions per game (default: 300)")
    parser.add_argument("--cuda", default=None,
                        help="Override CUDA device (e.g. cuda:0). Overrides GPU_ASSIGNMENT.")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results"),
                        help="Directory to save results")
    args = parser.parse_args()

    # Prepare output directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve game list
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()
    all_game_ids = [e.game_id.split("-")[0] for e in all_envs]

    if args.games == "all":
        game_ids = all_game_ids
    else:
        game_ids = [g.strip() for g in args.games.split(",")]

    print(f"📋 Games to evaluate: {game_ids}")
    print(f"🎯 Max steps per game: {args.max_steps}")

    # Resolve directions to run
    if args.direction == "all":
        dirs_to_run = list(DIRECTIONS.keys())
    else:
        dirs_to_run = [d.strip() for d in args.direction.split(",")]

    print(f"🔬 Directions: {dirs_to_run}")

    # Run each direction
    all_results = {}
    benchmark_meta = {
        "start_time": datetime.now().isoformat(),
        "games": game_ids,
        "max_steps": args.max_steps,
        "directions_run": dirs_to_run,
    }

    for direction_name in dirs_to_run:
        agent_path = DIRECTIONS.get(direction_name)
        if agent_path is None:
            print(f"⚠️  Unknown direction: {direction_name}")
            continue
        if not agent_path.exists():
            print(f"⚠️  Agent not found: {agent_path}")
            continue

        cuda = args.cuda or GPU_ASSIGNMENT.get(direction_name, "cuda:0")

        result = run_direction(
            direction_name=direction_name,
            agent_path=agent_path,
            game_ids=game_ids,
            max_steps=args.max_steps,
            cuda_device=cuda,
            results_dir=results_dir,
        )
        all_results[direction_name] = result

    # Save combined results
    benchmark_meta["end_time"] = datetime.now().isoformat()
    combined = {"meta": benchmark_meta, "results": all_results}
    combined_path = results_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n\n{'='*60}")
    print("📊 BENCHMARK COMPLETE — COMBINED RESULTS")
    print(f"{'='*60}")
    print(f"{'Direction':<20} {'Wins':>6} {'Levels':>8} {'Avg Steps':>10} {'Win%':>8}")
    print("-"*60)
    for dname, result in all_results.items():
        s = result.get("summary", {})
        wins = s.get("wins", 0)
        games = s.get("games_played", 1)
        levels = s.get("total_levels_completed", 0)
        avg_steps = s.get("avg_actions_per_game", 0)
        win_pct = s.get("win_rate", 0) * 100
        print(f"{dname:<20} {wins:>6}/{games:<3} {levels:>8} {avg_steps:>10.1f} {win_pct:>7.1f}%")
    print(f"{'='*60}")
    print(f"\n💾 Full results saved to: {combined_path}")
    print(f"📈 Run the report generator: python3 scripts/generate_report.py --input {combined_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
