#!/usr/bin/env python3
"""
Generate performance report from benchmark results.
Updates docs/PERFORMANCE.md with latest benchmark statistics.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_benchmarks(output_json: Path) -> None:
    """Run pytest-benchmark and save results to JSON."""
    print(f"Running benchmarks and saving to {output_json}...")
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/benchmarks/test_fast_perf.py",
        "--benchmark-only",
        f"--benchmark-json={output_json}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Benchmarks failed!")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)


def generate_markdown(data: dict) -> str:
    """Generate Markdown report from benchmark data."""
    report = []
    report.append("# Playchitect Performance Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    env = data.get("machine_info", {})
    report.append("\n## Environment")
    report.append(f"- **CPU**: {env.get('processor', 'Unknown')}")
    report.append(f"- **Cores**: {env.get('cpu_count', 'Unknown')}")
    report.append(f"- **OS**: {env.get('system', 'Unknown')} {env.get('release', '')}")
    report.append(f"- **Python**: {env.get('python_version', 'Unknown')}")

    report.append("\n## Benchmark Results")
    report.append(
        "| Test | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | StdDev (ms) | Ops/sec |"
    )
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for bench in data.get("benchmarks", []):
        name = bench["name"].split("::")[-1]
        stats = bench["stats"]
        m = 1000.0
        report.append(
            f"| {name} | {stats['mean'] * m:.2f} | {stats['median'] * m:.2f} | "
            f"{stats['min'] * m:.2f} | {stats['max'] * m:.2f} | "
            f"{stats['stddev'] * m:.2f} | {stats['ops']:.2f} |"
        )

    return "\n".join(report)


def main() -> None:
    repo_root = Path(__file__).parent.parent
    output_json = repo_root / ".benchmarks" / "latest_results.json"
    output_json.parent.mkdir(exist_ok=True)

    perf_md = repo_root / "docs" / "PERFORMANCE.md"

    run_benchmarks(output_json)

    with open(output_json) as f:
        data = json.load(f)

    markdown = generate_markdown(data)

    with open(perf_md, "w") as f:
        f.write(markdown)

    print(f"Successfully updated {perf_md}")


if __name__ == "__main__":
    main()
