import argparse
import json
from pathlib import Path


def escape_latex(text):
    return text.translate(
        str.maketrans(
            {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }
        )
    )


def format_metric(stats):
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None:
        return "--"
    if std is None:
        return f"${mean:.3g}$"
    return rf"${mean:.3g} \pm {std:.3g}$"


def make_table(results):
    metrics = list(
        dict.fromkeys(name for result in results.values() for name in result["metrics"])
    )
    columns = "l" + "r" * len(metrics)
    row_end = r" \\"
    headers = ["Optimizer"] + [
        escape_latex(name.replace("_", " ").capitalize()) for name in metrics
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\hline",
        " & ".join(headers) + row_end,
        r"\hline",
    ]
    for optimizer, result in results.items():
        cells = [escape_latex(optimizer)]
        cells.extend(format_metric(result["metrics"].get(name, {})) for name in metrics)
        lines.append(" & ".join(cells) + row_end)
    lines.extend([r"\hline", r"\end{tabular}%", "}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Write a LaTeX table of final metrics."
    )
    parser.add_argument(
        "data", type=Path, help="Experiment directory or final_metrics.json"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output .tex file")
    args = parser.parse_args()
    source = args.data / "final_metrics.json" if args.data.is_dir() else args.data
    table = make_table(json.loads(source.read_text(encoding="utf-8")))
    if args.output is None:
        print(table, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
