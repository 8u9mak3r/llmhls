import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torchao.quantization as aoq

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_DIR = Path(__file__).resolve().parent
BUDDY_OPT = REPO_ROOT / "buddy-mlir/build/bin/buddy-opt"
SCALEHLS_OPT = REPO_ROOT / "build/bin/scalehls-opt"
SCALEHLS_TRANSLATE = REPO_ROOT / "build/bin/scalehls-translate"
TOSA_TO_LINALG_PIPELINE = (
    "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, "
    "tosa-to-tensor, tosa-to-arith, canonicalize))"
)
sys.path[:0] = [
    str(REPO_ROOT / "buddy-mlir/build/python_packages"),
    str(REPO_ROOT / "buddy-mlir/llvm/build/tools/mlir/python_packages/mlir_core"),
]

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp

import config
from modules.layers import Block


def generate_hls_cpp(hls_cpp_output_dir):
    subgraph_path = SAMPLE_DIR / "subgraph0.mlir"
    decoder_path = SAMPLE_DIR / "decoder.mlir"
    subprocess.run(
        [
            str(BUDDY_OPT),
            str(subgraph_path),
            "-pass-pipeline",
            TOSA_TO_LINALG_PIPELINE,
            "-o",
            str(decoder_path),
        ],
        check=True,
        cwd=SAMPLE_DIR,
    )

    decoder_ir = decoder_path.read_text(encoding="utf-8")
    decoder_ir = decoder_ir.replace("arith.maximumf", "arith.maxf")
    decoder_ir = decoder_ir.replace("subgraph0", "forward")
    decoder_path.write_text(decoder_ir, encoding="utf-8")

    output_dir = Path(hls_cpp_output_dir)
    if not output_dir.is_absolute():
        output_dir = SAMPLE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_command = [
        str(SCALEHLS_OPT),
        str(decoder_path),
        "-scaleflow-pytorch-pipeline=top-func=forward start-point=2 debug-point=0",
    ]
    translate_command = [
        str(SCALEHLS_TRANSLATE),
        "-scalehls-emit-hlscpp",
        f"-hls-cpp-output-dir={output_dir}",
    ]
    with subprocess.Popen(
        opt_command, stdout=subprocess.PIPE, cwd=SAMPLE_DIR
    ) as opt_process:
        assert opt_process.stdout is not None
        with subprocess.Popen(
            translate_command,
            stdin=opt_process.stdout,
            cwd=SAMPLE_DIR,
        ) as translate_process:
            opt_process.stdout.close()
            translate_returncode = translate_process.wait()
        opt_returncode = opt_process.wait()

    if opt_returncode != 0:
        raise subprocess.CalledProcessError(opt_returncode, opt_command)
    if translate_returncode != 0:
        raise subprocess.CalledProcessError(translate_returncode, translate_command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w8only", action="store_true", required=True)
    parser.add_argument(
        "--hls-cpp-output-dir",
        required=True,
        help="Directory for the generated HLS C++ design",
    )
    args = parser.parse_args()

    torch.manual_seed(3407)
    layer = Block(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        max_len=config.max_len,
        dropout=config.dropout,
        dtype=torch.float16,
    ).eval()
    quantizer = aoq.int8_weight_only()
    aoq.quantize_(layer, quantizer)

    hidden_states = torch.randn(
        1,
        16,
        config.hidden_size,
        dtype=torch.float16,
    )
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = compiler.importer(layer, hidden_states)

    assert len(graphs) == 1
    graph = graphs[0]
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(SAMPLE_DIR / "subgraph0.mlir", "w") as module_file:
        print(driver.subgraphs[0]._imported_module, file=module_file)

    generate_hls_cpp(args.hls_cpp_output_dir)


if __name__ == "__main__":
    main()
