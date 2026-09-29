"""测试 scripts 入口的可调用性：参数解析器构建与 ``--help`` 退出码。

对应开发文档
    §20.2 推荐目录（scripts 下的 5 个入口）、§20.1 技术栈（CLI 入口约定）。

职责
    保证 5 个入口脚本在只读场景下不会崩：``build_parser()`` 可构建，
    ``main(["--help"])`` 以退出码 0 结束（argparse 的标准行为）。

不做（边界）
    - 不测试各脚本的业务流程——它们依赖尚未接入的数据、checkpoint 与 LLM 服务，
      当前会按设计抛出 ``NotImplementedError``；
    - 不做端到端数据回归（尚无数据）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pathlib
import types
import unittest

from scripts import build_context, evaluate, extract_features, run_agent, train_detector

#: 项目根目录与配置文件路径（--dry-run 需要真实存在的配置文件）。
ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH: pathlib.Path = ROOT / "config.yaml"

#: 全部入口脚本模块（与 §20.2 的 scripts 清单一致）。
SCRIPT_MODULES: tuple[types.ModuleType, ...] = (
    extract_features,
    build_context,
    train_detector,
    evaluate,
    run_agent,
)


class ScriptCliTest(unittest.TestCase):
    """入口脚本的参数解析与帮助输出。"""

    def test_all_scripts_expose_build_parser(self) -> None:
        for module in SCRIPT_MODULES:
            with self.subTest(module=module.__name__):
                parser = module.build_parser()
                self.assertIsInstance(parser, argparse.ArgumentParser)
                # description 必须指明对应的开发文档章节，便于 --help 自解释
                self.assertIn("开发文档", parser.description or "")

    def test_help_exits_with_zero(self) -> None:
        for module in SCRIPT_MODULES:
            with self.subTest(module=module.__name__):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as caught:
                        module.main(["--help"])
                self.assertEqual(caught.exception.code, 0)
                self.assertIn("usage:", stdout.getvalue())


class RunAgentDryRunTest(unittest.TestCase):
    """run_agent 的装配自检（--dry-run）必须可端到端跑通。

    该路径不读取真实数据，因此可以在无数据环境下验证 M4→M5→M6→M8 的装配链路：
    注册检测器 → 策略选择 → 执行 → 融合 → 降级摘要。
    """

    def test_dry_run_returns_zero(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = run_agent.main(["--config", str(CONFIG_PATH), "--dry-run"])
        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("已注册检测器", output)
        self.assertIn("融合：", output)

    def test_dry_run_with_fixed_policy(self) -> None:
        """§18.2 的“固定策略”对照路径同样可用。"""
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = run_agent.main(["--config", str(CONFIG_PATH), "--policy", "fixed", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("Level 0", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
