"""测试：源码、脚本与测试文件的头部注释规范（本项目统一的三段式约定）。

对应开发文档
    §20.2 推荐目录（模块清单）、§28 文档维护规则（变更需可追溯）。

职责
    用可执行的断言替代人工目视检查，保证每个 Python 文件头部都带有可追溯的注释：
        - ``src/**``：必须包含「对应开发文档」、职责、不做（边界）三段；
        - ``scripts/*``：必须包含「对应开发文档」与用法说明；
        - ``tests/*``：必须包含「对应开发文档」。

不做（边界）
    - 不校验注释内容的正确性（那属于代码评审），只校验结构与可追溯标记是否齐备。
"""

from __future__ import annotations

import ast
import pathlib
import unittest

#: 项目根目录（tests/ 的上一级）。
ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]

#: src 模块头部必须出现的三段标记。
SRC_REQUIRED_MARKERS: tuple[str, ...] = ("对应开发文档", "职责", "不做（边界）")
#: 脚本头部必须出现的标记（脚本以“用法”替代“不做”）。
SCRIPT_REQUIRED_MARKERS: tuple[str, ...] = ("对应开发文档", "用法")
#: 测试文件头部必须出现的标记。
TEST_REQUIRED_MARKERS: tuple[str, ...] = ("对应开发文档",)


#: 本节交付的入口脚本（§20.2）。目录中可能另有研究者自用脚本，不在此列。
DELIVERED_SCRIPTS: tuple[str, ...] = (
    "extract_features.py",
    "build_context.py",
    "train_detector.py",
    "evaluate.py",
    "run_agent.py",
)


def module_docstring(path: pathlib.Path) -> str:
    """读取模块级 docstring。

    Args:
        path: Python 文件路径。

    Returns:
        模块 docstring；不存在时返回空字符串。
    """
    return ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""


class HeaderCommentTest(unittest.TestCase):
    """头部注释结构检查。"""

    def _assert_markers(self, paths: list[pathlib.Path], markers: tuple[str, ...]) -> None:
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                doc = module_docstring(path)
                self.assertTrue(doc, f"{path} 缺少模块级 docstring")
                for marker in markers:
                    self.assertIn(marker, doc, f"{path} 头部注释缺少「{marker}」段")

    def test_src_modules_have_structured_header(self) -> None:
        """src 下所有模块（含各包的 __init__.py）必须具备三段式头部注释。"""
        paths = sorted((ROOT / "src").rglob("*.py"))
        self.assertGreater(len(paths), 20, "src 模块数量异常，请检查目录结构")
        self._assert_markers(paths, SRC_REQUIRED_MARKERS)

    def test_scripts_have_usage_header(self) -> None:
        """交付的 5 个入口脚本必须存在且说明用法（§20.2）。"""
        paths = [ROOT / "scripts" / name for name in DELIVERED_SCRIPTS]
        missing = [path.name for path in paths if not path.exists()]
        self.assertEqual(missing, [], f"缺少入口脚本：{missing}（见 §20.2）")
        self._assert_markers(paths, SCRIPT_REQUIRED_MARKERS)

    def test_tests_reference_doc(self) -> None:
        """测试文件应指明其对应的开发文档条款。"""
        paths = sorted((ROOT / "tests").glob("*.py"))
        self.assertGreaterEqual(len(paths), 5)
        self._assert_markers(paths, TEST_REQUIRED_MARKERS)


if __name__ == "__main__":
    unittest.main()
