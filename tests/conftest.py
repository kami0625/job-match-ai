"""pytest 全局配置：路径引导 + 公共 fixtures"""
import sys
from pathlib import Path

# 项目根目录加入 sys.path（pytest 从 tests/ 运行时仍可 import app.*）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
