"""完整r7结束后的同协议补充取证；复用六方法分析器。"""
from pathlib import Path
import forensic_worldsim_v73_paper_surfaces as audit
audit.MODELS={'Joint-r7':'WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7'}
audit.OUT=audit.ROOT/'docs/autoresearch/worldsim_v73/paper_forensics/20260909T190600Z__saved-r7-surface-oracles-r1'
audit.main()
