"""本地交付文件与远端实际解码记录一致性核验，不冒称浏览器测试。"""
import json
import argparse
import re
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

parser_args = argparse.ArgumentParser()
parser_args.add_argument("--review-dir", type=Path, required=True)
ROOT = parser_args.parse_args().review_dir.resolve()
data = json.loads((ROOT / "review_manifest.json").read_text(encoding="utf-8"))
assert len(data["clips"]) == 70
assert len({c["scene"] for c in data["clips"]}) == 45
videos = [v for c in data["clips"] for v in c["videos"].values()]
assert len(videos) == 280 and sum(v["decoded_frames"] for v in videos) == 7280
for video in videos:
    assert (ROOT / video["path"]).stat().st_size == video["bytes"]
    assert video["decoded_frames"] == 26 and video["fps"] == 10
    assert (ROOT / video["poster"]).is_file()
for c in data["clips"]:
    assert c["human_verdict"] is None
    assert c["assistant_one_frame_issue"] != "unreviewed"
    assert c["input_qualification"] != "pending_assistant_one_frame_review"
    assert (ROOT / c["one_frame_review"]).is_file()
    assert (ROOT / c["one_frame_components"]).is_file()


class AuditHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.videos = 0
        self.cards = 0
        self.references = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.videos += tag == "video"
        self.cards += tag == "article" and a.get("class") == "case"
        for k in ["src", "data-src", "poster", "href"]:
            val = a.get(k, "")
            if val and not val.startswith(("#", "http", "data:")):
                self.references.append(val)


html = (ROOT / "index.html").read_text(encoding="utf-8")
parser = AuditHTML()
parser.feed(html)
assert parser.videos == 280 and parser.cards == 70
assert all((ROOT / ref).is_file() for ref in parser.references)
assert not re.search(r"__[A-Z_]+__", html)
script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
js_path = Path(__file__).parent / "audit_page_syntax_check.js"
js_path.write_text(script, encoding="utf-8")
subprocess.run(["node", "--check", str(js_path)], check=True, capture_output=True)
out = {"task_id": "WS-V77-DELETE-AUDIT-20260928", "run_id": "r1", "local_cards": parser.cards,
       "local_video_links": parser.videos, "checked_local_references": len(parser.references),
       "videos_matching_remote_bytes": len(videos), "remote_actual_decoded_frames": 7280,
       "assistant_reviewed_fixed_frames": 70, "human_scores_filled": 0,
       "html_placeholders_remaining": 0, "browser_ui_verified": False,
       "javascript_syntax_check": "node --check passed",
       "browser_ui_limit": "Browser use URL policy blocked file URL; no alternate browser workaround attempted.",
       "single_frame_counts": dict(Counter(c["assistant_one_frame_issue"] for c in data["clips"]))}
(ROOT / "delivery_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
