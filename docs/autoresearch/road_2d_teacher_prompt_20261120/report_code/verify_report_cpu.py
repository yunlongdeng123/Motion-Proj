"""无需浏览器的实际HTML、内嵌60矩阵、统计与本地链接验收。"""
from html.parser import HTMLParser
from pathlib import Path
import argparse
import json
import re


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.refs=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ("href","src") and value:
                self.refs.append((tag,value))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    full=(root/"index.html").read_text(encoding="utf-8")
    publication=(root/"publication.html").read_text(encoding="utf-8")
    match=re.search(r'window\.REPORT_DATA=(.*?);</script>',full,re.S)
    assert match,"embedded report payload missing"
    data=json.loads(match.group(1))
    assert len(data["cases"])==6 and data["seeds"]==[101,202,303,404,505]
    assert len(data["per_seed"])==120
    assert len(data["summary"]["paired_prompt_deltas"])==60
    paths=set();matrix=[]
    for case in data["cases"]:
        keys={(j["prompt_id"],j["seed"]) for j in case["jobs"]}
        assert len(case["jobs"])==len(keys)==10
        assert keys=={(p,s) for p in ("legacy","material_match_v1") for s in data["seeds"]}
        for job in case["jobs"]:
            matrix.append((case["id"],job["prompt_id"],job["seed"]))
            for key in ("raw_model","raw_roi","final_roi","full_final_preview"):
                paths.add(data["asset_base"]+"/"+job[key])
        paths.update(data["asset_base"]+"/"+value for value in case["reference"].values())
        paths.update(data["asset_base"]+"/"+value for value in case["masks"].values())
    assert len(matrix)==len(set(matrix))==60
    for name,content in (("index.html",full),("publication.html",publication)):
        parser=Links();parser.feed(content)
        for tag,ref in parser.refs:
            if not ref.startswith(("http:","https:","data:","#","mailto:")):
                assert (root/ref).is_file(),(name,ref)
        if name=="publication.html":
            assert not any(ref.startswith("assets/") for _,ref in parser.refs),"publication depends on full images"
        assert "RGB MAE /255" not in content
        assert "accepted_method_current_cases" in content or name=="publication.html"
        assert "64" in content and "050_v1" in content and "7/20" in content and "8/20" in content
    for path in paths:
        assert (root/path).is_file(),path
    assert "let sid=\"scene065_raw110\",seed=101" in full
    assert "RGB MAE（0–255）" in full and "RGB MAE（0–255）" in publication
    assert "raw教师解码" in full and "W外逐位复制" in publication
    assert "只能评价整体prompt" in full
    result=dict(schema="road_2d_teacher_prompt_report_CPU_QA.v1",passed=True,case_count=6,candidate_matrix=60,
        paired_seed_rows=120,dynamic_asset_links=len(paths),all_dynamic_files_exist=True,static_links_passed=True,
        publication_without_60_image_dependency=True,default_case="scene065_raw110",default_seed=101,
        color_units="RGB MAE0-255",new_prompt_promoted=False,GPU_executed=False,browser_UI_QA="root-owned")
    a.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
