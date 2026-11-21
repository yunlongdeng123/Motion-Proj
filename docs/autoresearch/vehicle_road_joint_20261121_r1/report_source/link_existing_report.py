"""给既有统一道路报告增加车辆分支入口，保留原55条实验记录。"""
from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
NAV="""if(!document.getElementById('vehicle-road-report-entry')){document.querySelector('main').insertAdjacentHTML('afterbegin','<p id="vehicle-road-report-entry" style="padding:12px 16px;border:1px solid #38546f;border-radius:10px"><a href="../vehicle_road_joint_20261121/index.html">车辆与道路联合恢复 · 删除 a9 / 补全 a20</a> · <a href="../vehicle_road_joint_20261121/architecture.svg">更新后的四模块架构图</a></p>');}"""
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    new=ROOT/'outputs/vehicle_road_joint_20261121/index.html';assert new.is_file()
    old=ROOT/'outputs/road_material_three_rounds_20261121/index.html'
    data=old.parent/'surface_residual_r2/report_data.json';before=sha(data)
    assert len(json.loads(data.read_text('utf-8'))['catalog']['models'])==55
    source=ROOT/'work/road_surface_residual_20261121_r2/viewer.js'
    if 'vehicle-road-report-entry' not in source.read_text('utf-8'):
        source.write_text(source.read_text('utf-8')+'\n'+NAV+'\n','utf-8')
    text=old.read_text('utf-8')
    if 'vehicle-road-report-entry' not in text:
        b=W/'old_report_before_vehicle_entry.html'
        if not b.exists():b.write_bytes(old.read_bytes())
        old.write_text(text+'\n<script>'+NAV+'</script>\n','utf-8')
    assert sha(data)==before
    receipt=dict(passed=True,old_records=55,scientific_data_unchanged=True,entry=str(new),source=str(source))
    (W/'report_entry_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n','utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()
