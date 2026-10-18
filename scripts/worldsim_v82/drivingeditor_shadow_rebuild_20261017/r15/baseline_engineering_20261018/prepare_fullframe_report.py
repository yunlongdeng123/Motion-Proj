"""Retain original full-frame videos; remove ROI viewport and crop export UI."""
from pathlib import Path
import re,shutil
W=Path(__file__).parent;I=W.parent/'identity_completion';O=W.parents[1]/'outputs/identity_comparison_20261018'
def fullframe(s):
 s=s.replace('<label><input id="zoom" type="checkbox" checked>局部放大</label>','')
 s=s.replace("box=$('zoom').checked?DATA.rois[$('actor').value][frame-89]:[0,0,480,320]","box=[0,0,e.video.videoWidth,e.video.videoHeight]")
 s=s.replace("$('zoom').onchange=()=>entries.forEach(e=>draw(e));",'')
 s=re.sub(r'<a id="export".*?</a>。','',s)
 s=re.sub(r"\s*\$\('export'\)\.href=`[^`]*`;",'',s)
 s=s.replace('局部放大只改变显示窗口，所有原视频都是完整整帧；所选车辆不在画面时显示全图。','全部视频与显示窗口均使用完整画面，无ROI裁剪或局部放大。')
 return s
def main():
 src=I/'sync_player_template.html';backup=src.with_suffix('.before_fullframe.html')
 if not backup.exists():shutil.copy2(src,backup)
 s=fullframe(src.read_text(encoding='utf-8'));src.write_text(s,encoding='utf-8')
 for name in ['index.html','sync_player.html']:
  p=O/name;s=fullframe(p.read_text(encoding='utf-8'));p.write_text(s,encoding='utf-8')
  assert 'id="zoom"' not in s and "$('zoom')" not in s and 'id="export"' not in s
 # Separate reusable final-results template includes real future models only
 # after their complete49 receipt has arrived.
 s=src.read_text(encoding='utf-8')
 s=s.replace("choices=[['old5,r15'", "choices=[['matched5,old5','新五身份 与 历史旧5'],['matched5,repair4k','新五身份 与 修复R15'],['old5,r15'")
 s=s.replace('冻结检查点</text>','最终检查点</text>').replace('旧5 / 原R15 / 修复R15','新五身份 / 历史对照')
 (W/'fullframe_player_template.html').write_text(s,encoding='utf-8')
 print('FULL_FRAME_DISPLAY_AND_SOURCE_VIDEOS_ONLY; no media re-encoding')
if __name__=='__main__':main()
