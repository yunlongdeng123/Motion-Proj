from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read
R=Path('/root/autodl-tmp/motion_proj_v77')

def main():
    state=read(O/'training/state.json');assert state['stage']=='complete' and state['steps']==160
    e=R/'docs/autoresearch/worldsim_v77/target_protected_20260929/r8'
    for name in ['state','config','backward_probe','input_contract','same_recipe_as_r7','validation_base','validation_finetuned']:
        shutil.copy2(O/'training'/(name+'.json'),e/('training_'+name+'.json'))
    shutil.copy2(O/'input_distribution.json',e/'input_distribution.json')
    p=R/'docs/RESEARCH_STATUS.md';s=p.read_text();s=s.replace('下一步同r7有效初始化','已完成160步，严格恢复0 missing/0 unexpected、80有限非零梯度、峰值10.48GiB；训练配方逐字段与r7一致。同r7有效初始化').replace('真实原/r7对照已完成，等待本轮训练及合成/真实三臂完整采样和HTML','真实原/r7对照已完成，正在完成7合成+8真实的三臂采样和HTML；不以latent loss下降认证效果');p.write_text(s)
    p=R/'docs/v77/TARGET_PROTECTED_DATA_CONTROL_R8.md';s=p.read_text()
    if '## 同预算训练实际执行' not in s:s+='\n\n## 同预算训练实际执行\n\n160步完成，80/80张量有限非零梯度；严格0缺失/0额外、官方106目标encoder恢复，峰值10.48GiB；实际清零在resize前，隐藏X/Y变动都不进入条件。配方所有冻结字段逐项等于r7。固定teacher-noised验证loss原0.06826→0.06246（−8.49%），并非采样图像质量。全部checkpoint/optimizer快照保留，不增加步数。\n\n实际1600训练帧按step计：r8洞占画面平均1.62%，被遮真实保护B平均0.294%（r7为0.187%）；真实DEV生成mask平均5.60%、矩形fill=1，48.75%帧触边。合成训练车辆轮廓fill≈0.817、无触边。这是输入差异而非已证因果，继续固定本轮对照。完整[输入盘点](../autoresearch/worldsim_v77/target_protected_20260929/r8/input_distribution.json)。\n'
    p.write_text(s)
    p=R/'docs/EXPERIMENTS.md';s=p.read_text().replace('同80张量160步，两套三臂评测待完成','同80张量160步已完成，两套三臂采样进行中');p.write_text(s)
    p=R/'docs/research_failures/entries/V77-F02.md';s=p.read_text().replace('原/r7真实对照已运行，新数据训练及完整合成/真实三臂仍待完成，效果不提前填写','原/r7真实对照已运行，新数据160步同配方已完成（严格0missing/80梯度）；完整合成/真实三臂采样进行中，效果不提前填写');p.write_text(s)
    print('sampling checkpoint saved')
if __name__=='__main__':main()
