"""统一已保存PNG计分；保存旧浮点确定性预测统计，避免混合量化口径。"""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/evaluation_withdrawal_v2')


def main():
    for name in ['scene0017','scene0029']:
        folder=ROOT/name;out=folder/'measurement_evaluation';path=out/'summary.json'
        report=json.loads(path.read_text())
        if report.get('png_quantization_aligned'):continue
        shutil.copy2(path,out/'summary_before_png_alignment.json')
        totals={m:{k:dict(pixels=0,abs_sum=0.,square_sum=0.) for k in ['all','boundary','transition','core']} for m in report['methods']}
        for row in report['rows']:
            raw=row['raw'];data=np.load(out/f'{raw:03d}_reference.npz');valid=data['valid'];layer=data['layer']
            target=np.array(Image.open(folder/'target_rgb'/f'{raw:03d}.png'),dtype=float)/255.
            images={'zero_masked_input':folder/'input_rgb'/f'{raw:03d}.png',
                    'input_plane_neighbor_color':out/'deterministic_rgb'/f'{raw:03d}.png',
                    'fixed_propainter':folder/'propainter/model_png'/f'{raw:03d}.png'}
            row['metrics']={}
            for method,prediction in images.items():
                rgb=np.array(Image.open(prediction),dtype=float)/255.;error=rgb-target
                row['metrics'][method]={}
                for key,mask in [('all',valid),('boundary',layer==1),('transition',layer==2),('core',layer==3)]:
                    count=int(mask.sum())
                    if not count:continue
                    a=float(np.abs(error[mask]).sum());s=float((error[mask]**2).sum())
                    row['metrics'][method][key]={'pixels':count,'MAE':a/(count*3),'PSNR':-10*np.log10(max(s/(count*3),1e-12))}
                    total=totals[method][key];total['pixels']+=count;total['abs_sum']+=a;total['square_sum']+=s
            # 可信参考必须在原始输入被遮挡的位置；输入不得含目标颜色。
            assert not np.array(Image.open(images['zero_masked_input']))[valid].any()
            if raw==report['rows'][24]['raw']:
                rgb=np.uint8(target*255);overlay=rgb.copy()
                for label,color in [(1,[255,210,50]),(2,[30,220,240]),(3,[250,70,180])]:
                    overlay[layer==label]=color
                Image.fromarray(overlay).save(out/'reference_layers.png')
        for method,layers in totals.items():
            for key,total in layers.items():
                n=total['pixels'];total['MAE']=total['abs_sum']/(3*n) if n else None;total['PSNR']=-10*np.log10(max(total['square_sum']/(3*n),1e-12)) if n else None
        report.update(methods=totals,png_quantization_aligned=True,requested_world_rectangle_area_m2=9.6,
                      requested_core_area_m2=2.25,frame_summed_generation_mask_pixels=sum(r['masked_pixels'] for r in report['rows']),
                      frame_summed_reference_pixels=sum(r['sparse_reference_pixels'] for r in report['rows']),
                      frame_summed_core_reference_pixels=sum(r['core_pixels'] for r in report['rows']),
                      denominator_note='cumulative frame pixels; repeated surfaces are not unique world area. Mask includes padding/non-road unknown, not dense road GT.')
        path.write_text(json.dumps(report,indent=2));print(name,{m:v['core'] for m,v in totals.items()},flush=True)


if __name__=='__main__':main()
