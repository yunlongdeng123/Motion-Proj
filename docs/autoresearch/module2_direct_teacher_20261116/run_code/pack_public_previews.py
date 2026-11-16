"""纯CPU：公共加载器浮点结果→无损RGB8预览与固定物理C ROI五列对照。

不重新渲染、不修改评价域、不做曝光拟合或RGB/alpha。无损指RGB8编码，
源float NPZ保持原样；所有列使用同一clip[0,1]、round(255*x)显示约定。
"""
from pathlib import Path
import argparse,json,re,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont,features

FRAMES=(86,110,134,179,198)
PRODUCTS=('road_only_rgb','rgb')
LABELS=['原始观测（车下无GT）','固定2D教师（C内伪真值）','旧S2','direct','Fcontrol']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def spec_for(value):
    return dict(rendered_dir=value) if isinstance(value,str) else dict(value)


def array_path(spec,raw):
    if 'arrays' in spec:
        mapping=spec['arrays'];value=mapping.get(str(raw),mapping.get(f'{raw:03d}'))
        if value is None:raise ValueError(f'missing explicit arrays path for raw{raw}')
        return Path(value).expanduser().resolve()
    if 'arrays_template' in spec:
        return Path(spec['arrays_template'].format(raw=raw,frame=f'{raw:03d}')).expanduser().resolve()
    return (Path(spec['rendered_dir'])/f'{raw:03d}'/spec.get('array_filename','arrays.npz')).expanduser().resolve()


def load(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


def rgb_float(value,wh,name):
    a=np.asarray(value)
    w,h=wh
    if a.shape==(3,h,w):a=a.transpose(1,2,0)
    if a.shape!=(h,w,3) or not np.isfinite(a).all():raise ValueError(f'{name}: expected finite RGB {(h,w,3)}, got {a.shape}')
    return a


def display(value):
    return np.uint8(np.floor(np.clip(value,0,1)*255+.5))


def image_mask(value,wh,name):
    a=np.asarray(value);w,h=wh
    if a.shape==(1,h,w):a=a[0]
    if a.dtype!=bool or a.shape!=(h,w):raise ValueError(f'{name}: expected bool {(h,w)}')
    return a


def alpha_float(value,wh,name):
    a=np.asarray(value);w,h=wh
    if a.shape==(1,h,w):a=a[0]
    if a.shape!=(h,w) or not np.isfinite(a).all() or a.min()<-1e-5 or a.max()>1+1e-5:
        raise ValueError(f'{name}: expected finite alpha {(h,w)} in [0,1]')
    return a


def save_lossless(path,rgb,verify=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.asarray(rgb,np.uint8),'RGB').save(path,format='WEBP',lossless=True,method=4)
    if verify:
        with Image.open(path) as im:decoded=np.asarray(im.convert('RGB'))
        if not np.array_equal(decoded,rgb):raise RuntimeError('lossless RGB8 verification failed: '+str(path))
    return dict(path=str(path),width=int(rgb.shape[1]),height=int(rgb.shape[0]),bytes=path.stat().st_size,lossless_RGB8_verified=verify)


def fixed_roi(mask,padding):
    y,x=np.where(mask)
    if not len(y):raise ValueError('frozen physical C mask is empty; refuse result-selected fallback ROI')
    h,w=mask.shape
    return [max(int(x.min())-padding,0),max(int(y.min())-padding,0),
            min(int(x.max())+1+padding,w),min(int(y.max())+1+padding,h)]


def font(size,path=None):
    options=[path] if path else []
    options += [str(Path(__file__).with_name('preview_labels.ttf')),
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc','C:/Windows/Fonts/msyh.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for p in options:
        if p and Path(p).is_file():return ImageFont.truetype(p,size)
    return ImageFont.load_default()


def contact_sheet(images,labels,roi,font_path=None):
    x0,y0,x1,y1=roi;w,h=x1-x0,y1-y0
    gutter=8;margin=8;size=13;face=font(size,font_path)
    measure=ImageDraw.Draw(Image.new('RGB',(1,1)))
    wrapped=[]
    for label in labels:
        lines=[];line=''
        for char in label:
            if line and measure.textbbox((0,0),line+char,font=face)[2]>w:
                lines.append(line);line=char
            else:line+=char
        lines.append(line);wrapped.append(lines)
    line_height=size+5;header=8+max(map(len,wrapped))*line_height
    canvas=Image.new('RGB',(2*margin+5*w+4*gutter,2*margin+header+h),(10,17,27))
    draw=ImageDraw.Draw(canvas)
    for index,(rgb,lines) in enumerate(zip(images,wrapped)):
        left=margin+index*(w+gutter)
        for row,label in enumerate(lines):
            draw.text((left,margin+3+row*line_height),label,font=face,fill=(225,235,245))
        canvas.paste(Image.fromarray(rgb[y0:y1,x0:x1]),(left,margin+header))
    return np.asarray(canvas).copy()


def validate_plan(plan):
    if plan.get('schema')!='module2_direct_teacher_delivery_plan.v1':raise ValueError('unsupported plan schema')
    if tuple(plan.get('frames',FRAMES))!=FRAMES:raise ValueError('this delivery requires the fixed five-frame order')
    wh=tuple(plan.get('expected_wh',[480,320]))
    if len(wh)!=2 or any(type(n)is not int or n<=0 for n in wh):raise ValueError('expected_wh must contain positive width,height')
    variants=plan['variants']
    if not variants:raise ValueError('variants cannot be empty')
    for key,value in variants.items():
        if not re.fullmatch('[A-Za-z0-9_-]+',key):raise ValueError('variant IDs must be safe file components')
        spec=spec_for(value)
        if sum(name in spec for name in ['arrays','arrays_template','rendered_dir'])!=1:raise ValueError(key+': exactly one arrays source is required')
    crop=plan.get('crop',{});padding=crop.get('padding_px',8)
    if padding!=8:raise ValueError('ROI contract is frozen physical C bbox + 8 renderer pixels')
    if crop.get('mask_key','C_full_hole')!='C_full_hole':raise ValueError('ROI must use full frozen physical C, not photo/result masks')
    keys=crop.get('variants',['baseline','direct_final','Fcontrol'])
    if len(keys)!=3 or any(k not in variants for k in keys):raise ValueError('crop needs baseline/direct/Fcontrol variant keys')
    fields=crop.get('products',list(PRODUCTS))
    if not fields or any(f not in PRODUCTS for f in fields):raise ValueError('unknown crop product')
    labels=crop.get('labels',LABELS)
    if len(labels)!=5 or not all(isinstance(x,str) and x for x in labels):raise ValueError('crop labels must contain exactly five nonempty strings')
    diag=plan.get('diagnostics',{})
    if diag.get('enabled',False):
        if diag.get('K_reference','baseline') not in variants:raise ValueError('K_reference is absent')
        if not float(diag.get('K_diff_display_max',1/255))>0:raise ValueError('K diff display scale must be positive')
        for raw in FRAMES:
            path=Path(diag['domain_dir'])/diag.get('domain_template','{raw:03d}.npz').format(raw=raw)
            if not path.is_file():raise FileNotFoundError(path)
    inventory=[]
    for key,value in variants.items():
        spec=spec_for(value)
        for raw in FRAMES:
            path=array_path(spec,raw)
            if not path.is_file():raise FileNotFoundError(path)
            inventory.append(dict(variant=key,raw=raw,arrays=str(path)))
    sup=Path(plan['supervision_dir']).expanduser().resolve()
    for raw in [86,110]:
        if not (sup/f'{raw:03d}.npz').is_file():raise FileNotFoundError(sup/f'{raw:03d}.npz')
    return wh,inventory


def pack(plan,output):
    wh,inventory=validate_plan(plan)
    if not features.check('webp'):raise RuntimeError('Pillow needs WebP support; no lossy fallback is allowed')
    output=Path(output).expanduser().resolve()
    if output.exists():raise FileExistsError('preserve prior delivery; select a fresh output directory: '+str(output))
    output.mkdir(parents=True)
    verify=plan.get('verify_lossless',True)
    assets=[];crops=[];diagnostic_assets=[];stats=[];cache={}
    diag=plan.get('diagnostics',{});diag_enabled=diag.get('enabled',False)
    for item in inventory:
        raw,key,path=item['raw'],item['variant'],Path(item['arrays']);spec=spec_for(plan['variants'][key]);data=load(path)
        products={}
        for product in PRODUCTS:
            field=spec.get('array_keys',{}).get(product,product)
            if field not in data:raise KeyError(f'{path}: missing {field}')
            value=rgb_float(data[field],wh,str(path)+':'+field);rgb8=display(value);products[product]=rgb8
            record=save_lossless(output/'frames'/key/product/f'{raw:03d}.webp',rgb8,verify)
            record.update(variant=key,raw=raw,product=product,source_arrays=str(path),source_key=field,
                display_clipped_channel_values=int(((value<0)|(value>1)).sum()))
            assets.append(record)
        cache[key,raw]=products
        if diag_enabled:
            domain_file=Path(diag['domain_dir'])/diag.get('domain_template','{raw:03d}.npz').format(raw=raw)
            masks=load(domain_file);domain=image_mask(masks[diag.get('domain_key','frozen_evaluation_domain')],wh,'domain')
            K=image_mask(masks[diag.get('K_key','K')],wh,'K')
            alpha=alpha_float(data[spec.get('alpha_key',diag.get('alpha_key','road_only_alpha'))],wh,'road_only_alpha')
            gap=domain&(alpha<.05);thin=domain&(alpha>=.05)&(alpha<=.9);support=domain&(alpha>.9)
            classified=np.zeros((wh[1],wh[0],3),np.uint8)
            for mask,color in [(gap,[255,77,103]),(thin,[255,186,73]),(support,[57,198,150])]:classified[mask]=color
            record=save_lossless(output/'diagnostics'/key/'coverage'/f'{raw:03d}.webp',classified,verify)
            record.update(variant=key,raw=raw,kind='fixed_domain_coverage');diagnostic_assets.append(record)
            stats.append(dict(variant=key,raw=raw,domain_pixels=int(domain.sum()),gap_pixels=int(gap.sum()),thin_pixels=int(thin.sum()),supported_pixels=int(support.sum()),K_pixels=int(K.sum())))
    crop=plan.get('crop',{});crop_keys=crop.get('variants',['baseline','direct_final','Fcontrol'])
    labels=crop.get('labels',LABELS)
    sup=Path(plan['supervision_dir'])
    for raw in [86,110]:
        data=load(sup/f'{raw:03d}.npz');mask=image_mask(data['C_full_hole'],wh,'frozen physical C_full_hole')
        roi=fixed_roi(mask,8)
        observed=display(rgb_float(data['target_A_srgb'],wh,'actual observation A'))
        teacher=display(rgb_float(data['target_C_srgb'],wh,'original raw teacher downsample'))
        for product in crop.get('products',list(PRODUCTS)):
            images=[observed,teacher]+[cache[key,raw][product] for key in crop_keys]
            sheet=contact_sheet(images,labels,roi,plan.get('font_path'))
            record=save_lossless(output/'crops'/f'{raw:03d}_{product}_five_way.webp',sheet,verify)
            record.update(raw=raw,product=product,roi_xyxy=roi,roi_mask='r3 C_full_hole; any physical native C footprint',
                roi_mask_pixels=int(mask.sum()),padding_renderer_pixels=8,columns=labels,variant_columns=crop_keys,
                image_resize_performed=False,exposure_adjustment=False)
            crops.append(record)
    if diag_enabled:
        reference=diag.get('K_reference','baseline')
        if reference not in plan['variants']:raise ValueError('K_reference is absent')
        display_max=float(diag.get('K_diff_display_max',1/255))
        if not display_max>0:raise ValueError('K diff display scale must be positive')
        for item in inventory:
            raw,key=item['raw'],item['variant'];spec=spec_for(plan['variants'][key]);refspec=spec_for(plan['variants'][reference])
            current=load(item['arrays']);base=load(array_path(refspec,raw))
            a=rgb_float(current[spec.get('array_keys',{}).get('rgb','rgb')],wh,'joint RGB')
            b=rgb_float(base[refspec.get('array_keys',{}).get('rgb','rgb')],wh,'baseline joint RGB')
            masks=load(Path(diag['domain_dir'])/diag.get('domain_template','{raw:03d}.npz').format(raw=raw))
            K=image_mask(masks[diag.get('K_key','K')],wh,'K');difference=np.abs(a-b).max(2)
            heat=np.zeros((wh[1],wh[0],3),np.uint8);heat[...,0]=display(np.where(K,difference/display_max,0))
            record=save_lossless(output/'diagnostics'/key/'K_diff'/f'{raw:03d}.webp',heat,verify)
            record.update(variant=key,raw=raw,kind='fixed_full_K_float_RGB_difference',reference=reference,display_max=display_max,
                K_pixels=int(K.sum()),K_RGB_max_abs=float(difference[K].max()) if K.any() else 0.,
                K_any_float_difference_pixels=int((K&(difference>0)).sum()),K_gt_1byte_pixels=int((K&(difference>1/255)).sum()))
            diagnostic_assets.append(record)
    manifest=dict(schema='module2_direct_teacher_preview_delivery.v1',plan=plan,source_inventory=inventory,
        frames=list(FRAMES),full_frame_assets=assets,crop_assets=crops,diagnostic_assets=diagnostic_assets,coverage_statistics=stats,
        display_convention='native sRGB values; one shared clip[0,1] and floor(255*x+.5); RGB8 losslessly compressed; source float NPZ not altered',
        road_only_convention='stored native premultiplied road_only_rgb; no RGB/alpha division',
        roi_rule='frozen r3 full physical C bbox plus eight renderer pixels, independent of all compared results',
        observation_column='full original observation; car-occluded road in the crop has no observed GT; A loss uses its separate visible-road mask',
        teacher_column='fixed raw 2D completion; pseudo target inside C, not hidden-road ground truth',
        new_GPU_operations=0,evaluation_masks_modified=False)
    for record in assets+crops+diagnostic_assets:record['path']=str(Path(record['path']).relative_to(output))
    write(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-only',action='store_true');parser.add_argument('--zip',type=Path)
    args=parser.parse_args();plan=read(args.plan)
    if args.validate_only:
        wh,inventory=validate_plan(plan);print(json.dumps(dict(valid=True,expected_wh=wh,sources=inventory),ensure_ascii=False));return
    if not args.output:parser.error('--output is required for packing')
    if args.zip:
        args.zip=args.zip.expanduser().resolve();resolved_output=args.output.expanduser().resolve()
        if args.zip.exists():raise FileExistsError(args.zip)
        if args.zip==resolved_output or resolved_output in args.zip.parents:
            raise ValueError('zip must be outside the output directory')
    result=pack(plan,args.output)
    if args.zip:
        args.zip.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.zip,'w',compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(args.output.rglob('*')):
                if path.is_file():archive.write(path,path.relative_to(args.output))
    print(json.dumps(dict(full_frames=len(result['full_frame_assets']),crops=len(result['crop_assets']),diagnostics=len(result['diagnostic_assets']),output=str(args.output.resolve()),GPU_used=False),ensure_ascii=False))


if __name__=='__main__':main()
