"""Adapt immutable report records for one viewer; never recompute metrics."""
from pathlib import Path
import copy,hashlib,json
from PIL import Image

LEGACY_LABELS={
    '1':'历史① 共享世界颜色场（480）',
    '2':'历史② 高频许可与颜色读取（480）',
    '3':'历史③ 原生细节与数值控制（1920）',
}
COMMON_LABELS={
    '1':'1152重训① 共享世界颜色场',
    '2':'1152重训② 高频许可与颜色读取',
    '3':'1152重训③ 细节目标与数值控制',
}
COMMON_NAMES={
    'common_r1':'冻结基础材质', 'L800':'仅低频颜色场',
    'ABC800':'低频＋高频联合', 'matched_I0':'独立高斯颜色对照',
    'common_avg':'足迹平均初始态', 'common_pixel':'像素读取初始态',
    'M00':'平均取色／原高频许可', 'M10':'平均取色／教师高频许可',
    'M01':'像素读取／原高频许可', 'M11':'像素读取／教师高频许可',
    'M12':'像素读取／附加对照', 'F0':'冻结普通高斯基线',
    'G_RGB':'普通高斯／RGB拟合', 'G_DETAIL':'普通高斯／显式细节目标',
    'G_NUMERIC':'普通高斯／数值控制',
}

def attach(data,root):
    root=Path(root).resolve()
    def read(name):return json.loads((root/name).read_text('utf-8'))
    old=read('report_data.json')
    common=read('common1152/report_data.json')
    adaptive=read('surface_adaptive/report_data.json')
    assert len(old['methods'])==30 and len(common['models'])==15 and len(adaptive['models'])==5 and len(data['models'])==5
    assert [len(x)for x in [old['metrics'],common['rows'],adaptive['rows'],data['rows']]]==[294,270,90,90]
    batches=[];models=[];rows=[];gaps=[];newly=[];sizes={}
    def image_size(path):
        if path not in sizes:
            p=(root/path).resolve()
            if not p.is_relative_to(root)or not p.is_file():raise ValueError('missing/unsafe report image: '+path)
            with Image.open(p)as im:sizes[path]=list(im.size)
        return sizes[path]
    def frame_images(images):
        frames={}
        for raw,views in images.items():
            f={}
            for view,v in views.items():
                if not isinstance(v,dict)or 'overview'not in v:continue
                f[view]=v['overview']
                for region,c in v.get('crops',{}).items():f[region+'_'+view+'_crop']=c['path']
            frames[str(raw)]=f
        return frames
    def legacy_teacher(group):
        teacher=dict(frames={},roles={})
        for raw,regions in old['teachers'].get(group,{}).items():
            teacher['frames'][str(raw)]={}
            teacher['roles'][str(raw)]={}
            for region,v in regions.items():
                teacher['frames'][str(raw)][region]=v['overview']
                crop=v.get('crops',{}).get(region)
                if crop:teacher['frames'][str(raw)][region+'_crop']=crop['path']
                teacher['roles'][str(raw)][region]=v.get('role','teacher')
        return teacher
    def add_model(model,batch,original_id):
        m=copy.deepcopy(model);m['original_id']=original_id;m['batch']=batch['id'];m['id']=batch['id']+'::'+original_id
        m['frame_sizes']={}
        for raw,f in m['frames'].items():
            m['frame_sizes'][raw]={}
            for key,path in f.items():
                if isinstance(path,str)and Path(path).suffix.lower()in ['.png','.webp','.jpg']:
                    m['frame_sizes'][raw][key]=image_size(path)
        assert set(m['frames'])=={'86','110','134','179','198'}
        for f in m['frames'].values():assert f.get('road')and f.get('joint')
        models.append(m)
        return m['id']
    for group,label in LEGACY_LABELS.items():
        g=old['groups'][group]
        b=dict(id='legacy_'+group,label=label,teacher=legacy_teacher(group),scope=g['resolution']+'；'+g['metric_scope'],
               metric_scope=g['metric_scope'],geometry_support={},defaults=[('legacy_'+group+'::'+group+':'+v)for v in g['defaults']],kind='legacy')
        batches.append(b)
        for original_id,m in old['methods'].items():
            if m['group']!=group:continue
            add_model(dict(name=m['name']+' · '+m['key'],budget=m['budget'],renderer=m['kind'],role=m['role'],frames=frame_images(m['images'])),b,original_id)
        for r in old['metrics']:
            if not r['id'].startswith(group+':'):continue
            a=copy.deepcopy(r)
            a.update(id=b['id']+'::'+r['id'],batch=b['id'],original_id=r['id'],source_path=r.get('source'),source=r['region'],
                     RGB_MAE=r.get('mae'),HF_cosine=r.get('hf'),HF_energy_ratio=r.get('energy'),edge_F1=r.get('edge'),flat_pred_RMS=r.get('flat'),core_n=r.get('core'),score_n=r.get('score_n',r.get('n')),stable_score_n=r.get('n'))
            rows.append(a)
    def add_new(source,batch_id,label,chosen=None,scope='',kind='native'):
        selected=[m for m in source['models']if chosen is None or str(m['round'])==chosen]
        selected_ids={m['id']for m in selected}
        b=dict(id=batch_id,label=label,teacher=copy.deepcopy(source['teacher']),scope=scope,metric_scope=scope,
               geometry_support=copy.deepcopy(source.get('geometry_support',{})),kind=kind)
        b['defaults']=[batch_id+'::'+selected[0]['id'],batch_id+'::'+selected[-1]['id']]
        batches.append(b)
        for m in selected:
            adapted=copy.deepcopy(m)
            if batch_id.startswith('common_'):
                key=m['name'];adapted['name']=COMMON_NAMES.get(key,key)+' · '+m['id']
            add_model(adapted,b,m['id'])
        for key,target in [('rows',rows),('gaps',gaps),('newly_covered_rows',newly)]:
            for r in source.get(key,[]):
                if r['id']not in selected_ids:continue
                a=copy.deepcopy(r);a.update(id=batch_id+'::'+r['id'],batch=batch_id,original_id=r['id'])
                if key=='rows':a['domain']=scope
                target.append(a)
    for group,label in COMMON_LABELS.items():
        add_new(common,'common_'+group,label,group,'1152×768；固定父高覆盖且世界场有效域；原生像素HF。')
    add_new(adaptive,'adaptive','道路增密① 启用锚点与真实补点',scope='1152×768；首轮父资产与终态的共同稳定域；新增覆盖另列。',kind='coverage')
    add_new(data,'residual','道路增密② 残余覆盖与外观拟合',scope='1152×768；第二轮直接父资产与终态的共同稳定域；新增覆盖另列。',kind='coverage')
    for b in batches:
        for f in b['teacher'].get('frames',{}).values():
            for path in f.values():
                if isinstance(path,str)and Path(path).suffix.lower()in ['.png','.webp','.jpg']:image_size(path)
        for path in b['geometry_support'].values():image_size(path)
        b['count']=sum(m['batch']==b['id']for m in models)
    assert len(models)==55 and len({m['id']for m in models})==55 and len(rows)==744 and len(gaps)==150
    assert [b['count']for b in batches]==[11,11,8,4,7,4,5,5]
    expected={b['id'] for b in batches}
    assert all(r['batch']in expected for r in rows+gaps+newly)
    catalog=dict(schema='road_report_catalog.v1',batches=batches,models=models,rows=rows,gaps=gaps,newly_covered_rows=newly,
                 default_pair=['common_3::R3_G_NUMERIC','residual::appearance'],metric_recomputed=False)
    data['catalog']=catalog
    evidence=[('原三轮完整数据','report_data.json'),('1152教师重训完整数据','common1152/report_data.json'),('第一轮增密完整数据','surface_adaptive/report_data.json')]
    data['evidence'].extend(dict(name=name,path=path)for name,path in evidence)
    audit=dict(schema='report_catalog_integrity.v1',passed=True,records=55,batches={b['id']:b['count']for b in batches},
               metric_rows=744,gap_rows=150,unique_images_validated=len(sizes),metrics_recomputed=False,
               source_SHA256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for _,name in evidence})
    return audit
