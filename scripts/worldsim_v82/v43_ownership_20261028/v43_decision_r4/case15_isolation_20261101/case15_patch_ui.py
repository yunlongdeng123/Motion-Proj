import pathlib,re,shutil,json
W=pathlib.Path(__file__).resolve().parent;B=pathlib.Path(r'C:\Users\dengyunlong\Documents\Codex\2026-10-31\g\outputs\V43_R4_GPU_20261031\actor_candidates');H=W/'case15/ui';backup=H/'before_lidar3d';backup.mkdir(parents=True,exist_ok=True)
original={n:(B/n).read_bytes() for n in ['index.html','sync_panel.js']}
for n,b in original.items():
 p=backup/n
 if p.exists():raise RuntimeError('Backup exists; inspect before rerunning patch')
 p.write_bytes(b)
sync=original['sync_panel.js'].decode('utf8')
helper='''
 function renderLidar3D(points){
  const {data,row}=C();if(!data||!row||!sceneMedia){window.ActorLidar3D?.clear();return;}
  const mode=$('lidarMode').value,raw=displayedRaw,actor=sceneMedia.actors?.[row.track_id],pose=actor?.trajectory?.find(x=>x.raw===raw)||actor?.trajectory?.[0];
  const item=mode==='frame'?pointFramesData?.[String(raw)]:mode==='gs'?gsBundle?.frames?.find(x=>x.raw===raw):pointData;
  const actual=mode==='gs'?item?.measured_lidar_points:item?.measured_points_before_voxel;
  window.ActorLidar3D?.setData({points,contextId:data.context_id,trackId:row.track_id,actorAlias:window.ActorNames.actor(data.context_id,row.track_id)||'车辆',sceneName:sceneMedia.display_name||data.display_label,dimensions:pose?.dimensions,mode,raw,actualCount:actual});
 }
'''
assert sync.count(' function renderCloud()')==1
sync=sync.replace(' function renderCloud()',helper+' function renderCloud()')
sync=sync.replace("$('lidarViews').hidden=true;","$('lidarViews').hidden=true;window.ActorLidar3D?.clear();")
# Empty-but-measured frames remain distinct from missing labels/data and get a visible empty 3D box.
needle="setText('lidarInfo',item.measured_points_before_voxel===0?info+' 该帧框内无实测点。':info+' 该帧没有可显示的采样点。');return;"
assert sync.count(needle)==1
sync=sync.replace(needle,needle.replace('return;','renderLidar3D(points);return;'))
needle="setText('lidarInfo',info+' 该范围没有可显示点。');return;"
assert sync.count(needle)==1
sync=sync.replace(needle,needle.replace('return;','renderLidar3D(points);return;'))
needle="$('lidarViews').hidden=false;setText('lidarInfo',info);"
assert sync.count(needle)==1
sync=sync.replace(needle,needle+'renderLidar3D(points);')
assert sync.count("$('lidarMode').value='all';")==1
sync=sync.replace("$('lidarMode').value='all';","$('lidarMode').value='frame';")
needle="if(li.display_points===0){setText('lidarInfo','该车辆没有可显示的实测激光点。');return;}"
assert sync.count(needle)==1
sync=sync.replace(needle,'')
needle="pointData=null;setText('lidarInfo','该车辆的激光点未能读取。');return;}renderCloud();"
assert sync.count(needle)==1
sync=sync.replace(needle,"pointData=null;setText('lidarInfo','该车辆的激光点未能读取。');return;}if($('lidarMode').value==='frame'&&!pointFramesData)await requestFrameData();else renderCloud();")
page=original['index.html'].decode('utf8')
assert 'id="lidar3DPanel"' not in page
section='''<section id="lidar3DPanel" hidden><h3>3D 实测 LiDAR</h3><p id="lidar3DIdentity" class="candidate-note"></p><div class="lidar3d-controls"><button id="lidar3DFit" type="button">适配车辆</button><label>观察方向 <select id="lidar3DView"><option value="orbit">自由观察</option><option value="top">俯视</option><option value="side">侧视</option><option value="front">正视</option></select></label><label><input id="lidar3DBox" type="checkbox" checked>标注框</label><label><input id="lidar3DGrid" type="checkbox" checked>0.5米网格</label><label>点大小 <input id="lidar3DPointSize" type="range" min="1" max="4" step="0.5" value="2"><span id="lidar3DPointValue">2.0 px</span></label></div><canvas id="lidar3DCanvas" width="960" height="520" aria-label="当前车辆真实实测LiDAR三维点云，可拖动旋转、滚轮缩放"></canvas><p id="lidar3DInfo" role="status"></p><p class="candidate-note">拖动旋转，滚轮缩放，Shift＋拖动平移，双击适配车辆。青色为实测激光点；黄色框为原始标注尺寸，底面网格不是测得地面。上方“显示范围”同时控制3D与三视图。</p></section><h3>正交三视图</h3>'''
needle='<div class="lidar-views" id="lidarViews" hidden>'
assert page.count(needle)==1
page=page.replace(needle,section+needle)
page=page.replace('</head>','<link rel="stylesheet" href="lidar3d.css?v=lidar3d-20261101-r1"></head>')
pattern=r'<script src="sync_panel\.js[^\"]*"></script>'
assert len(re.findall(pattern,page))==1
page=re.sub(pattern,'<script src="lidar3d.js?v=lidar3d-20261101-r1"></script><script src="sync_panel.js?v=lidar3d-20261101-r1"></script>',page)
for n,b in original.items():
 if (B/n).read_bytes()!=b:raise RuntimeError('Concurrent page change; preserve it')
for n,text in [('index.html',page),('sync_panel.js',sync)]: (B/n).write_bytes(text.replace('\r\n','\n').encode('utf8'))
for src,dest in [('case15_lidar3d.js','lidar3d.js'),('case15_lidar3d.css','lidar3d.css')]:
 if (B/dest).exists():raise RuntimeError('Unexpected existing '+dest)
 shutil.copy2(W/src,B/dest)
(H/'after_lidar3d').mkdir(exist_ok=True)
for n in ['index.html','sync_panel.js','lidar3d.js','lidar3d.css']:shutil.copy2(B/n,H/'after_lidar3d'/n)
print(json.dumps({'updated_page':str(B/'index.html'),'files':['index.html','sync_panel.js','lidar3d.js','lidar3d.css'],'default_scope':'current frame','source_points_modified':False,'native_GPU_queries':0},ensure_ascii=False))
