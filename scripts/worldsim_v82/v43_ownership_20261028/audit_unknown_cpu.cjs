// CPU only. Audit full exported contribution rows, never infer RGB removal from dots.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const root=path.resolve('outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000');
const out=path.resolve('outputs/V43_Ownership_20261027/gpu_v43');
const sb={console,Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,document:{getElementById(){return null}}};sb.window=sb;vm.createContext(sb);
for(const name of ['scene.js','identity.js','gpu_semantics/boundary_index.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),sb);
vm.runInContext(fs.readFileSync(path.resolve('outputs/GS_Scene_Diagnostics_20261027/viewer.js'),'utf8').replace(/init\(\);\s*$/,''),sb);
const a=sb.SceneViewer,ns=sb.BOUNDARY_INDEX.branch_sizes.original[0],ledger=new Map(),records=[];
function vehicle(k){return k>=14&&k<=19;}
(async()=>{
 for(const raw of Object.keys(sb.BOUNDARY_INDEX.frames).map(Number)){
  vm.runInContext(fs.readFileSync(path.join(root,sb.BOUNDARY_INDEX.frames[raw]),'utf8'),sb);
  const p=sb.BOUNDARY_FRAMES[raw],image=await a.unpack(p.image),stat={raw,variants:[]};
  let bandPixels=0,roadPixels=0;for(let i=0;i<480*320;i++)if(image.bands[480*320+i]){bandPixels++;if(image.semantic_image[i]===1)roadPixels++;}
  stat.band5_pixels=bandPixels;stat.band5_road_pixels=roadPixels;
  for(const variant of ['original','actor10k']){
   const d=await a.unpack(p[variant]),s={variant,contributing_rows:d.ids.length,base_unknown:0,band5_added_unknown:0,road_to_unknown:0,road_to_unknown_static:0,base_unknown_dynamic:0,total_mass:0,base_unknown_mass:0,band5_added_mass:0,road_to_unknown_mass:0};
   for(let j=0;j<d.ids.length;j++){
    const id=d.ids[j],k=d.class_id[j],m=d.mass[j],inBand=d.boundary_fraction[j*3+1]>=.2;
    s.total_mass+=m;if(k===0){s.base_unknown++;s.base_unknown_mass+=m;if(id>=ns)s.base_unknown_dynamic++;}
    else if(inBand){s.band5_added_unknown++;s.band5_added_mass+=m;if(k===1){s.road_to_unknown++;s.road_to_unknown_mass+=m;if(id<ns)s.road_to_unknown_static++;}}
    // Abstain on unconfirmed static vehicle appearance. This is NOT a deletion/owner rule.
    if(id<ns&&m>.01&&d.target_fraction[j]>.05&&d.vehicle_fraction[j]>.3){
     let e=ledger.get(id);if(!e)ledger.set(id,e={row:id,frames:new Set(),max_target_fraction:0,max_vehicle_fraction:0});
     e.frames.add(raw);e.max_target_fraction=Math.max(e.max_target_fraction,d.target_fraction[j]);e.max_vehicle_fraction=Math.max(e.max_vehicle_fraction,d.vehicle_fraction[j]);
    }
   }
   stat.variants.push(s);
  }
  records.push(stat);delete sb.BOUNDARY_FRAMES[raw];
 }
 const entries=[...ledger.values()].sort((x,y)=>x.row-y.row).map(e=>({...e,frames:[...e.frames].sort((a,b)=>a-b)}));
 const candidate={schema:'static-vehicle-appearance-unknown-v1',context:sb.SCENE_DATA.context,static_namespace:'V42 original static rows (same frozen rows in V43)',rule:{mass_gt:.01,target_fraction_gt:.05,vehicle_fraction_gt:.3,aggregation:'union across 49 observed frames and original/actor10k visibility; deduplicated static row IDs'},ids:entries.map(e=>e.row),evidence:entries,owner_unchanged:true,semantic_source_unchanged:true,not_identity_ground_truth:true,not_safe_deletion_mask:true};
 fs.writeFileSync(path.join(root,'gpu_semantics/unknown_candidates.js'),'window.STATIC_UNKNOWN_CANDIDATES='+JSON.stringify(candidate)+';\n');
 sb.BOUNDARY_INDEX.unknown_candidates='gpu_semantics/unknown_candidates.js';
 fs.writeFileSync(path.join(root,'gpu_semantics/boundary_index.js'),'window.BOUNDARY_INDEX='+JSON.stringify(sb.BOUNDARY_INDEX)+';\n');
 const report={schema:'unknown-cpu-audit-v1',context:sb.SCENE_DATA.context,frames:records.length,records,static_quarantine_rows:entries.length,static_quarantine_supported_at_least_3_frames:entries.filter(e=>e.frames.length>=3).length,source:'previously exported full visible alpha*T row arrays; not browser sample counts',radius_unit:'480x320 observed pixels',contour_scope:'all predicted vehicle classes 14..19 UNION target DEVA body',training_updates:0,gpu_calls:0,rgb_ablation_run:false,rgb_ghost_fully_removed:null,road_holes_evaluated:false};
 assert.equal(records.length,49);assert(entries.length>0);assert(records.every(r=>r.variants.every(s=>s.road_to_unknown>=s.road_to_unknown_static)));
 fs.writeFileSync(path.join(out,'unknown_cpu_audit.json'),JSON.stringify(report,null,2));
 fs.writeFileSync(path.join(out,'unknown_candidates.json'),JSON.stringify(candidate,null,2));
 console.log(JSON.stringify({frames:49,static_quarantine_rows:entries.length,raw113:records.find(r=>r.raw===113)}));
})().catch(e=>{console.error(e);process.exitCode=1});
