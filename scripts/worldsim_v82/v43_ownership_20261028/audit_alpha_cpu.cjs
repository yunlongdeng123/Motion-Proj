// Read saved native maps and original target masks. No renderer, GPU or RGB score.
const fs=require('fs'),path=require('path'),vm=require('vm');
const folder=path.resolve('outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000'),out=path.resolve('outputs/V43_Ownership_20261027/gpu_v43');
const s={Float32Array,Float64Array,Uint8Array,Uint8ClampedArray,Uint16Array,Int32Array,ArrayBuffer,TextDecoder,Map,Set,Blob,Response,DecompressionStream,atob,document:{getElementById(){return null}},console};s.window=s;s.EVIDENCE_MAPS={};vm.createContext(s);
function load(f){vm.runInContext(fs.readFileSync(path.join(folder,f),'utf8'),s,{filename:path.basename(f)});}
for(const f of ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','v43_groups.js','semantic_observations.js'])load(f);
vm.runInContext(fs.readFileSync(path.resolve('outputs/GS_Scene_Diagnostics_20261027/viewer.js'),'utf8').replace(/init\(\);\s*$/,''),s);
(async()=>{const records=[];for(const raw of [91,98,101,109,113,114]){
 const f=s.SCENE_DATA.groups.find(x=>x.id==='v43_motion_no_edit').frames[raw];load(f.maps);const m=await s.SceneViewer.unpack(s.EVIDENCE_MAPS[f.maps]),obs=await s.SceneViewer.unpack(s.SEMANTIC_OBSERVATIONS[raw]);let n=0,inside=0,outside=0;
 for(let j=0;j<480*320;j++){const a=m.isolated_alpha[j]/255;if(obs.mask[j]){n++;inside+=Math.abs(1-a)}else outside+=Math.abs(a)}
 records.push({raw,body_pixels:n,body_area_fraction:n/(480*320),inside_alpha_MAE:inside/n,outside_alpha_MAE:outside/(480*320-n),inside_coefficient_in_current_loss:.1*n/(480*320),outside_coefficient_in_current_loss:.1*(1-n/(480*320)),inside_alpha_loss_contribution:.1*inside/(480*320),outside_alpha_loss_contribution:.1*outside/(480*320)});
 }
 const r={source:'existing 8-bit quantized no-edit isolated-alpha maps and original target DEVA masks',quantization_error_bound:1/255,training_alpha_loss:'.1 * full_image_mean(abs(alpha-body))',rgb_scores_used:false,gpu_calls:0,records,causal_claim:'area normalization weakens small foreground weighting; causal effect on pale edits not yet tested'};
 fs.writeFileSync(path.join(out,'alpha_normalization_cpu_audit.json'),JSON.stringify(r,null,2));console.log(JSON.stringify(r));
})().catch(e=>{console.error(e.message);process.exitCode=1});
