const fs=require('fs');
const {createCanvas,loadImage}=require('C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas');
const path=require('path');
(async()=>{const root=path.resolve(__dirname,'../../outputs/V43_Ownership_20261027/gpu_v43/target_detection_v3');const im=await loadImage(fs.readFileSync(path.join(root,'architecture.svg')));const c=createCanvas(1320,390);c.getContext('2d').drawImage(im,0,0,1320,390);fs.writeFileSync(path.join(root,'architecture.png'),c.toBuffer('image/png'));console.log('rendered architecture');})();
