"""Build a private forward-only half-space path from the verified native CUDA source."""
from pathlib import Path
import hashlib
import shutil
import sys

run=Path(sys.argv[1])
source=run.parent.parent/'deps/diff-gaussian-rasterization'
glm=run.parent.parent/'infra/rasterizer_compact/third_party/glm'
assert (glm/'glm/glm.hpp').is_file()
target=run/'private_cuda'
target.mkdir()
for name in ('cuda_rasterizer','rasterize_points.cu','rasterize_points.h','ext.cpp','setup.py','LICENSE.md'):
    path=source/name
    if path.is_dir():
        shutil.copytree(path,target/name)
    else:
        shutil.copyfile(path,target/name)
shutil.copytree(source/'diff_gaussian_rasterization',target/'a6_clip_rasterization')
(target/'third_party').mkdir()
(target/'third_party/glm').symlink_to(glm,target_is_directory=True)
setup=target/'setup.py'
setup.write_text(setup.read_text().replace('diff_gaussian_rasterization','a6_clip_rasterization'))
p=target/'cuda_rasterizer/forward.cu'
text=p.read_text()
old='			float alpha = min(0.99f, con_o.w * exp(power));'
assert text.count(old)==1
new=r'''
			// 24 channels are private to this forward-only experiment. The
			// first six are normal feature channels. The rest specify a
			// camera-space Gaussian precision and a persistent half-space.
			if (S == 24)
			{
				const float* q = features + collected_id[j] * S + 6;
				if (q[0] > 0.5f)
				{
					const float x = fmaf(q[14], pixf.x, q[15]);
					const float y = fmaf(q[16], pixf.y, q[17]);
					const float px = fmaf(q[1], x, fmaf(q[2], y, q[3]));
					const float py = fmaf(q[2], x, fmaf(q[4], y, q[5]));
					const float pz = fmaf(q[3], x, fmaf(q[5], y, q[6]));
					const float precision = fmaxf(fmaf(x, px, fmaf(y, py, pz)), 1e-12f);
					const float t = fmaf(x, q[7], fmaf(y, q[8], q[9])) / precision;
					const float n_ray = fmaf(x, q[10], fmaf(y, q[11], q[12]));
					const float side = fmaf(n_ray, t, -q[13]);
					const float deviation = fabsf(n_ray) * rsqrtf(precision);
					const float retained = deviation > 1e-10f
						? 0.5f * erfcf(-0.7071067811865475f * side / deviation)
						: (side >= 0.0f ? 1.0f : 0.0f);
					alpha *= retained;
				}
			}
'''
text=text.replace(old,old+new)
orig='for (int ch = 0; ch < S; ch++)\n\t\t\t\tF[ch] +='
assert text.count(orig)==1
text=text.replace(orig,'for (int ch = 0; ch < (S == 24 ? 6 : S); ch++)\n\t\t\t\tF[ch] +=')
p.write_text(text)
receipt=dict(native_forward_sha256=hashlib.sha256((source/'cuda_rasterizer/forward.cu').read_bytes()).hexdigest(),
             private_forward_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
import json
(run/'build_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
