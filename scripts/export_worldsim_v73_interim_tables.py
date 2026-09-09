"""从已归档的真实结果生成阶段论文表格；不重新推理或重算统计。"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    joint = json.loads((args.evidence_root / 'm2/global/population_joint_r5_analysis.json').read_text())
    tsdf = json.loads((args.evidence_root / 'm2/global/actor_tsdf_r1_analysis.json').read_text())
    native = json.loads((args.evidence_root / 'm2/global/population_native_r11_analysis.json').read_text())
    triangle = json.loads((args.evidence_root / 'm2/global/population_joint_r12_analysis.json').read_text())
    methods = [('LiDAR PCA', 'lidar_pca'), ('Native fusion', 'native_fusion'),
               ('LiDAR R6 (window labels)', 'lidar_r6'), ('LiDAR R7 (track labels)', 'lidar_r7'),
               ('LiDAR R8 (beam free)', 'lidar_r8'), ('LiDAR R9 (+event)', 'lidar_event_r9'),
               ('CAPA-style TTA R2', 'capa_r2'), ('AdaPoinTr R2', 'adapointr_r2'),
               ('Joint R5 (window labels)', 'final')]
    rows = [(label, joint['stages'][key]['development']) for label, key in methods]
    rows.insert(1, ('Actor TSDF + carving', tsdf['stages']['final']['development']))
    rows.append(('Native DPT R11 (track labels)', native['stages']['final']['development']))
    rows.extend((label,triangle['stages'][key]['development']) for label,key in [
        ('Joint R10 (track, hard free)','joint_r10'),('Joint R12 (track, beam free)','final'),
        ('Native R14 (track, beam free)','native_r14')])
    for label,filename in [('Q-v2 joint r1','qv2/shared_mesh_joint_r1_analysis.json'),
                           ('Q-v2 LiDAR r2','qv2/shared_mesh_lidar_r2_analysis.json'),
                           ('Open charts r3 (LiDAR)','open_charts/open_charts_lidar_r3_analysis.json'),
                           ('Ray attraction r4 (LiDAR)','ray_support/ray_support_r4_analysis.json')]:
        completed=json.loads((args.evidence_root/filename).read_text())
        rows.append((label,completed['stages']['final']['development']))
    metrics = [('hit_rate', 100, 2), ('early_rate', 100, 2), ('miss_rate', 100, 2),
               ('free_intrusion_m', 1, 4), ('surface_recall_02', 100, 2)]
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Method & Hit (\%) & Early (\%) & Miss (\%) & Free (m) & Recall (\%) \\', r'\midrule']
    for label, values in rows:
        cells = [f"{values[metric]['mean'] * scale:.{digits}f}" for metric, scale, digits in metrics]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (args.output / 'actor_population.tex').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    lines = [r'\begin{tabular}{llrr}', r'\toprule',
             r'Joint R5 minus & Metric & Difference & 95\% log bootstrap \\', r'\midrule']
    for label, key, metric, unit, scale in [
            ('R5 initialization', 'initial', 'early_rate', 'Early (pp)', 100),
            ('R5 initialization', 'initial', 'miss_rate', 'Miss (pp)', 100),
            ('LiDAR R6', 'lidar_r6', 'hit_rate', 'Hit (pp)', 100),
            ('LiDAR R6', 'lidar_r6', 'free_intrusion_m', 'Free (m)', 1),
            ('Native fusion', 'native_fusion', 'early_rate', 'Early (pp)', 100),
            ('Native fusion', 'native_fusion', 'free_intrusion_m', 'Free (m)', 1)]:
        value = joint['paired_final_minus'][key]['development'][metric]
        lo, hi = value['bootstrap95']
        lines.append(f"{label} & {unit} & {value['mean_delta'] * scale:+.3f} & "
                     f"[{lo * scale:+.3f}, {hi * scale:+.3f}]" + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (args.output / 'joint_paired.tex').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Metric & Difference & 95\% log bootstrap & Improved logs \\', r'\midrule']
    for metric, label, scale in [('hit_rate', 'Hit (pp)', 100),
                                 ('early_rate', 'Early (pp)', 100),
                                 ('miss_rate', 'Miss (pp)', 100),
                                 ('free_intrusion_m', 'Free (m)', 1),
                                 ('surface_distance_m', 'One-way distance (m)', 1),
                                 ('surface_recall_02', 'Recall (pp)', 100)]:
        value = native['paired_final_minus']['initial']['development'][metric]
        lo, hi = value['bootstrap95']
        lines.append(f"{label} & {value['mean_delta'] * scale:+.3f} & "
                     f"[{lo * scale:+.3f}, {hi * scale:+.3f}] & {value['improved_logs']}/{value['logs']}" + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (args.output / 'native_paired.tex').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    lines=[r'\begin{tabular}{llrr}',r'\toprule',
           r'R12 minus & Metric & Difference & 95\% log bootstrap \\',r'\midrule']
    for key,label,selected in [('joint_r10','R10',['hit_rate','early_rate','miss_rate','free_intrusion_m','surface_distance_m','surface_recall_02']),
                               ('native_r14','R14',['hit_rate','early_rate','miss_rate','free_intrusion_m']),
                               ('lidar_r8','R8',['hit_rate','miss_rate'])]:
        for metric,unit,scale in [('hit_rate','Hit (pp)',100),('early_rate','Early (pp)',100),
                                  ('miss_rate','Miss (pp)',100),('free_intrusion_m','Free (m)',1),
                                  ('surface_distance_m','Distance (m)',1),('surface_recall_02','Recall (pp)',100)]:
            if metric not in selected: continue
            value=triangle['paired_final_minus'][key]['development'][metric]; lo,hi=value['bootstrap95']
            lines.append(f"{label} & {unit} & {value['mean_delta']*scale:+.3f} & [{lo*scale:+.3f}, {hi*scale:+.3f}]"+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (args.output/'triangle_paired.tex').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    ray=json.loads((args.evidence_root/'ray_support/ray_support_r4_analysis.json').read_text())
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Metric & Difference & 95\% log interval & Improved logs \\',r'\midrule']
    for key,label,scale in [('hit_rate','Hit (pp)',100),('early_rate','Early (pp)',100),('miss_rate','Miss (pp)',100),
                          ('free_intrusion_m','Free (m)',1),('surface_distance_m','Distance (m)',1),('surface_recall_02','Recall (pp)',100)]:
        value=ray['paired_final_minus']['open_charts_r3']['development'][key]; lo,hi=value['bootstrap95']
        lines.append(f"{label} & {value['mean_delta']*scale:+.3f} & [{lo*scale:+.3f}, {hi*scale:+.3f}] & {value['improved_logs']}/{value['logs']}"+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (args.output/'ray_support_paired.tex').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    print(json.dumps({'tables': ['actor_population.tex', 'joint_paired.tex', 'native_paired.tex','triangle_paired.tex','ray_support_paired.tex'],
                      'source': str(args.evidence_root), 'neural_updates': 0}))


if __name__ == '__main__':
    main()
