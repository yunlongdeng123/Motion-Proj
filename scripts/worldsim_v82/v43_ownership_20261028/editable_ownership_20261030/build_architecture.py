"""在既有 V42 大框与内部小框中更新 V43；图中设计不代表完成结果。"""
from html import escape


def architecture():
    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1710" viewBox="0 0 1800 1710" role="img" aria-labelledby="title desc">',
        '<title id="title">V43：身份与表示修复，保留 V1 原生通路与 V42 完整编辑架构</title>',
        '<desc id="desc">RGB、LiDAR、相机与轨迹进入时空编辑、身份与表示修复、原生渲染及按需生成三个大组件。V43 分开语义、归属、运动与证据，检测混合支撑，保留两侧子高斯局部拟合，并分离道路接收面上的车辆阴影效应。无编辑请求保留 V1 原生输出。</desc>',
        '<rect width="1800" height="1710" fill="#f5f9fc"/>',
        '<g font-family="Microsoft YaHei,Arial,sans-serif" fill="#203f51">',
    ]

    def text(x, y, s, size=20, bold=False, anchor='middle', color='#203f51'):
        p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(s)}</text>')

    def box(x, y, w, h, title, lines=(), color='#e6eef6', size=21):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" fill="{color}" stroke="#a5bfce" stroke-width="1.5"/>')
        text(x + w / 2, y + 32, title, size, True)
        for i, line in enumerate(lines):
            text(x + w / 2, y + 61 + 26 * i, line, 18)

    def arrow(points, color='#66879a', dash=False):
        import math
        a, b = points[-2:]
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        p.append('<polyline points="' + ' '.join(f'{x},{y}' for x, y in points) + f'" stroke="{color}" stroke-width="2.5" fill="none"' + (' stroke-dasharray="7 5"' if dash else '') + '/>')
        p.append(f'<path d="M0 0L-11 -5L-11 5Z" transform="translate({b[0]} {b[1]}) rotate({angle})" fill="{color}"/>')

    text(40, 48, 'V43：身份 + 表示修复 → 时空编辑 → 背景修复与按需生成', 32, True, 'start')
    text(40, 84, '保留 V1 原生无编辑通路与 V42 三大组件；分类、几何修复、阴影解耦和生成收益分别验证。', 21, False, 'start')
    box(40, 115, 710, 90, '只读原始高斯 / checkpoint / 已登记身份', ['保留原参数、来源索引与 parent–child lineage'])
    box(1310, 115, 450, 90, 'V1 无编辑：原生通路', ['已有零差对照；修复分支单独验收'])
    arrow([(750, 160), (1310, 160)])
    text(1030, 146, '无编辑请求直通', 18)
    box(40, 235, 1720, 72, '输入：全片段 RGB / LiDAR / 相机 / 时间 / 跟踪；请求：对象、轨迹、资产、操作区间', [], '#eaf1f7', 23)
    arrow([(395, 205), (395, 235)])

    for x, w, title, sub, color in [
        (40, 450, '① 时空编辑与对象管理', '继承 V42 接口 · 保留已有操作身份', '#fff1dc'),
        (540, 720, '② 身份与表示修复', 'V43 主线 · 多头归属与可修复表示', '#e0f1ea'),
        (1310, 450, '③ 渲染、背景修复与生成', '继承 V42 大架构 · 生成按需调用', '#fff1dc'),
    ]:
        box(x, 345, w, 1030, title, [sub], color, 24)
        arrow([(x + w / 2, 307), (x + w / 2, 345)])

    # 保留原有左右组件和有向内部链。
    box(65, 445, 400, 130, '已有实例 / 连续轨迹', ['黑车 A、白车 B、道路表面 R', '对象 owner 不因停车或遮挡改变', 'Actor 与自车时钟分别求值'])
    box(65, 625, 400, 112, '编辑请求与保护范围', ['移动 / 删除 / 插入 / 转向', '目标、时刻、姿态与保留对象'])
    box(65, 787, 400, 138, '编辑状态与可见性', ['原位显露、新遮挡与目标范围', '稳定 owner ≠ 当前可见性', '置信随证据变化，身份不逐帧跳变'])
    box(65, 975, 400, 138, '按目标查询 1 / 0 / U', ['1 属于目标；0 确认非此目标', 'U 为证据不足、冲突或混合待拆', 'static 分支 ≠ 语义背景'])
    text(265, 1200, '已有：对象编辑与原生基座', 20, True)
    text(265, 1233, '待实现：完整跨分支归属与表示修复', 18)
    text(265, 1266, '原始资产与所有历史对照保留', 18)
    for ya, yb in [(575, 625), (737, 787), (925, 975)]:
        arrow([(265, ya), (265, yb)])

    box(565, 435, 670, 108, '②a-1 跨分支证据与来源账本', [
        '实测来源 / 多帧关联 / 深度轨迹 / αT 贡献',
        '剥离后重算真实贡献；保留原行号与 parent–child lineage',
    ], '#f1faf5', 21)
    box(565, 567, 670, 190, '②a-2 分类器合同：四字段 + mixed / effect', [
        'semantic：车辆 / 道路等；instance / surface owner：A / R',
        'motion binding：Actor 局部 / world；evidence / confidence',
        'support type：pure / spatial mixed / unknown',
        'effect type：shadow 等；effect owner 与 receiver 分开',
        '已有规则候选；多头可学习分类器尚未实现',
    ], '#e9f5ef', 21)
    box(565, 781, 670, 164, '②a-3 先查空间相容性；不足仍 unknown', [
        'pure → 归属；mixed 先查父椭球与实测支持相容性',
        '相容再双侧 split/refit；不相容先校几何 / 重建表面',
        'shadow → world 道路 receiver 上的 car-owner 外观效应',
        'no-edit 保真不代表分离；须查 after 真实透射下两侧',
    ], '#fff2dd', 21)
    box(565, 969, 670, 112, '②a-4 首轮强控制 → 再做分类器迁移', [
        '归属理想 / 证据候选上界 + mixed 检测 → split × shadow4',
        '再用外部数据训练；场景 1 只做推理 / 开发诊断',
    ], '#fff2dd', 21)
    box(565, 1105, 670, 112, '②a-5 独立验收：归属、几何、外观与编辑收益', [
        'R2 ROI 白底 ≠ unknown SH 涂白 ≠ 学习分类 ≠ 编辑收益',
        '尚无人工 GT；不宣称 oracle 完成或人工验收通过',
    ], '#fff2dd', 21)
    box(565, 1241, 670, 108, '②b 显露背景修复 · 保留 V42 后续组件', [
        '保留正确表面 + 污染修正 + 有来源补层',
        '分步 / assignment-only / joint / joint-detached 配对',
    ], '#f4f1e9', 21)
    for ya, yb in [(543, 567), (757, 781), (945, 969), (1081, 1105), (1217, 1241)]:
        arrow([(900, ya), (900, yb)])

    box(1335, 445, 400, 130, '同一时刻求值与原生合成', ['稳定 owner + 当前编辑变换', 'RGB / 深度 / 实例与真实贡献', '无编辑保留 V1 直接输出'])
    box(1335, 625, 400, 112, '三维结果与分量诊断', ['车身 / 道路 receiver / effect', '高斯层与原环境泄漏分开核对'])
    box(1335, 787, 400, 138, '背景修复与按需生成', ['指定显露区域 → 固定 ProPainter', '三维原生结果与生成结果分别保存', '生成图不转成独立实测或标签'])
    box(1335, 975, 400, 138, '分项验收与强对照', ['归属 / mixed / 几何 / shadow4', '保留区损伤、时序稳定与资源', '相同证据、容量、更新及后端'])
    text(1535, 1200, '输出：可追溯三维状态 + 编辑视频', 20, True)
    text(1535, 1233, '补全与联合收益均须实验成立', 18)
    text(1535, 1266, '现有诊断产物不等于方法验收', 18)
    for ya, yb in [(575, 625), (737, 787), (925, 975)]:
        arrow([(1535, ya), (1535, yb)])

    # 大组件之间的箭头仍在大框间留白，避免覆盖内部框。
    arrow([(490, 665), (540, 665)])
    arrow([(1260, 665), (1310, 665)])
    box(40, 1415, 1720, 112, '贯穿机制：稳定 owner 与逐时刻可见性 / 置信分开', [
        '世界 / Actor 局部状态保留来源、冲突和时间；混合分裂保存两侧子高斯、原父高斯与局部 refit 记录。',
        '正序 / 逆序 / 跳帧检查证据复用；剥离后新显露贡献可追查，不用自己的输出循环生成 GT。',
    ], '#eaf1f7', 23)
    for x in [265, 900, 1535]:
        arrow([(x, 1415), (x, 1375)], dash=True)
    box(40, 1560, 1720, 72, '共同验收：归属精度与覆盖 → 表示与几何 → 背景 / 阴影 → 编辑视频质量与成本', [], '#ffffff', 24)
    text(40, 1663, '蓝：继承组件　绿：身份合同　黄：表示修复 / 强控制　灰：后续补全；双侧 refit 已有短探针，空间分离尚未证实。', 17, False, 'start', '#5b778a')
    text(40, 1692, '本轮证据：三锚点原生短探针 + 200 次 child 参数更新；分类器未训练，人工 GT 与编辑收益验收仍未完成。', 17, False, 'start', '#5b778a')
    p.append('</g></svg>')
    return ''.join(p)
