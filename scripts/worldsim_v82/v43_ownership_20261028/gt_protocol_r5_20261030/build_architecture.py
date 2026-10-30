"""V43 revision 5：保留主架构嵌套框，新增世界GT与合法输入分支图。"""
from html import escape


def architecture():
    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1710" viewBox="0 0 1800 1710" role="img" aria-labelledby="title desc">',
        '<title id="title">V43：共享证据编码与局部可编辑联合分解，保留 V1 与 V42 架构</title>',
        '<desc id="desc">保留时空编辑、局部联合分解、渲染与按需生成三个大组件和内部小框。共享证据联合输出身份、几何、混合支持、两侧拆分及车辆效应；场景判断不接收编辑指令。GT在独立世界表面、对象与编辑干预上，不预设每颗Gaussian唯一标签。原分支和语义为噪声证据；alphaT语义只是派生责任。错深度、mixed、无对应和缺测分开。先一个普通最小场景经受限观测与当前代码实际重建，观察自然污染，再三至五个机制场景与小学习；M1M2是真值辅助参考而非数学上界。环境冻结四路配对保留；场景一只开发与推理。</desc>',
        '<rect width="1800" height="1710" fill="#f5f9fc"/>',
        '<g font-family="Microsoft YaHei,Arial,sans-serif" fill="#203f51">',
    ]

    def text(x, y, s, size=20, bold=False, anchor='middle', color='#203f51'):
        p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(s)}</text>')

    def box(x, y, w, h, title, lines=(), color='#e6eef6', size=21, line_size=18, line_gap=26):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" fill="{color}" stroke="#a5bfce" stroke-width="1.5"/>')
        text(x + w / 2, y + 32, title, size, True)
        for i, line in enumerate(lines):
            text(x + w / 2, y + 61 + line_gap * i, line, line_size)

    def arrow(points, color='#66879a', dash=False):
        import math
        a, b = points[-2:]
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        p.append('<polyline points="' + ' '.join(f'{x},{y}' for x, y in points) + f'" stroke="{color}" stroke-width="2.5" fill="none"' + (' stroke-dasharray="7 5"' if dash else '') + '/>')
        p.append(f'<path d="M0 0L-11 -5L-11 5Z" transform="translate({b[0]} {b[1]}) rotate({angle})" fill="{color}"/>')

    text(40, 48, 'V43：共享证据 → 局部可编辑联合分解 → 按请求编辑与修复', 32, True, 'start')
    text(40, 84, '联合归属、几何、混合与效应；独立世界 GT 不等于每颗 Gaussian 标签；场景判断不接收编辑指令。', 21, False, 'start')
    box(40, 115, 710, 90, '只读原始高斯 / checkpoint / 已登记身份', ['保留原参数、来源索引与 parent–child lineage'])
    box(1310, 115, 450, 90, 'V1 无编辑：原生通路', ['保留已有零差对照；修复另行验收'])
    arrow([(750, 160), (1310, 160)])
    text(1030, 146, '无编辑请求直通', 18)
    box(40, 235, 1720, 72, '合法证据：受限 RGB / 稀疏 LiDAR / 相机时间 / 跟踪 / 重建高斯；完整世界 GT 走隔离评价分支', [], '#eaf1f7', 23)
    arrow([(395, 205), (395, 235)])

    for x, w, title, sub, color in [
        (40, 450, '① 时空编辑与对象管理', '继承 V42 接口 · 保留用户允许域', '#fff1dc'),
        (540, 720, '② 局部可编辑联合分解', 'V43 主线 · 共享证据与耦合表示', '#e0f1ea'),
        (1310, 450, '③ 渲染、背景修复与生成', '继承 V42 大架构 · 生成按需调用', '#fff1dc'),
    ]:
        box(x, 345, w, 1030, title, [sub], color, 24)
        arrow([(x + w / 2, 307), (x + w / 2, 345)])

    box(65, 445, 400, 130, '已有实例 / 连续轨迹', ['黑车 A、白车 B、道路表面 R', '稳定 owner ≠ 当前运动 / 可见性', 'Actor 与自车时钟分别求值'])
    box(65, 625, 400, 112, '编辑请求与用户允许域', ['删除 / 移动 / 插入 / 转向', '目标、时刻、姿态与保留对象'])
    box(65, 787, 400, 138, '高召回 unknown 候选与记账', ['候选由场景证据产生，非操作集合', '候选内背景误召回仅记账', '候选扩大不扩大用户允许操作域'])
    box(65, 975, 400, 138, '按目标查询 1 / 0 / U', ['1 属于目标；0 确认非此目标', 'U 为证据不足、冲突或混合', '原 static 分支 ≠ 语义背景'])
    text(265, 1200, '已有：对象编辑与原生基座', 20, True)
    text(265, 1233, '候选召回、编辑授权与决策分开', 18)
    text(265, 1266, '模型允许保持原状或暂缓操作', 18)
    arrow([(265, 575), (265, 609)], dash=True)
    arrow([(265, 925), (265, 975)])

    box(565, 435, 670, 108, '②a 共享场景证据编码与来源账本', [
        'unknown + 周围可靠点 / 多帧 / 深度 / αT 派生证据',
        '原分支与语义有噪声；不接收 delete / shift 指令',
    ], '#f1faf5', 21)

    # 同一个耦合区域内并列四个头，避免画成先修几何再判归属的流水线。
    box(565, 567, 670, 382, '②b 联合分解：共享潜变量与一致性约束', [
        '四类输出相互约束；可联合优化，也可分阶段训练',
    ], '#e9f5ef', 21)
    box(580, 650, 310, 122, '语义 / 实例 / 运动 / 证据多头', [
        'semantic；instance / surface owner',
        'Actor 局部 / world；置信与冲突',
        '稳定身份与可见性、后验分开',
    ], '#f1faf5', 17, 15, 23)
    box(910, 650, 310, 122, '几何修正与局部重拟合', [
        '独立观测支持位置、尺度与轨迹',
        '错深度不按二维语义硬贴标签',
        '归属、可见性与几何联合约束',
    ], '#fff2dd', 18, 15, 23)
    box(580, 798, 310, 130, 'mixed / 两侧拆分 / 新表面', [
        '保留两侧子高斯与 parent lineage',
        'mixed / 无对应 / 缺测分开',
        '有限新增；不预设逐行唯一 GT',
    ], '#fff2dd', 18, 15, 23)
    box(910, 798, 310, 130, '车辆关联 shadow effect', [
        'vehicle effect + world road receiver',
        '道路几何与车辆关联外观分离',
        '与归属、几何及支撑共同建模',
    ], '#fff2dd', 18, 15, 23)
    # 无先后序：共享编码分发至四类输出；小箭头为双向约束。
    arrow([(900, 543), (900, 567)])
    for xa, xb, yy in [(890, 910, 711), (910, 890, 731), (890, 910, 856), (910, 890, 876)]:
        arrow([(xa, yy), (xb, yy)])
    for xx in [735, 1065]:
        arrow([(xx - 8, 772), (xx - 8, 798)])
        arrow([(xx + 8, 798), (xx + 8, 772)])

    box(565, 975, 670, 112, '②c 编辑决策：联合状态 + 请求 + 用户允许域', [
        '保持 / 改归属 / 修几何 / 拆分 / 新增 / 暂缓',
        '仅此处接收编辑指令；证据不足可不修改模型',
    ], '#e9f5ef', 21)
    arrow([(900, 949), (900, 975)])
    # 独立请求经过左框外侧留白进入决策，不穿过场景判断模块。
    arrow([(465, 682), (515, 682), (515, 1031), (565, 1031)], '#b5843b')
    p.append('<rect x="492" y="745" width="40" height="26" fill="#f5f9fc"/>')
    text(512, 765, '请求', 16, False, 'middle', '#986f31')
    # 候选与周围可靠点进入共享编码；跨过请求线时留断口，表示无连接。
    p.append('<circle cx="515" cy="856" r="6" fill="#f5f9fc"/>')
    arrow([(465, 856), (535, 856), (535, 489), (565, 489)], '#348767')

    box(565, 1111, 670, 114, '②d 验收路线：先最小实际重建，再机制集', [
        '1 普通场景真实训练 → 3–5 机制场景',
        'M1/M2 真值辅助参考，非数学上界 → 小学习',
        '场景 1 仅推理 / 开发，不训练可迁移模块',
    ], '#fff2dd', 21, 16, 21)
    box(565, 1249, 670, 100, '②e 显露背景修复 · 保留 V42 后续组件', [
        '保留正确表面 + 污染修正 + 有来源补层',
        '分步 / assignment-only / joint / joint-detached 配对',
    ], '#f4f1e9', 21)
    # 训练与验收是独立路线，不把它画成推理中的必经阶段。
    arrow([(900, 1249), (900, 1225)], dash=True)

    box(1335, 445, 400, 130, '同相机 / 时刻 / 请求配对', ['原生 C = C_G + (1−α)E', 'E 参数冻结 ≠ 回填贡献固定', '环境不可当 car；无编辑直通 V1'])
    box(1335, 625, 400, 112, '三维结果与四分量诊断', ['保存 C_G / α / (1−α)E / C', '另存 E；统一常量底色诊断'])
    box(1335, 787, 400, 138, '背景修复与按需生成', ['指定显露区域 → 固定 ProPainter', '三维原生与生成结果分别保存', '生成图不转成独立实测或标签'])
    box(1335, 975, 400, 138, '分项验收与强对照', ['归属 / mixed / 几何 / shadow', '保留区损伤、时序稳定与资源', '同证据、容量、更新及后端'])
    text(1535, 1200, '输出：可追溯三维状态 + 编辑视频', 20, True)
    text(1535, 1233, '联合收益与生成收益分别验收', 18)
    text(1535, 1266, '本图为设计；实验须有独立运行凭据', 18)
    for ya, yb in [(575, 625), (737, 787), (925, 975)]:
        arrow([(1535, ya), (1535, yb)])

    # 大组件箭头与折线在框间留白，输出只从编辑决策进入渲染。
    arrow([(1235, 1031), (1285, 1031), (1285, 510), (1335, 510)])
    box(40, 1415, 1720, 112, '贯穿机制：局部可编辑联合状态，保留原始资产与完整来源链', [
        '稳定 owner 与逐时刻可见性 / 置信分开；几何、两侧子高斯、新表面及 effect 保存来源和局部拟合记录。',
        '正序 / 逆序 / 跳帧复用证据；剥离后重算贡献；独立表面 / 深度 / 编辑参考验收，环境参数冻结。',
    ], '#eaf1f7', 23)
    for x in [265, 900, 1535]:
        arrow([(x, 1415), (x, 1375)], dash=True)
    box(40, 1560, 1720, 72, '共同验收：候选召回与归属 → 可编辑表示与几何 → 背景 / 阴影 → 视频质量与成本', [], '#ffffff', 24)
    text(40, 1663, '蓝：继承组件　绿：共享证据与联合状态　黄：表示 / effect 与真值辅助参考　灰：V42 后续补全。', 17, False, 'start', '#5b778a')
    text(40, 1692, '状态：既有探针与历史对照保留；本图为设计，实验状态须有独立运行凭据，不由文档生成推定完成。', 17, False, 'start', '#5b778a')
    p.append('</g></svg>')
    return ''.join(p)


def data_architecture():
    """数据主路线与隔离评价；真值不沿部署输入箭头进入共享编码器。"""
    import math

    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1390" viewBox="0 0 1800 1390" role="img" aria-labelledby="data-title data-desc">',
        '<title id="data-title">V43 Revision5：普通世界、真实重建污染与隔离GT评价</title>',
        '<desc id="data-desc">普通车辆道路灯光世界分别生成受限RGB稀疏LiDAR观测和独立表面对象干预真值。合法观测输入当前重建代码实际训练，自然污染审计后进入V43共享联合模型。独立深度表面与编辑配对只进入隔离评价和明确的真值辅助M1M2诊断，不进入部署特征。先一个最小场景真实重建，再三至五机制场景，再小规模学习批量；人工扰动仅辅助，不承诺自然出现指定错误。阴影是world道路receiver与vehicle effect owner，车有减车无不是纯阴影GT。</desc>',
        '<rect width="1800" height="1390" fill="#f5f9fc"/>',
        '<g font-family="Microsoft YaHei,Arial,sans-serif" fill="#203f51">',
    ]

    def text(x, y, value, size=20, bold=False, anchor='middle', color='#203f51'):
        p.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{"bold" if bold else "normal"}" fill="{color}">{escape(value)}</text>')

    def box(x, y, width, height, title, lines=(), color='#e6eef6', size=22, line_size=18, gap=27, module=None):
        marker = f' data-module="{module}"' if module else ''
        p.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="11" fill="{color}" stroke="#a5bfce" stroke-width="1.5"{marker}/>')
        text(x + width / 2, y + 33, title, size, True)
        for index, line in enumerate(lines):
            text(x + width / 2, y + 64 + index * gap, line, line_size)

    def arrow(points, color='#348767', dash=False):
        a, b = points[-2:]
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        p.append('<polyline points="' + ' '.join(f'{x},{y}' for x, y in points) + f'" stroke="{color}" stroke-width="3" fill="none"' + (' stroke-dasharray="8 5"' if dash else '') + '/>')
        p.append(f'<path d="M0 0L-12 -6L-12 6Z" transform="translate({b[0]} {b[1]}) rotate({angle})" fill="{color}"/>')

    text(40, 48, 'V43 数据路线：普通世界 → 受限观测 → 实际重建 → 联合分解 → 独立核验', 30, True, 'start')
    text(40, 84, 'GT 在世界表面、对象与干预结果上；每颗 Gaussian 不保证唯一对应。评价真值与部署输入分路保存。', 21, False, 'start')

    box(40, 125, 1720, 250, '① 独立仿真世界：现成 / 程序化普通车辆、道路、材质、灯光', [
        '同一世界状态、相机与时刻；保存资产、坐标系、对象 / 表面 ID、轨迹和干预参数',
    ], '#fff1dc', 25, 20, module='world')
    box(65, 215, 900, 130, '普通观察条件，不手造主评测的错误 Gaussian', [
        '有限视角 / 遮挡 / 稀疏度 / 时间轨迹按合同登记',
        '不承诺真实训练一定产生漏归属、错深度或混合',
    ], '#fff9ef', 22)
    box(1010, 215, 725, 130, '独立真值分支', [
        '世界表面 / 对象 / 深度 / 可见性 + 对象级编辑干预',
        '完整网格与稠密 GT 不进入部署输入',
    ], '#f5ebf8', 22)

    box(40, 430, 1120, 445, '② 受限观测与当前重建代码：主污染来源', [
        '不以干净表面直接构造 Gaussian；实际训练、checkpoint、损失和预算可追溯',
    ], '#e0f1ea', 25, 19, module='reconstruction')
    box(65, 535, 1070, 115, '受限 RGB + 稀疏 LiDAR + 相机 / 时间', [
        '已有跟踪 / mask / 语义也只作为带噪观测证据',
        '输入清单与评价 GT 清单分开，禁止测试答案泄漏',
    ], '#f1faf5', 22)
    box(65, 700, 1070, 145, '当前重建代码真实训练 → 自然污染审计', [
        '保留初始化、活跃约束、seed、采样、更新与适配差异',
        '错深度 / mixed / 无对应 / 缺测分别检查；原分支与网页语义不是 GT',
        '人工标签 / 参数扰动只作辅助机制控制，独立记账',
    ], '#f1faf5', 22, 18, 26)
    # 跨模块箭头停在大框边界，避免穿过标题和说明文字。
    arrow([(515, 345), (515, 405), (600, 405), (600, 430)])
    arrow([(600, 650), (600, 700)])
    text(815, 684, '实际训练', 18)

    box(40, 930, 1120, 250, '③ V43：共享证据、多头联合状态与按请求决策', [
        '部署只接合法观测、真实重建、来源账本与可计算贡献',
    ], '#e0f1ea', 25, 19, module='v43')
    box(65, 1030, 745, 120, '归属 + 几何 / mixed + effect + 不确定性', [
        '场景判断不接收 delete / shift / rotate',
        '请求只进决策；有预算修复、拆分、新增或暂缓',
    ], '#f1faf5', 21, 17, 25)
    box(835, 1030, 300, 120, '输出与来源', [
        '三维状态 / 编辑结果',
        '参数、动作、资源记录',
    ], '#eaf1f7', 21, 17, 25)
    arrow([(600, 845), (600, 930)])
    text(805, 909, '真实重建与证据', 18)
    arrow([(810, 1090), (835, 1090)])

    box(1220, 430, 540, 750, '④ 隔离 GT 参考与评价', [
        '真值不进入部署特征',
    ], '#eee5f3', 25, 20, module='evaluation')
    box(1245, 535, 490, 170, '独立表面 / 深度 / 对象参考', [
        '对应须核对深度、表面与可见性',
        'αT × 像素语义只是派生责任',
        '错深度 / mixed / 无对应 / 缺测分开',
        '不强迫每颗 Gaussian 贴硬标签',
    ], '#faf5fc', 21, 17, 25)
    box(1245, 750, 490, 170, '独立配对编辑与 shadow4', [
        '删车 / 横移 / 转向后重新渲染世界',
        'world road receiver + vehicle effect',
        '体可见与投影可分别控制',
        '有车减无车不是纯阴影 GT',
    ], '#faf5fc', 21, 17, 25)
    box(1245, 965, 490, 185, '表面、编辑结果与公平性核验', [
        '残留 / 深度 / 保留域 / 完成率',
        'M1/M2 真值辅助参考，非数学上界',
        '列额外信息、容量与优化预算',
        '完整干净参考不作公平方法胜利',
    ], '#faf5fc', 21, 17, 25)
    arrow([(1372, 345), (1372, 405), (1490, 405), (1490, 430)], '#9a71a5')
    arrow([(1490, 705), (1490, 750)], '#9a71a5')
    arrow([(1490, 920), (1490, 965)], '#9a71a5')
    arrow([(1135, 1090), (1220, 1090), (1245, 1090)], '#66879a')
    text(1190, 1067, '核验', 16)

    box(40, 1220, 1720, 100, '执行门槛：1 个最小场景实际重建 → 3–5 机制场景 → 小学习 / 批量', [
        '训练 GT 仅监督训练分区；测试 GT 不调优实际方法。真实三场景无完整 GT；场景 1 仅开发 / 推理，不训练可迁移模块。',
    ], '#eaf1f7', 25, 19)
    text(40, 1355, '本图为协议设计：自然污染不足就报告不足；人工 / subagent 参考不是完美 GT；运行完成须有独立凭据。', 18, False, 'start', '#5b778a')
    p.append('</g></svg>')
    return ''.join(p)
