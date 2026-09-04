# WorldSim V7.1 M36 — Anchor/Child Optical Contribution Decomposition

日期：2026-09-05  
状态：frozen

M35证明learned authority相对unit transmittance方向正确，但整体被unit optical thickness拖垮。M36冻结M35模型、
M8 geometry与66-Actor holdout，分别测量五个不训练的transmittance arm：unit anchors only、learned anchors
only、unit children only、unit all、learned anchors + unit children；原unit-energy仍为参照。

该分解只定位采样密度/重叠opacity来自observed anchors、completion children还是二者组合。不调scale、segment、
threshold或模型，不产生性能候选，不读external/M21 partial。结果决定下一步是anchor surface-measure calibration、
completion opacity supervision或共同coverage normalization。下一failure ID=`V71-F38`。
