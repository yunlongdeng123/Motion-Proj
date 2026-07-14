# P2-V2 V5 object-only 人工复核

每个 panel 为 `[Base rollout | Projected object-only | Object mask（static 已禁用） | abs difference]`。
判 `yes`：局部 object 修正与可见运动/支撑关系相符，且未引入明显撕裂或主体破坏；判 `no`：局部修正明显错位、漂移或破坏主体；判 `uncertain`：无法可靠判断。

复制 `reviews.template.jsonl` 为 `reviews.jsonl`，填写全部 20 条后运行同一命令附加 `--aggregate-only`。门槛：decisive verdict 的 yes 比例不低于 70%；未通过不得训练。
