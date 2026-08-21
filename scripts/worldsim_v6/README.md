# WorldSim V6 脚本

R1–R4 的只读审计、schema 验证、确定性 replay 与 artifact manifest 入口放在本目录。脚本不得绕过
`AUTORESEARCH_STATE.json` 的 active task、confirmation lock 或单重进程资源约束。
