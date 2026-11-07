# v44 最终交付 QA

状态：PASS_FINAL_V44_INDEPENDENT_DELIVERY_QA。无阻塞发现。

正式49帧主页面与独立四帧pilot页面已完成只读实物核查。390个发布数据／收据文件的字节数与SHA256一致；主页面及静态逐帧页1081个src/href引用、253个HTML fragment目标有效，pilot对应130个引用、27个fragment有效。主页面196个成对PNG与294个静态图片映射、pilot 16个成对PNG与24个静态图片映射正确。

两页面初始raw113/delete三列均使用真实相对PNG src，无需JavaScript。raw89–137全部49帧的keep/delete与current/v44共用继承的同帧crop；静态图src与内嵌manifest逐项匹配。控制器JavaScript与已通过848次CPU模拟选择的版本同SHA，含失败保留旧整组、图片尺寸／解码失败与过期请求保护。

原蓝ROI两阶段共16条空域记录保留pixels=0和mean=null，页面显示“—”／“空域不可评价”，没有当作误差为0或去蓝通过。补充道路与背景域、pilot三次delete状态分层重放、历史旧M2证据分别标明。银残片按用户最新范围延期，诊断折叠保存。full49和pilot分别绑定自己的raw列表／mode／comparison SHA审查；旧pilot暂缓判断仅作历史收据。两页human verdict与quality_accepted均为null，没有将engine pass代填为人工接受。

原CPU index按原字节保存在cpu_stage.html，SHA256为d060141ae5ce85be51403bf24719c048b1304b2c044145e88f3753b60a7c393d。组件图存在且包含输入、冻结车体、已保存背景、统一排序与输出。包内无float NPZ。

冻结实物：

- index.html SHA256：8977d9be9b596c88860f79816f0c5c1da27d9cab22950fcf8985502dfd8dd6bc
- pilot.html SHA256：77874ed298d91ce0316ffc9260875bdce228dff0ef0467894563982e1b765a8b

限制：本QA仅静态文件／HTML核查与CPU/JS mock，未使用浏览器、CDP、HTTP桥、GPU或电源操作，也未观察真实浏览器布局或作人工画质接受结论。未修改报告或实验代码，唯一写入为本目录审查文件。

证据：final_independent_qa.json、js_mock_review.json、review_scope_mock.json、original_cpu_snapshot.json。preview_independent_qa.json为先前pilot预览检查，不替代正式49帧收据。
