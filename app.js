"use strict";
const split = document.getElementById("split");
split.addEventListener("input", () => {
  document.getElementById("road-comparison").style.setProperty("--split", `${split.value}%`);
  document.getElementById("split-output").value = `${split.value}%`;
});
const values = {100:["0.008189","0.006780"],110:["0.008336","0.008122"],117:["0.008325","0.008391"]};
document.querySelectorAll("[data-frame]").forEach(button => {
  button.addEventListener("click", () => {
    const frame = button.dataset.frame;
    document.querySelectorAll("[data-frame]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
    for (const [name, description] of [["photo","实拍路面局部"],["two","两帧监督结果"],["eight","八帧监督结果"]]) {
      const img = document.getElementById(`query-${name}`);
      img.src = `assets/query-${frame}-${name}.png`;
      img.alt = `查询视角${frame}的${description}`;
    }
    document.getElementById("query-status").textContent = `视角 ${frame} · 颜色 MAE：${values[frame][0]} → ${values[frame][1]}`;
  });
});
