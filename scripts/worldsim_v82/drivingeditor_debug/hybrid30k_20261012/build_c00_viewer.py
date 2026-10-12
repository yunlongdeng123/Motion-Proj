"""把本地帧清单嵌入离线 C00 分支查看器；图片继续保留在 assets/。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>__TITLE__</title>
  <style>
    :root{color-scheme:dark;--bg:#090e17;--panel:#101824;--panel2:#131e2c;--line:#273449;--text:#edf3fb;--muted:#9aacc3;--faint:#677b96;--accent:#65dbca;--static:#b5a0ff;--dynamic:#ffba80;--radius:15px}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 6% 0%,#142238 0%,transparent 40%),var(--bg);font-family:Inter,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--text);font-size:14px;line-height:1.55}button,input,select{font:inherit}button,select{color:var(--text)}button{cursor:pointer}button:disabled{cursor:default;opacity:.35}button:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:4px}button{touch-action:manipulation}button svg{display:block;width:18px;height:18px}a{color:var(--accent)}[hidden]{display:none!important}.shell{max-width:1160px;margin:auto;padding:30px 26px 50px}.eyebrow{font-size:11px;color:var(--accent);letter-spacing:.2em;font-weight:700;text-transform:uppercase}.header{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.header h1{font-size:31px;font-weight:650;letter-spacing:-.03em;margin:7px 0 5px;line-height:1.25}.subtitle{color:var(--muted);margin:0;font-size:13px}.status{display:flex;align-items:center;gap:7px;color:#bcecdf;background:#142e2b;border:1px solid #284b43;white-space:nowrap;border-radius:99px;padding:6px 11px;font-size:12px;margin-top:20px}.status i{width:6px;height:6px;border-radius:50%;background:#82dfae}.meta-line{display:flex;flex-wrap:wrap;align-items:center;gap:6px 15px;color:var(--muted);font-size:12px;margin-top:17px}.meta-item{white-space:nowrap}.meta-item strong{font-weight:550;color:#cdd8e8}.badge{border:1px solid #344155;border-radius:5px;padding:2px 7px;font-size:11px;color:#c0cadd}.scope{margin:18px 0 0;padding:10px 13px;border-left:2px solid #587694;background:#142033;border-radius:0 8px 8px 0;color:#b3c3d8;font-size:12px}.scope b{color:#d4deeb;font-weight:550}.overview{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin:19px 0 23px}.stat{padding:13px 15px;background:linear-gradient(135deg,#152235,#101925);border:1px solid var(--line);border-radius:var(--radius);min-height:87px}.stat .label{font-size:11px;color:var(--muted)}.stat .value{font-variant-numeric:tabular-nums;font-size:24px;line-height:1.35;font-weight:600;margin:3px 0 0;letter-spacing:-.025em}.stat .unit{font-size:12px;color:var(--muted);font-weight:400;margin-left:5px}.stat .detail{font-size:10px;color:var(--faint);margin-top:3px}.tabs{display:flex;align-items:center;gap:22px;border-bottom:1px solid var(--line);margin-bottom:16px}.tab{background:none;border:0;border-bottom:2px solid transparent;margin-bottom:-1px;padding:10px 0;color:var(--muted);font-weight:550;font-size:13px}.tab[aria-selected=true]{border-color:var(--accent);color:var(--accent)}.tab-count{font-size:10px;background:#243246;border-radius:4px;color:#b3c5dd;padding:2px 5px;margin-left:5px}.tab[aria-selected=true] .tab-count{background:#213d3f;color:#a4e7db}.tabs-note{margin-left:auto;font-size:11px;color:var(--faint)}.controls{border:1px solid var(--line);border-radius:var(--radius);background:#101a28;padding:13px 15px;margin-bottom:15px}.control-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.navigation{display:flex;align-items:center;gap:6px}.icon-button,.button{display:inline-flex;align-items:center;justify-content:center;border:1px solid #324158;border-radius:8px;background:#192537;height:34px;padding:0 10px;gap:6px;white-space:nowrap;font-size:12px}.icon-button{width:34px;padding:0}.button:hover,.icon-button:hover{background:#22334a;border-color:#54708c}.play{background:#153b37;color:#a8f0df;border-color:#306156;min-width:80px}.play:hover{background:#1c4a44;border-color:#4e8a7b}.play.active{background:#225449}.frame-label{font-size:14px;font-weight:600;line-height:1.35;font-variant-numeric:tabular-nums}.frame-meta{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;margin-top:2px}.frame-info{min-width:175px}.spacer{flex:1}.fps-label{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted)}select{background:#152135;border:1px solid #33425a;border-radius:6px;padding:5px 6px;font-size:12px;height:30px}.segmented{display:flex;padding:3px;background:#0b1320;border:1px solid #26354a;border-radius:8px;gap:2px}.segmented button{background:none;border:0;border-radius:5px;padding:4px 9px;color:var(--muted);font-size:11px}.segmented button[aria-pressed=true]{background:#2a3c53;color:var(--text)}.segmented button:disabled{opacity:.3}.scrub-row{display:flex;align-items:center;gap:12px;margin-top:13px}.range-caption{font-variant-numeric:tabular-nums;font-size:10px;color:var(--faint);min-width:14px}.range-caption:last-child{text-align:right}input[type=range]{width:100%;height:5px;appearance:none;-webkit-appearance:none;background:#2b3e57;border-radius:20px;cursor:pointer;accent-color:var(--accent)}input[type=range]::-webkit-slider-thumb{appearance:none;-webkit-appearance:none;border:2px solid #0d1c24;height:15px;width:15px;background:var(--accent);border-radius:50%;box-shadow:0 0 0 3px #65dbca16}input[type=range]::-moz-range-thumb{border:2px solid #0d1c24;height:12px;width:12px;background:var(--accent);border-radius:50%}.control-hint{display:flex;justify-content:space-between;align-items:center;margin-top:9px;font-size:10px;color:var(--faint);gap:12px}.control-hint span{line-height:1.4}.control-hint .roi-note{color:#b8c6d8}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.image-card{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);overflow:hidden}.card-heading{display:flex;align-items:center;gap:8px;padding:12px 13px 10px;min-height:47px}.branch-dot{width:7px;height:7px;border-radius:50%;background:#9eafc6;flex-shrink:0}.image-card[data-key=composite] .branch-dot{background:var(--accent)}.image-card[data-key=static] .branch-dot,.image-card[data-key=alpha_s] .branch-dot{background:var(--static)}.image-card[data-key=dynamic] .branch-dot,.image-card[data-key=alpha_d] .branch-dot{background:var(--dynamic)}.card-heading h2{font-size:13px;line-height:1;margin:0;font-weight:600}.card-tag{margin-left:auto;font-size:10px;color:var(--muted);white-space:nowrap}.image-button{display:block;width:100%;padding:0;border:0;background:#05080d;color:inherit;text-align:inherit;cursor:zoom-in}.image-surface{position:relative;overflow:hidden;width:100%;aspect-ratio:3/2;background:#04070a;isolation:isolate}.image-surface img{position:absolute;display:block;max-width:none;width:100%;height:auto;left:0;top:0;user-select:none;-webkit-user-drag:none}.image-surface .empty{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:var(--faint);font-size:12px;padding:18px;background:#0a111b}.image-surface .empty[hidden]{display:none}.image-button:hover img{opacity:.98}.card-footer{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:7px 13px;color:var(--faint);font-size:10px;background:#101925}.card-footer .zoom-hint{white-space:nowrap;color:#8fa4bf}.image-card[data-key=dynamic] .card-footer{color:#d0a37c}.section-title{font-size:12px;color:#c5d1e1;display:flex;justify-content:space-between;align-items:center;margin:20px 0 9px;font-weight:550}.section-title small{font-size:10px;color:var(--faint);font-weight:400}.metrics-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.metric-card{padding:12px 14px;border:1px solid #26364b;background:#111c2b;border-radius:11px}.metric-card .region{color:#9cafc8;font-size:11px}.metric-card .metric-value{font-size:23px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.35;margin:3px 0}.metric-card .metric-value span{font-size:11px;font-weight:400;color:var(--muted);margin-left:4px}.metric-card .metric-description{font-size:10px;color:#788da9}.metric-pairs{margin-top:9px;padding-top:7px;border-top:1px solid #26354a;display:grid;gap:3px}.metric-pairs div{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);gap:5px}.metric-pairs strong{font-weight:500;font-variant-numeric:tabular-nums;color:#c9d5e5}.metrics-footnote{font-size:10px;color:var(--faint);margin:8px 2px 0}.fold{border:1px solid #263448;background:#0e1723;border-radius:12px;margin-top:16px}.fold>summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:9px;padding:13px 15px;font-size:12px;color:#c1cde0}.fold>summary::-webkit-details-marker{display:none}.fold>summary::before{content:'›';font-size:19px;line-height:1;color:#8096b2;transform:rotate(0);transition:transform .15s}.fold[open]>summary::before{transform:rotate(90deg)}.summary-detail{color:var(--faint);font-size:10px;margin-left:auto}.fold-body{padding:0 15px 15px}.diagnostic-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.diagnostic-grid .card-heading{padding:10px 10px 8px;min-height:39px}.diagnostic-grid .card-heading h2{font-size:11px}.diagnostic-grid .card-tag{font-size:9px}.diagnostic-grid .card-footer{font-size:9px;padding:6px 10px}.diagnostic-note{color:#a1b3ca;font-size:11px;padding:0 1px 11px}.scale-bar{display:inline-flex;align-items:center;gap:5px;color:var(--muted);margin-left:9px;font-variant-numeric:tabular-nums}.scale-bar i{display:inline-block;width:80px;height:7px;background:linear-gradient(to right,#000,#fff);border:1px solid #566171;border-radius:2px}.frames-grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:6px}.frame-chip{border:1px solid #2b3c53;border-radius:7px;background:#152034;text-align:left;padding:7px 8px;color:#adc0d8;min-width:0}.frame-chip span{display:block;font-size:10px;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.frame-chip small{font-size:9px;color:#6f86a4;font-variant-numeric:tabular-nums}.frame-chip:hover{background:#20334a;border-color:#55738d}.frame-chip[aria-current=true]{border-color:#499889;background:#193832;color:#c1f1e7}.frame-chip[aria-current=true] small{color:#89b7ae}.time-notice{font-size:12px;background:#1b1c33;border:1px solid #3a3356;color:#c9c4e7;border-radius:11px;padding:11px 14px;margin-bottom:15px}.time-notice strong{color:#e0daf8;font-weight:550}.time-controls .control-hint{color:#aaa0c5}.about-text{font-size:11px;color:var(--muted);margin:0 0 13px}.architecture{display:grid;grid-template-columns:1.05fr 30px 1.35fr 30px 1.15fr;align-items:center;gap:8px;padding:14px;border-radius:10px;border:1px solid #283951;background:#101c2b}.arch-module{border:1px solid #3c526f;border-radius:8px;background:#18283b;padding:12px;text-align:center;font-size:11px;color:#d4deec;line-height:1.7}.arch-module small{display:block;color:#7f96b4;font-size:9px}.arch-components{display:grid;grid-template-columns:1fr 1fr;gap:7px}.arch-branch{border:1px solid #6b598d;background:#24253b;border-radius:7px;text-align:center;padding:9px 4px;font-size:11px;color:#cdc1ff}.arch-branch.dynamic{border-color:#775b43;background:#30251d;color:#ecc199}.arch-env{grid-column:1/-1;text-align:center;border:1px solid #3d586b;background:#152e38;color:#a8d8e4;border-radius:6px;padding:4px;font-size:10px}.arch-arrow{text-align:center;color:#718aaa;font-size:25px}.arch-output{display:grid;gap:5px}.arch-output span{display:block;border:1px solid #344b63;border-radius:5px;padding:4px 7px;font-size:10px;color:#b9cbe1;background:#152438}.about-notes{display:grid;gap:8px;margin-top:14px;color:#9caec6;font-size:11px}.about-notes p{margin:0}.about-notes b{font-weight:550;color:#cfdaea}.source-info{color:#657d9c;overflow-wrap:anywhere;font-size:10px;margin-top:12px}.footer{display:flex;justify-content:space-between;gap:14px;border-top:1px solid #203048;margin-top:24px;padding-top:13px;font-size:10px;color:var(--faint)}kbd{display:inline-block;padding:0 4px;border:1px solid #354459;border-radius:3px;color:#afbdd0;background:#152030;font-family:inherit;font-size:10px}.modal{width:min(1120px,calc(100vw - 36px));max-height:calc(100vh - 36px);padding:0;border:1px solid #405573;border-radius:15px;background:#101925;color:var(--text);box-shadow:0 25px 100px #0009;overflow:auto}.modal::backdrop{background:#030711ed;backdrop-filter:blur(5px)}.modal-top{display:flex;align-items:center;gap:10px;padding:14px 17px;border-bottom:1px solid #2e4058}.modal-heading{font-size:15px;font-weight:600}.modal-meta{font-size:11px;color:var(--muted);margin-left:5px;font-variant-numeric:tabular-nums}.modal-branches{display:flex;flex-wrap:wrap;gap:5px;padding:12px 17px 0}.modal-branches button{border:1px solid #354962;border-radius:6px;color:#a4b9d3;background:#172437;padding:4px 8px;font-size:11px}.modal-branches button[aria-pressed=true]{border-color:#4b978a;background:#1e403b;color:#bfeee4}.modal-image-wrap{overflow:auto;display:flex;align-items:center;justify-content:center;min-height:250px;padding:20px;background:#05090f;margin-top:12px}.modal-image-wrap .image-surface{width:480px;flex-shrink:0;max-width:100%}.modal-image-wrap.fit .image-surface{width:100%}.modal-bottom{display:flex;align-items:center;gap:8px;padding:12px 17px}.modal-caption{font-size:10px;color:var(--muted);margin-left:4px}.modal-zoom{margin-left:auto}.noscript{border:1px solid #b88b45;background:#312619;color:#f0d3a8;border-radius:8px;margin:12px 0;padding:12px}
    @media(min-width:1500px){.shell{max-width:1240px}}
    @media(max-width:900px){.shell{padding:22px 20px 38px}.header h1{font-size:27px}.controls{padding:12px}.overview{gap:8px}.stat{padding:12px}.stat .value{font-size:22px}.card-heading{padding:10px 11px}.card-tag{font-size:9px}.diagnostic-grid{gap:8px}.diagnostic-grid .card-tag{display:none}.frames-grid{grid-template-columns:repeat(7,minmax(0,1fr))}.tabs-note{display:none}.frame-info{min-width:145px}.fps-label{margin-left:auto}.spacer{display:none}.segmented{margin-left:auto}.summary-detail{max-width:55%;text-align:right}.modal-meta{display:block;margin-left:0;margin-top:2px}.modal-top{align-items:flex-start}.modal-top .spacer{display:block}}
    @media(max-width:630px){.shell{padding:18px 13px 28px}.header{gap:10px}.header h1{font-size:23px}.eyebrow{font-size:9px;letter-spacing:.13em}.status{font-size:10px;padding:5px 8px;margin-top:16px}.subtitle{font-size:11px}.scope{font-size:11px;margin-top:13px}.meta-line{font-size:10px;gap:6px 10px;margin-top:12px}.overview{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:14px 0 13px}.stat{min-height:76px;padding:10px 12px}.stat .value{font-size:22px}.stat .label{font-size:10px}.stat .detail{font-size:9px}.grid{gap:10px;grid-template-columns:1fr}.card-tag{font-size:10px}.control-row{gap:8px}.frame-info{min-width:0;flex:1}.frame-label{font-size:12px}.frame-meta{font-size:9px}.fps-label{margin-left:0}.segmented{margin-left:auto}.controls{padding:11px}.control-hint{font-size:9px;align-items:flex-start}.control-hint span:first-child{max-width:63%}.tabs{gap:20px;margin-bottom:12px}.tab{font-size:12px}.metrics-grid{grid-template-columns:1fr;gap:7px}.metric-card{padding:10px 13px;display:grid;grid-template-columns:1fr auto;align-items:center;column-gap:14px}.metric-card .metric-value{grid-column:2;grid-row:1/3;font-size:24px;margin:0}.metric-pairs{grid-column:1/-1;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:7px}.metric-pairs div{font-size:9px;flex-wrap:wrap;column-gap:5px}.diagnostic-grid{grid-template-columns:1fr}.diagnostic-grid .card-heading h2{font-size:12px}.diagnostic-grid .card-tag{display:block}.diagnostic-grid .card-footer{font-size:10px}.diagnostic-note{font-size:10px}.scale-bar{margin-left:0;margin-top:3px;display:flex}.frames-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:5px}.frame-chip{padding:6px}.summary-detail{font-size:9px}.fold>summary{padding:11px 12px}.fold-body{padding:0 12px 12px}.architecture{grid-template-columns:1fr;gap:5px;padding:12px}.arch-arrow{transform:rotate(90deg);line-height:1;font-size:22px}.arch-components{gap:6px}.arch-module{padding:7px}.arch-output{grid-template-columns:repeat(3,1fr)}.arch-output span{text-align:center;font-size:9px;padding:5px}.footer{font-size:9px;flex-wrap:wrap;margin-top:18px}.modal{width:calc(100vw - 18px);max-height:calc(100vh - 18px);border-radius:10px}.modal-top{padding:11px}.modal-heading{font-size:12px}.modal-meta{font-size:9px}.modal-branches{padding:9px 11px 0}.modal-branches button{font-size:10px}.modal-image-wrap{padding:8px;min-height:180px}.modal-bottom{padding:10px 11px;flex-wrap:wrap}.modal-caption{order:3;width:100%;margin-left:0}.modal-zoom{margin-left:auto}.section-title{font-size:11px}.time-notice{font-size:11px;padding:10px 12px}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
  </style>
</head>
<body>
<main class="shell">
  <header class="header">
    <div><div class="eyebrow">DrivingEditor · branch inspection</div><h1 id="page-title">C00 · 30k 分支查看器</h1><p class="subtitle" id="page-subtitle">静态、动态与合成结果，同一帧同步查看</p></div>
    <div class="status"><i aria-hidden="true"></i>30k 已完成</div>
  </header>
  <div class="meta-line" id="metadata"><span class="badge">SGCM 关闭</span><span class="badge">NPS 关闭</span></div>
  <p class="scope"><b>49 帧全观测开发重建</b> · body-only mask 无阴影；此处查看分支与重建结果，完整论文复现及编辑质量尚未验收。</p>
  <noscript><div class="noscript">请允许浏览器运行本地 JavaScript，以便同步切换帧。所有图片均在本地 assets/ 目录。</div></noscript>
  <div class="overview" id="overview"></div>
  <nav class="tabs" role="tablist" aria-label="查看模式">
    <button class="tab" id="tab-reconstruction" role="tab" aria-selected="true" aria-controls="reconstruction" data-tab="reconstruction">分支重建<span class="tab-count" id="reconstruction-count"></span></button>
    <button class="tab" id="tab-time" role="tab" aria-selected="false" aria-controls="time-panel" data-tab="time" hidden>固定相机 · 时间控制<span class="tab-count" id="time-count"></span></button>
    <span class="tabs-note">RGB 固定范围 [0, 1] · 无逐图归一化</span>
  </nav>
  <section id="reconstruction" role="tabpanel" aria-labelledby="tab-reconstruction">
    <div class="controls" data-controls="reconstruction">
      <div class="control-row">
        <div class="navigation"><button class="icon-button prev" aria-label="上一帧"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m14 6-6 6 6 6"/></svg></button><button class="button play"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg><span>播放</span></button><button class="icon-button next" aria-label="下一帧"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m10 6 6 6-6 6"/></svg></button></div>
        <div class="frame-info"><div class="frame-label"></div><div class="frame-meta"></div></div><span class="spacer"></span>
        <label class="fps-label">速度 <select class="fps" aria-label="播放帧率"><option value="6">6 FPS</option><option value="10">10 FPS</option></select></label>
        <div class="segmented" aria-label="图像范围"><button data-view="full" aria-pressed="true">原图</button><button data-view="roi" aria-pressed="false">车体 ROI</button></div>
      </div>
      <div class="scrub-row"><span class="range-caption">1</span><input class="slider" type="range" min="0" max="48" value="0" step="1" aria-label="重建帧"><span class="range-caption frame-total">49</span></div>
      <div class="control-hint"><span class="roi-note">同帧四卡同步 · 点击图片查看原始大图</span><span><kbd>←</kbd> <kbd>→</kbd> 切帧 · <kbd>空格</kbd> 播放</span></div>
    </div>
    <div class="grid" id="reconstruction-grid"></div>
    <div class="section-title">当前帧 · 区域指标 <small>PSNR 使用真实 GT；透明度取原始未裁 alpha</small></div>
    <div class="metrics-grid" id="frame-metrics"></div>
    <p class="metrics-footnote">背景 = 非天空且车体外。RGB 指标先 clip 到 [0, 1]；空区域或缺失指标显示 —。</p>
    <details class="fold" id="diagnostics"><summary>透明度与车体掩码<span class="summary-detail">α 静态 / α 动态 / body mask</span></summary><div class="fold-body"><div class="diagnostic-note">原始 alpha，未乘 body mask，也未逐图归一化。<span class="scale-bar">0 <i aria-hidden="true"></i> 1 <span>黑 = 透明 · 白 = 不透明</span></span></div><div class="diagnostic-grid" id="diagnostic-grid"></div></div></details>
    <details class="fold"><summary>全部观测帧<span class="summary-detail" id="frames-list-count"></span></summary><div class="fold-body"><div class="frames-grid" id="frames-list"></div></div></details>
  </section>
  <section id="time-panel" role="tabpanel" aria-labelledby="tab-time" hidden>
    <div class="time-notice" id="time-notice"></div>
    <div class="controls time-controls" data-controls="time">
      <div class="control-row">
        <div class="navigation"><button class="icon-button prev" aria-label="上一个模型时间"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m14 6-6 6 6 6"/></svg></button><button class="button play"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg><span>播放</span></button><button class="icon-button next" aria-label="下一个模型时间"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m10 6 6 6-6 6"/></svg></button></div>
        <div class="frame-info"><div class="frame-label"></div><div class="frame-meta"></div></div><span class="spacer"></span>
        <label class="fps-label">速度 <select class="fps" aria-label="时间控制播放帧率"><option value="6">6 FPS</option><option value="10">10 FPS</option></select></label>
      </div>
      <div class="scrub-row"><span class="range-caption">1</span><input class="slider" type="range" min="0" max="48" value="0" step="1" aria-label="模型时间"><span class="range-caption frame-total">49</span></div>
      <div class="control-hint"><span>固定相机位姿与光线 · 仅改变 model time</span><span>本组无对应 GT</span></div>
    </div>
    <div class="grid" id="time-grid"></div>
    <div class="section-title">相对锚点时间的静态变化 <small>差异指标；不作为重建质量分数</small></div>
    <div class="metrics-grid" id="time-metrics"></div>
    <details class="fold"><summary>全部模型时间<span class="summary-detail" id="time-list-count"></span></summary><div class="fold-body"><div class="frames-grid" id="time-list"></div></div></details>
  </section>
  <details class="fold" id="about"><summary>关于 C00 与数据范围<span class="summary-detail">组件图 / 说明 / 来源</span></summary><div class="fold-body">
    <p class="about-text">C00 关闭 SGCM 与 NPS，继承 2024 公开代码的活跃目标和约束。分支与合成图取自同一实际渲染前向。</p>
    <div class="architecture" role="img" aria-label="RGB、LiDAR、body mask输入两分支高斯与EnvMap，输出静态、动态和合成结果">
      <div class="arch-module">RGB · LiDAR<br>body mask<small>观测与监督</small></div><div class="arch-arrow" aria-hidden="true">→</div>
      <div class="arch-components"><div class="arch-branch">静态高斯</div><div class="arch-branch dynamic">动态高斯</div><div class="arch-env">EnvMap</div></div><div class="arch-arrow" aria-hidden="true">→</div>
      <div class="arch-output"><span>静态 RGB / α</span><span>动态 RGB / α</span><span>合成 RGB</span></div>
    </div>
    <div class="about-notes"><p><b>显示规则</b>　RGB 使用固定 [0, 1] 范围转为 8-bit；动态分支没有预乘车体 mask。alpha 使用固定 [0, 1] 灰度，body mask 独立展示。</p><p><b>范围</b>　49 帧是全观测开发重建；body-only mask 不含阴影。此结果不等同于完整论文复现、留出评估或编辑质量验收。</p><p><b>分支查看</b>　分支分离不等同于车位补景或静态性已验收；请直接查看静态背景、残影及固定相机时间变化。</p><p id="about-time" hidden><b>时间控制</b>　相机与光线固定，只替换为各观测帧的 model time。时间与位移采用模型单位，不能解释为物理秒或速度；反事实组合没有对应 GT。</p></div>
    <div class="source-info" id="source-info"></div>
  </div></details>
  <footer class="footer"><span>本地离线查看 · index.html + assets/</span><span>分支对齐，范围固定，指标按真实区域读取</span></footer>
</main>
<dialog class="modal" id="lightbox" aria-labelledby="lightbox-title">
  <div class="modal-top"><div><span class="modal-heading" id="lightbox-title"></span><span class="modal-meta" id="lightbox-meta"></span></div><span class="spacer"></span><button class="icon-button" id="close-lightbox" aria-label="关闭大图"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m6 6 12 12M18 6 6 18"/></svg></button></div>
  <div class="modal-branches" id="modal-branches"></div>
  <div class="modal-image-wrap" id="modal-image-wrap"><div class="image-surface" id="modal-image-surface"><img id="modal-image" alt=""><span class="empty" hidden>当前帧没有这张图</span></div></div>
  <div class="modal-bottom"><button class="icon-button" id="modal-prev" aria-label="上一帧"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m14 6-6 6 6 6"/></svg></button><button class="icon-button" id="modal-next" aria-label="下一帧"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="m10 6 6 6-6 6"/></svg></button><span class="modal-caption" id="modal-caption"></span><div class="segmented modal-zoom"><button id="zoom-native" aria-pressed="true">原始 100%</button><button id="zoom-fit" aria-pressed="false">适合屏幕</button></div></div>
</dialog>
<script id="viewer-data" type="application/json">__MANIFEST__</script>
<script>
'use strict';
(() => {
  const data = JSON.parse(document.getElementById('viewer-data').textContent);
  const frames = Array.isArray(data.frames) ? data.frames : [];
  const timeFrames = Array.isArray(data.static_time_frames) ? data.static_time_frames : [];
  const summary = data.summary || {};
  const timeSummary = data.static_time_summary || {};
  const width = number(data.width) || 480, height = number(data.height) || 320;
  const state = {tab:'reconstruction', indices:{reconstruction:0,time:0}, fps:6, playing:false, timer:null, roi:false, modalKey:'gt', fit:false};
  const cards = {reconstruction:[],time:[]};
  const branches = {
    gt:{label:'GT 真值',tag:'观测 RGB',footer:'同帧原始观测',short:'GT'},
    composite:{label:'合成结果',tag:'Composite',footer:'静态 + 动态 + EnvMap',short:'合成'},
    static:{label:'静态分支',tag:'Static RGB',footer:'含 EnvMap 背景',short:'静态'},
    dynamic:{label:'动态分支',tag:'Dynamic RGB',footer:'原始分支 RGB · 未乘 body mask',short:'动态'},
    alpha_s:{label:'静态透明度',tag:'α static',footer:'原始 α · 黑 0 / 白 1',short:'α 静态'},
    alpha_d:{label:'动态透明度',tag:'α dynamic',footer:'原始 α · 黑 0 / 白 1',short:'α 动态'},
    body_mask:{label:'车体掩码',tag:'Body mask',footer:'车体为白色 · 无阴影',short:'车体 mask'}
  };
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const list = tab => tab === 'time' ? timeFrames : frames;
  const current = tab => list(tab || state.tab)[state.indices[tab || state.tab]] || {};
  const frameId = frame => String(frame.id ?? frame.frame_id ?? frame.source_timestamp_frame ?? '—');
  function number(value){if(value === null || value === undefined || value === '' || typeof value === 'boolean')return null;const n=Number(value);return Number.isFinite(n)?n:null;}
  function fmt(value,digits=2){const n=number(value);return n===null?'—':n.toFixed(digits);}
  function compact(value){const n=number(value);if(n===null)return '—';if(n!==0&&Math.abs(n)<.0001)return n.toExponential(2);return n.toFixed(6);}
  function first(...values){return values.find(v=>v!==null&&v!==undefined);}
  function metricRegion(frame,name){const m=frame.metrics||frame.regions||{};const regions=m.regions||m;return regions[name]||{};}
  function metric(frame,...names){const m=frame.metrics||{};return first(...names.map(name=>m[name]),...names.map(name=>frame[name]));}
  function timestamp(frame){return first(frame.timestamp,frame.timestamp_model);}
  function textElement(tag,className,text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
  function overviewStat(label,value,unit,detail){const el=textElement('div','stat');el.append(textElement('div','label',label));const v=textElement('div','value',value);if(unit)v.append(textElement('span','unit',unit));el.append(v,textElement('div','detail',detail));$('#overview').append(el);}
  function metadataItem(label,value){if(value===undefined||value===null||value==='')return;const el=textElement('span','meta-item',label+' ');el.append(textElement('strong','',String(value)));$('#metadata').append(el);}
  function stepValue(){const cp=data.checkpoint;if(cp&&typeof cp==='object')return first(cp.iteration,cp.steps,summary.training_steps);if(number(summary.training_steps)!==null)return summary.training_steps;if(typeof cp==='string'){const match=cp.match(/(?:chkpnt|iteration[_-]?)(\d+)/i);if(match)return Number(match[1]);}return null;}
  $('#page-title').textContent=data.title||'C00 · 30k 分支查看器';
  document.title=data.title||'C00 · 30k 分支查看器';
  const scene=typeof data.scene==='object'?first(data.scene.name,data.scene.id):data.scene;
  $('#page-subtitle').textContent=[scene?('场景 '+scene):null,'静态、动态与合成结果，同一帧同步查看'].filter(Boolean).join(' · ');
  metadataItem('控制臂',data.arm||'C00');
  const steps=number(stepValue());metadataItem('训练',steps!==null?steps.toLocaleString('en-US')+' 步':null);
  const wallMinutes=first(summary.wall_minutes,number(summary.wall_seconds)!==null?Number(summary.wall_seconds)/60:null,number(summary.wall_hours)!==null?Number(summary.wall_hours)*60:null);
  metadataItem('耗时',number(wallMinutes)!==null?fmt(wallMinutes,1)+' 分钟':null);
  const deviceGiB=first(summary.device_peak_GiB,summary.peak_vram_gib,number(summary.peak_vram_mib)!==null?Number(summary.peak_vram_mib)/1024:null);
  metadataItem('显存采样峰',number(deviceGiB)!==null?fmt(deviceGiB,2)+' GiB':null);
  metadataItem('reserved 峰值',number(summary.reserved_peak_GiB)!==null?fmt(summary.reserved_peak_GiB,2)+' GiB':null);
  if(number(summary.static_points)!==null&&number(summary.dynamic_points)!==null)metadataItem('高斯',Number(summary.static_points).toLocaleString('en-US')+' 静态 / '+Number(summary.dynamic_points).toLocaleString('en-US')+' 动态');
  const actualCount=frames.length;
  $('.scope b').textContent=actualCount+' 帧全观测开发重建';
  overviewStat('全图 · 合成 PSNR',fmt(summary.full_composite_psnr_db),'dB','所有观测帧的汇总');
  overviewStat('车体 · 合成 PSNR',fmt(summary.actor_composite_psnr_db),'dB','body mask 区域');
  overviewStat('背景 · 静态 PSNR',fmt(summary.background_static_psnr_db),'dB','非天空且车体外');
  overviewStat('车体外 · 动态 α 均值',fmt(summary.outside_dynamic_alpha_mean,4),'','原始未裁 alpha');
  $('#reconstruction-count').textContent=actualCount+' 帧';
  $('#frames-list-count').textContent=actualCount+' 帧 · 点击跳转';
  if(timeFrames.length){$('#tab-time').hidden=false;$('#about-time').hidden=false;$('#time-count').textContent=timeFrames.length+' 时刻';$('#time-list-count').textContent=timeFrames.length+' 个模型时间 · 点击跳转';}
  const anchor=first(timeSummary.anchor_pose_frame,timeSummary.anchor_frame,'0017010');
  const timeNotice=$('#time-notice');timeNotice.append(textElement('strong','', '相机固定在 '+anchor+'。'),' 仅改变 model time，观察静态分支；相对锚点时间 '+compact(first(timeSummary.anchor_timestamp_model,timeSummary.anchor_timestamp))+' 测量差异。本组没有对应 GT，不计算重建 PSNR。');
  const checkpoint=data.checkpoint;const cpText=checkpoint&&typeof checkpoint==='object'?first(checkpoint.path,checkpoint.name,checkpoint.iteration):checkpoint;
  const sourcePath=typeof data.source==='string'?data.source:(data.source||{}).model_path;
  $('#source-info').textContent=['检查点：'+(cpText??'—'),sourcePath?('模型路径：'+sourcePath):null,'原图尺寸：'+width+' × '+height].filter(Boolean).join(' · ');

  function createCard(key,container,tab){
    const config=branches[key],card=textElement('article','image-card');card.dataset.key=key;
    const heading=textElement('div','card-heading');heading.append(textElement('i','branch-dot'),textElement('h2','',config.label),textElement('span','card-tag',config.tag));
    const button=textElement('button','image-button');button.setAttribute('aria-label','查看'+config.label+'原始大图');
    const surface=textElement('div','image-surface'),img=document.createElement('img'),empty=textElement('span','empty','当前帧没有这张图');empty.hidden=true;img.alt=config.label;img.decoding='async';img.draggable=false;surface.append(img,empty);button.append(surface);
    const footer=textElement('div','card-footer');footer.append(textElement('span','',config.footer),textElement('span','zoom-hint','↗ 查看大图'));
    card.append(heading,button,footer);container.append(card);
    const record={key,card,surface,img,empty,button};cards[tab].push(record);
    img.addEventListener('error',()=>{empty.hidden=false;img.hidden=true;empty.textContent='图片未能载入 · 请保留 assets/ 目录';});
    img.addEventListener('load',()=>{if(img.getAttribute('src')){empty.hidden=true;img.hidden=false;}});
    button.addEventListener('click',()=>{if((current(tab).images||{})[key])openModal(key);});
  }
  ['gt','composite','static','dynamic'].forEach(key=>createCard(key,$('#reconstruction-grid'),'reconstruction'));
  ['alpha_s','alpha_d','body_mask'].forEach(key=>createCard(key,$('#diagnostic-grid'),'reconstruction'));
  ['static','alpha_s'].forEach(key=>createCard(key,$('#time-grid'),'time'));

  function bboxFor(frame){
    let raw=frame.bbox||frame.actor_bbox||frame.roi_bbox;if(!raw)return null;
    let box;if(Array.isArray(raw)&&raw.length>=4)box=raw.slice(0,4).map(number);else if(typeof raw==='object'){
      const x=first(raw.x0,raw.x,raw.left),y=first(raw.y0,raw.y,raw.top);
      box=[number(x),number(y),number(first(raw.x1,raw.right,number(x)!==null&&number(raw.width)!==null?Number(x)+Number(raw.width):null)),number(first(raw.y1,raw.bottom,number(y)!==null&&number(raw.height)!==null?Number(y)+Number(raw.height):null))];
    }else return null;
    if(box.some(v=>v===null)||box[2]<=box[0]||box[3]<=box[1])return null;
    let x0=Math.max(0,Math.min(width,box[0])),y0=Math.max(0,Math.min(height,box[1])),x1=Math.max(0,Math.min(width,box[2])),y1=Math.max(0,Math.min(height,box[3]));
    if(x1<=x0||y1<=y0)return null;
    // 所有面板使用同一裁切；只扩展到原图比例，不收缩已有上下文框。
    const ratio=width/height,cx=(x0+x1)/2,cy=(y0+y1)/2;let w=x1-x0,h=y1-y0;
    if(w/h<ratio)w=h*ratio;else h=w/ratio;
    if(w>width){w=width;h=w/ratio;}if(h>height){h=height;w=h*ratio;}
    x0=Math.max(0,Math.min(width-w,cx-w/2));y0=Math.max(0,Math.min(height-h,cy-h/2));
    return [x0,y0,w,h];
  }
  function applyCrop(img,surface,frame,tab,native=false){
    const box=state.roi&&tab==='reconstruction'?bboxFor(frame):null;
    surface.style.aspectRatio=width+' / '+height;
    if(box){const [x,y,w,h]=box;img.style.width=(width/w*100)+'%';img.style.left=(-x/w*100)+'%';img.style.top=(-y/h*100)+'%';if(native&&!state.fit)surface.style.width=Math.round(w)+'px';}
    else{img.style.width='100%';img.style.left='0';img.style.top='0';if(native&&!state.fit)surface.style.width=width+'px';}
    if(native&&state.fit)surface.style.width='100%';
  }
  function updateImage(img,empty,surface,key,frame,tab,native=false){
    const path=(frame.images||{})[key];applyCrop(img,surface,frame,tab,native);
    img.alt=branches[key].label+' · '+frameId(frame)+(tab==='time'?' · 固定相机 model time '+compact(timestamp(frame)):'');
    if(!path){img.removeAttribute('src');img.hidden=true;empty.textContent='当前帧没有这张图';empty.hidden=false;return;}
    if(img.getAttribute('src')!==String(path)){empty.hidden=true;img.hidden=false;img.src=String(path);}
  }
  function regionCard(title,region,mainKey,mainLabel,secondaryKey,secondaryLabel){
    const el=textElement('div','metric-card');el.append(textElement('div','region',title));
    const value=textElement('div','metric-value',fmt(region[mainKey]));value.append(textElement('span','','dB'));el.append(value,textElement('div','metric-description',mainLabel));
    const pairs=textElement('div','metric-pairs');
    [[secondaryLabel,fmt(region[secondaryKey])+' dB'],['α 动态',fmt(region.alpha_d_mean,4)],['α 静态',fmt(region.alpha_s_mean,4)]].forEach(([label,value])=>{const row=textElement('div');row.append(textElement('span','',label),textElement('strong','',value));pairs.append(row);});el.append(pairs);return el;
  }
  function renderMetrics(frame){
    $('#frame-metrics').replaceChildren(
      regionCard('全图',metricRegion(frame,'full'),'composite_psnr_db','合成 PSNR','static_psnr_db','静态 PSNR'),
      regionCard('车体 · body mask',metricRegion(frame,'actor'),'composite_psnr_db','合成 PSNR','static_psnr_db','静态 PSNR'),
      regionCard('背景 · 非天空且车体外',metricRegion(frame,'non_sky_outside_actor'),'static_psnr_db','静态 PSNR','composite_psnr_db','合成 PSNR')
    );
  }
  function differenceCard(label,value,description){const el=textElement('div','metric-card');el.append(textElement('div','region',label),textElement('div','metric-value',compact(value)),textElement('div','metric-description',description));return el;}
  function renderTimeMetrics(frame){$('#time-metrics').replaceChildren(
    differenceCard('静态 RGB · 平均绝对差',metric(frame,'static_rgb_MAE_from_anchor','rgb_mae'),'相对锚点时间的 MAE'),
    differenceCard('静态 RGB · 最大绝对差',metric(frame,'static_rgb_max_abs_from_anchor','rgb_max_abs'),'相对锚点时间的 max |Δ|'),
    differenceCard('静态 α · 平均绝对差',metric(frame,'static_alpha_MAE_from_anchor','alpha_mae'),'相对锚点时间的 MAE')
  );}
  function createFrameList(tab,container){list(tab).forEach((frame,index)=>{const button=textElement('button','frame-chip');button.dataset.index=index;button.append(textElement('span','',String(index+1).padStart(2,'0')+' · '+frameId(frame)),textElement('small','','t '+compact(timestamp(frame))));button.setAttribute('aria-label','跳转到第'+(index+1)+'帧，'+frameId(frame));button.addEventListener('click',()=>setIndex(index,tab));container.append(button);});}
  createFrameList('reconstruction',$('#frames-list'));createFrameList('time',$('#time-list'));
  function updateControls(tab){
    const control=$('[data-controls="'+tab+'"]'),items=list(tab),index=state.indices[tab],frame=items[index]||{};
    control.querySelector('.frame-label').textContent=tab==='time'?('时间 '+(index+1)+' / '+items.length+' · '+frameId(frame)):('帧 '+(index+1)+' / '+items.length+' · '+frameId(frame));
    const rawIndex=frame.raw_index!==undefined?' · raw '+frame.raw_index:'';
    control.querySelector('.frame-meta').textContent='model time '+compact(timestamp(frame))+rawIndex;
    const slider=control.querySelector('.slider');slider.max=Math.max(0,items.length-1);slider.value=index;slider.disabled=items.length<2;slider.setAttribute('aria-valuetext',frameId(frame)+'，model time '+compact(timestamp(frame)));
    control.querySelector('.frame-total').textContent=items.length;
    control.querySelector('.prev').disabled=index===0;control.querySelector('.next').disabled=index>=items.length-1;
    control.querySelector('.play').disabled=items.length<2;
    const roi=control.querySelector('[data-view="roi"]');if(roi)roi.disabled=!bboxFor(frame);
    $$(tab==='time'?'#time-list .frame-chip':'#frames-list .frame-chip').forEach(button=>button.setAttribute('aria-current',Number(button.dataset.index)===index?'true':'false'));
  }
  function render(tab=state.tab){
    const frame=current(tab);cards[tab].forEach(record=>updateImage(record.img,record.empty,record.surface,record.key,frame,tab));
    updateControls(tab);if(tab==='time')renderTimeMetrics(frame);else{renderMetrics(frame);const note=$('[data-controls="reconstruction"] .roi-note');note.textContent=state.roi?(bboxFor(frame)?'车体附近同一 ROI · 所有分支与诊断图同步':'当前帧无车体 ROI · 显示原图'):'同帧四卡同步 · 点击图片查看原始大图';}
    if($('#lightbox').open)renderModal();
  }
  function setIndex(index,tab=state.tab){state.indices[tab]=Math.max(0,Math.min(list(tab).length-1,index));render(tab);}
  function step(delta){setIndex(state.indices[state.tab]+delta);}
  function playUI(){$$('.play').forEach(button=>{const active=state.playing&&button.closest('[data-controls]').dataset.controls===state.tab;button.classList.toggle('active',active);button.setAttribute('aria-pressed',active?'true':'false');button.querySelector('span').textContent=active?'暂停':'播放';button.querySelector('path').setAttribute('d',active?'M6 5h4v14H6zm8 0h4v14h-4z':'M8 5v14l11-7z');});}
  function stop(){if(state.timer!==null)clearInterval(state.timer);state.timer=null;state.playing=false;playUI();}
  function start(){stop();if(list(state.tab).length<2)return;state.playing=true;playUI();state.timer=setInterval(()=>{const next=(state.indices[state.tab]+1)%list(state.tab).length;setIndex(next);},1000/state.fps);}
  function togglePlay(){state.playing?stop():start();}
  function setTab(tab){if(tab==='time'&&!timeFrames.length)return;stop();if($('#lightbox').open)$('#lightbox').close();state.tab=tab;$('#reconstruction').hidden=tab!=='reconstruction';$('#time-panel').hidden=tab!=='time';$$('[data-tab]').forEach(button=>button.setAttribute('aria-selected',button.dataset.tab===tab?'true':'false'));render(tab);}
  $$('[data-tab]').forEach(button=>button.addEventListener('click',()=>setTab(button.dataset.tab)));
  $$('[data-controls]').forEach(control=>{const tab=control.dataset.controls;control.querySelector('.prev').addEventListener('click',()=>setIndex(state.indices[tab]-1,tab));control.querySelector('.next').addEventListener('click',()=>setIndex(state.indices[tab]+1,tab));control.querySelector('.play').addEventListener('click',togglePlay);control.querySelector('.slider').addEventListener('input',event=>setIndex(Number(event.target.value),tab));control.querySelector('.fps').addEventListener('change',event=>{state.fps=Number(event.target.value);$$('.fps').forEach(select=>select.value=state.fps);if(state.playing)start();});});
  $$('[data-view]').forEach(button=>button.addEventListener('click',()=>{state.roi=button.dataset.view==='roi';$$('[data-view]').forEach(option=>option.setAttribute('aria-pressed',(option.dataset.view==='roi')===state.roi?'true':'false'));render('reconstruction');}));

  function openModal(key){state.modalKey=key;const available=state.tab==='time'?['static','alpha_s']:Object.keys(branches);$('#modal-branches').replaceChildren();available.forEach(branch=>{const button=textElement('button','',branches[branch].short);button.dataset.branch=branch;button.addEventListener('click',()=>{state.modalKey=branch;renderModal();});$('#modal-branches').append(button);});$('#lightbox').showModal();renderModal();}
  function renderModal(){
    const frame=current(),key=state.modalKey,index=state.indices[state.tab],total=list(state.tab).length;
    $('#lightbox-title').textContent=branches[key].label;
    $('#lightbox-meta').textContent=frameId(frame)+' · '+(index+1)+' / '+total+' · model time '+compact(timestamp(frame));
    $('#modal-branches').querySelectorAll('button').forEach(button=>button.setAttribute('aria-pressed',button.dataset.branch===key?'true':'false'));
    $('#modal-image-wrap').classList.toggle('fit',state.fit);
    updateImage($('#modal-image'),$('#modal-image-surface .empty'),$('#modal-image-surface'),key,frame,state.tab,true);
    $('#modal-prev').disabled=index===0;$('#modal-next').disabled=index>=total-1;
    const roi=state.roi&&state.tab==='reconstruction'&&bboxFor(frame);$('#modal-caption').textContent=(roi?'同步车体 ROI':'完整原图')+' · '+width+' × '+height+' 源图 · '+branches[key].footer;
  }
  $('#modal-image').addEventListener('error',()=>{$('#modal-image-surface .empty').textContent='图片未能载入 · 请保留 assets/ 目录';$('#modal-image-surface .empty').hidden=false;$('#modal-image').hidden=true;});
  $('#modal-image').addEventListener('load',()=>{$('#modal-image-surface .empty').hidden=true;$('#modal-image').hidden=false;});
  $('#close-lightbox').addEventListener('click',()=>$('#lightbox').close());
  $('#lightbox').addEventListener('click',event=>{if(event.target!==$('#lightbox'))return;const box=$('#lightbox').getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)$('#lightbox').close();});
  $('#modal-prev').addEventListener('click',()=>step(-1));$('#modal-next').addEventListener('click',()=>step(1));
  function zoom(fit){state.fit=fit;$('#zoom-native').setAttribute('aria-pressed',fit?'false':'true');$('#zoom-fit').setAttribute('aria-pressed',fit?'true':'false');renderModal();}
  $('#zoom-native').addEventListener('click',()=>zoom(false));$('#zoom-fit').addEventListener('click',()=>zoom(true));
  document.addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey)return;const tag=(event.target.tagName||'').toLowerCase();if(['input','select','textarea'].includes(tag))return;if(event.key==='ArrowLeft'){event.preventDefault();step(-1);}else if(event.key==='ArrowRight'){event.preventDefault();step(1);}else if(event.code==='Space'&&tag!=='button'){event.preventDefault();togglePlay();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  render('reconstruction');if(timeFrames.length)render('time');
})();
</script>
</body>
</html>
'''


def build(manifest_path: Path, output_dir: Path) -> Path:
    """只生成 index.html，不改动 manifest 或图片资产。"""
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        raise ValueError("manifest 必须包含 frames 数组")
    if not data["frames"]:
        raise ValueError("manifest.frames 不能为空")
    if any(not isinstance(frame, dict) for frame in data["frames"]):
        raise ValueError("manifest.frames 每帧必须是对象")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # 防止清单字符串提前结束 JSON script，file:// 打开时无需 fetch。
    payload = (payload.replace("&", "\\u0026").replace("<", "\\u003c")
               .replace(">", "\\u003e").replace("\u2028", "\\u2028")
               .replace("\u2029", "\\u2029"))
    title = html.escape(str(data.get("title") or "C00 · 30k 分支查看器"), quote=True)
    page = PAGE.replace("__TITLE__", title).replace("__MANIFEST__", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="本地 manifest.json，默认 OUTPUT/manifest.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/drivingeditor_C00_30k"),
                        help="写入 index.html 的目录，默认 outputs/drivingeditor_C00_30k")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    manifest_path = (args.manifest or output_dir / "manifest.json").resolve()
    try:
        output_path = build(manifest_path, output_dir)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, "生成失败：%s\n" % exc)
    print("已生成离线查看器：%s" % output_path)


if __name__ == "__main__":
    main()
