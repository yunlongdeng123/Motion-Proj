/* Share the V4 initializer inside the existing diagnostic index, with isolated IDs. */
(() => {
  'use strict';
  const SIZE = 'v4-initialization-content-size';
  const REQUEST = 'v4-initialization-request-size';
  if (window.parent !== window && new URLSearchParams(window.location.search).get('embedded') === '1') {
    document.documentElement.classList.add('embedded-v4');
    let pending = false, lastHeight = 0;
    const sendSize = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const height = Math.ceil(document.body.getBoundingClientRect().height) + 1;
        if (height > 0 && Math.abs(height - lastHeight) > 2) {
          lastHeight = height;
          window.parent.postMessage({type: SIZE, height}, '*');
        }
      });
    };
    window.addEventListener('message', event => {
      if (event.source === window.parent && event.data?.type === REQUEST) {
        lastHeight = 0;
        sendSize();
      }
    });
    window.addEventListener('resize', sendSize);
    new ResizeObserver(sendSize).observe(document.body);
    sendSize();
    return;
  }
  const workspace = document.getElementById('workspace');
  const main = document.querySelector('main');
  if (!workspace || !main || document.getElementById('v4-initialization')) return;
  const section = document.createElement('section');
  section.id = 'v4-initialization';
  section.className = 'panel';
  section.style.scrollMarginTop = '65px';
  section.innerHTML = '<div class="panel-title"><span class="number">V4</span><h2>初始化与覆盖验收 · 共同0.2米世界网格</h2></div>' +
    '<p class="note">V4原模型初始地面、新增固定道路与V3道路：选择a5／a37／a6，比较几何、尺度、来源、初始覆盖与深度。以下直接在本页操作。</p>' +
    '<iframe id="v4InitializationFrame" title="V4初始化与覆盖验收" src="v4_initialization.html?embedded=1" style="display:block;width:100%;height:1200px;border:0;background:#0a101a" loading="eager"></iframe>' +
    '<p class="note"><a href="v4_initialization.html" target="_blank" rel="noopener">单独展开V4验收页 ↗</a> · <a href="#workspace">继续查看已有实验的渲染与诊断 ↓</a></p>';
  main.insertBefore(section, main.firstChild);
  const frame = document.getElementById('v4InitializationFrame');
  window.addEventListener('message', event => {
    if (event.source !== frame.contentWindow || event.data?.type !== SIZE) return;
    const height = Number(event.data.height);
    if (Number.isFinite(height) && height >= 400 && height <= 30000) frame.style.height = Math.ceil(height) + 'px';
  });
  frame.addEventListener('load', () => frame.contentWindow.postMessage({type: REQUEST}, '*'));
  for (const link of document.querySelectorAll('nav a[href="v4_initialization.html"]')) link.href = '#v4-initialization';
  document.title = 'Gaussian Evidence Lab · V4初始化与覆盖验收';
  const version = document.querySelector('header h1 em');
  if (version) version.textContent = 'V4 初始化 / V3';
})();
