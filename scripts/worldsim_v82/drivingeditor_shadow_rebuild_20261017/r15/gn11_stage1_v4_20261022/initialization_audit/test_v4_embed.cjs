// Actual embed module in a minimal DOM: no browser/HTTP, and no training objects.
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname, 'v4_embed.js'), 'utf8');
const events = {}, sent = [], elements = new Map();
const frame = {style: {height: '1200px'}, contentWindow: {postMessage: (...v) => sent.push(v)}, addEventListener: (name, fn) => events['frame:' + name] = fn};
const main = {children: [], firstChild: {}, insertBefore(section) { this.children.unshift(section); elements.set(section.id, section); elements.set('v4InitializationFrame', frame); }};
const links = [{href: 'v4_initialization.html'}], version = {}, workspace = {};
elements.set('workspace', workspace);
const window = {location: {search: ''}, addEventListener: (name, fn) => events[name] = fn}; window.parent = window;
const document = {getElementById: id => elements.get(id), querySelector: s => s === 'main' ? main : version,
  querySelectorAll: () => links, createElement: () => ({style: {}})};
const host = vm.createContext({window, document, URLSearchParams, Number, Math});
vm.runInContext(source, host);
assert.equal(main.children.length, 1); assert.match(main.children[0].innerHTML, /v4_initialization\.html\?embedded=1/);
assert.equal(elements.get('workspace'), workspace); assert.equal(links[0].href, '#v4-initialization');
events.message({source: {}, data: {type: 'v4-initialization-content-size', height: 9000}}); assert.equal(frame.style.height, '1200px');
events.message({source: frame.contentWindow, data: {type: 'v4-initialization-content-size', height: 4321}}); assert.equal(frame.style.height, '4321px');
for (const height of [NaN, 0, -4, 90000]) events.message({source: frame.contentWindow, data: {type: 'v4-initialization-content-size', height}});
assert.equal(frame.style.height, '4321px'); events['frame:load'](); assert.equal(sent[0][0].type, 'v4-initialization-request-size');
vm.runInContext(source, host); assert.equal(main.children.length, 1);
const childEvents = {}, childSent = [], classes = []; let childHeight = 2450, resize;
const parent = {postMessage: message => childSent.push(message)};
const childWindow = {parent, location: {search: '?embedded=1'}, addEventListener: (name, fn) => childEvents[name] = fn};
const childDocument = {documentElement: {classList: {add: name => classes.push(name)}}, body: {getBoundingClientRect: () => ({height: childHeight})}};
vm.runInNewContext(source, {window: childWindow, document: childDocument, URLSearchParams, Math,
  requestAnimationFrame: fn => fn(), ResizeObserver: class { constructor(fn) { resize = fn; } observe() {} }});
assert(classes.includes('embedded-v4')); assert.equal(childSent[0].height, 2451); resize(); assert.equal(childSent.length, 1);
childHeight = 2800; resize(); assert.equal(childSent[1].height, 2801);
childEvents.message({source: {}, data: {type: 'v4-initialization-request-size'}}); assert.equal(childSent.length, 2);
childEvents.message({source: parent, data: {type: 'v4-initialization-request-size'}}); assert.equal(childSent.length, 3);
console.log(JSON.stringify({status: 'PASSED',checks: ['mounts audit on original index without replacing legacy workspace','accepts height only from the actual child and bounds it','idempotent host mount','child hides standalone header and resizes after content changes'],browser_ui_verified: false}));
