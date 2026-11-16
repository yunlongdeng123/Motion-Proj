"""Local browser acceptance for the published scene065 diagnostic extension.

    python qa_diagnostics_ui.py --run

Runs only local Chrome/CPU checks. Does not publish pages or modify assets.
Screenshots and a JSON receipt are retained in a fresh output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

import add_diagnostics_ui as ui

IDS = ui.BASE_COLUMNS + ["closedloopRoadGap", "closedloopStaticAlphaT"]
VIEWS = ["old_delete", "new_delete", "road_only", "static_contribution", "road_gap", "static_alphaT"]


def check(receipt, name, condition, detail=None):
    receipt["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def hashes(page):
    # File access is explicitly enabled for these already-local preview images.
    return page.evaluate("""ids => Object.fromEntries(ids.map(id => {
      const c=document.getElementById(id), data=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      let h=2166136261;for(let i=0;i<data.length;i++){h^=data[i];h=Math.imul(h,16777619);}
      return [id,h>>>0];
    }))""", IDS)


def wait_frame(page, raw):
    page.wait_for_function("""raw => {
      const status=document.getElementById('status'), a=document.getElementById('closedloopRoadGap'),b=document.getElementById('closedloopStaticAlphaT');
      return Number(document.getElementById('frame').value)===raw && status.textContent.includes('第 '+raw+' 帧') && !status.textContent.includes('正在加载') &&
        a.dataset.raw===String(raw)&&b.dataset.raw===String(raw)&&a.dataset.loaded==='true'&&b.dataset.loaded==='true';
    }""", arg=raw)


def set_frame(page, raw):
    page.locator('#rawNumber').fill(str(raw))
    page.locator('#rawNumber').press('Tab')
    wait_frame(page, raw)


def layout(page, receipt, name):
    data = page.evaluate("""() => ({width:innerWidth,scroll:Math.max(document.body.scrollWidth,document.documentElement.scrollWidth),
      background:getComputedStyle(document.body).backgroundColor,
      columns:getComputedStyle(document.querySelector('.native-diagnostics')||document.querySelector('.frames')).gridTemplateColumns,
      panels:Array.from(document.querySelectorAll('.native-diagnostics article')).map(c=>({x:c.getBoundingClientRect().x,y:c.getBoundingClientRect().y}))})""")
    check(receipt, name + '_no_horizontal_overflow', data['scroll'] <= data['width'] + 1, data)
    receipt['layouts'].append({'name': name, **data})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--platform', type=Path, default=ui.PLATFORM / 'index.html')
    parser.add_argument('--report', type=Path, default=ui.REPORT)
    parser.add_argument('--browser', type=Path, default=Path('C:/Program Files/Google/Chrome/Application/chrome.exe'))
    parser.add_argument('--out', type=Path, default=ui.WORK / ('qa_diagnostics_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'mode': 'plan_only', 'command': 'python work/scene065_closedloop_20261116/qa_diagnostics_ui.py --run'}, indent=2))
        return 0
    args.platform, args.report, args.out = args.platform.resolve(), args.report.resolve(), args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose a fresh output directory to preserve earlier browser evidence')
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='scene065_diagnostics_browser_qa.v1', complete=False, checks=[], layouts=[],
                   page_errors=[], console_errors=[], failed_requests=[], GPU_compute_used=False,
                   quality_automatically_accepted=False, started_utc=datetime.now(timezone.utc).isoformat())
    page = browser = None
    try:
        manifest = json.loads((args.platform.parent / 'diagnostic_delivery/manifest.json').read_text(encoding='utf-8'))
        by_raw = {row['raw']: row for row in manifest['frames']}
        platform_text = args.platform.read_text(encoding='utf-8')
        soup = BeautifulSoup(platform_text, 'html.parser')
        check(receipt, 'original_six_panels_preserved', [c['id'] for c in soup.select('.native-comparison canvas')] == ui.BASE_COLUMNS)
        check(receipt, 'two_panels_below_original_six', [c['id'] for c in soup.select('.native-diagnostics canvas')] == IDS[-2:] and
              platform_text.index('id="contribution-diagnostics"') > platform_text.index('id="closedloopNative30"'))
        for path, selector in [(args.platform, '#closedloop-outcome'), (args.report, '.result')]:
            backup = path.with_name(path.name + '.before_contribution_diagnostics_20261116.bak')
            before = BeautifulSoup(backup.read_text(encoding='utf-8'), 'html.parser').select_one(selector).get_text()
            after = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser').select_one(selector).get_text()
            check(receipt, 'quality_verdict_unchanged_' + selector, before == after, after)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=str(args.browser), headless=True,
                args=['--disable-gpu', '--allow-file-access-from-files'])
            receipt['browser_version'] = browser.version
            context = browser.new_context(viewport={'width': 1440, 'height': 1100}, device_scale_factor=1)
            page = context.new_page()
            page.set_default_timeout(30000)
            page.on('pageerror', lambda error: receipt['page_errors'].append(str(error)))
            page.on('console', lambda message: receipt['console_errors'].append(message.text) if message.type == 'error' else None)
            page.on('requestfailed', lambda request: receipt['failed_requests'].append({'url': request.url, 'failure': request.failure})
                    if 'ERR_ABORTED' not in (request.failure or '') else None)
            page.goto(args.platform.as_uri() + '#raw=179', wait_until='load')
            wait_frame(page, 179)
            for raw in [0, 13, 14, 60, 86, 110, 179, 198]:
                set_frame(page, raw)
                text = page.locator('#diagnosticsGapStats').inner_text()
                stats = by_raw[raw]['stats']
                check(receipt, f'frame_{raw}_caption_matches_domain', (f"{stats['domain_pixels']:,}" in text)
                      if stats['domain_pixels'] else ('无道路恢复域' in text and '0 / 0' not in text), text)
                if raw == 179:
                    check(receipt, 'frame179_reference_counts', [stats[k] for k in ['domain_pixels','gap_pixels','thin_pixels','occluded_pixels']] == [9242,892,418,285], stats)
            set_frame(page, 179)
            for view in ['full', 'roi', 'pixel']:
                page.locator('#view').select_option(view)
                page.locator('#diagnosticsGapHighlight').uncheck()
                values = hashes(page)
                check(receipt, view + '_unhighlighted_gap_equals_new_delete', values['closedloopRoadGap'] == values['closedloopNewDelete'], values)
                page.locator('#diagnosticsGapHighlight').check()
                values = hashes(page)
                check(receipt, view + '_gap_highlights_visible', values['closedloopRoadGap'] != values['closedloopNewDelete'], values)
            page.locator('#view').select_option('full')
            before = hashes(page)
            page.locator('#closedloopStaticAlphaT').click(position={'x': 100, 'y': 90})
            after = hashes(page)
            check(receipt, 'diagnostic_click_updates_all_eight_crosshairs', all(before[key] != after[key] for key in IDS),
                  page.locator('#pixel').inner_text())
            for canvas in IDS[-2:]:
                check(receipt, 'shared_pixel_' + canvas,
                      page.locator('#' + canvas).get_attribute('data-pixel') in page.locator('#pixel').inner_text().replace(', ', ','))
            before = hashes(page)
            page.locator('#diagnosticsVisible').uncheck()
            check(receipt, 'diagnostics_hidden_original_six_visible', page.locator('#diagnostic-panels').is_hidden() and
                  all(page.locator('#' + key).is_visible() for key in ui.BASE_COLUMNS))
            page.locator('#diagnosticsVisible').check()
            check(receipt, 'visibility_toggle_preserves_original_pixels', before == hashes(page))
            slider = page.locator('#frame')
            slider.focus(); slider.press('Home'); wait_frame(page, 0)
            slider.press('End'); wait_frame(page, 198)
            check(receipt, 'slider_endpoints_synchronize_diagnostics', page.locator('#contribution-diagnostics').get_attribute('data-raw') == '198')
            set_frame(page, 86)
            page.locator('#fps').select_option('2')
            page.locator('#play').click()
            page.wait_for_function("Number(document.getElementById('frame').value)>=88")
            page.locator('#play').click()
            paused = int(page.locator('#frame').input_value())
            wait_frame(page, paused)
            page.wait_for_timeout(650)
            check(receipt, 'play_pause_synchronizes_eight_panels', int(page.locator('#frame').input_value()) == paused,
                  {'paused': paused, 'diagnostic_raw': page.locator('#contribution-diagnostics').get_attribute('data-raw')})
            set_frame(page, 179)
            layout(page, receipt, 'platform_desktop')
            page.locator('#contribution-diagnostics').screenshot(path=str(args.out / 'platform_179_diagnostics_desktop.png'))
            page.screenshot(path=str(args.out / 'platform_desktop.png'), full_page=True)
            page.set_viewport_size({'width': 393, 'height': 852})
            layout(page, receipt, 'platform_mobile')
            positions = page.locator('.native-diagnostics article').evaluate_all('(cards)=>cards.map(c=>c.getBoundingClientRect().x)')
            check(receipt, 'mobile_diagnostics_stack', abs(positions[0] - positions[1]) < 1, positions)
            page.locator('#contribution-diagnostics').screenshot(path=str(args.out / 'platform_179_diagnostics_mobile.png'))
            page.goto(args.report.as_uri(), wait_until='load')
            for width in [1440, 393]:
                page.set_viewport_size({'width': width, 'height': 1100 if width == 1440 else 852})
                for view in VIEWS:
                    page.locator(f'button[data-view="{view}"]').click()
                    page.locator('.frame img').evaluate_all('(images)=>images.forEach(im=>im.loading="eager")')
                    page.wait_for_function("""key => Array.from(document.querySelectorAll('.frame')).every(card=>{
                      const image=card.querySelector('img'), expected=JSON.parse(card.dataset.previews)[key];
                      return image.complete && image.naturalWidth>0 && image.src===new URL(expected,document.baseURI).href;
                    })""", arg=view)
                    check(receipt, f'report_{width}_all_seven_{view}', page.locator('.frame img').count() == 7)
                layout(page, receipt, 'report_' + str(width))
                page.locator('.diagnostic-report-legend').screenshot(path=str(args.out / f'report_legend_{width}.png'))
                page.screenshot(path=str(args.out / f'report_static_alphaT_{width}.png'), full_page=True)
            check(receipt, 'no_page_errors', not receipt['page_errors'], receipt['page_errors'])
            check(receipt, 'no_console_errors', not receipt['console_errors'], receipt['console_errors'])
            check(receipt, 'no_failed_requests', not receipt['failed_requests'], receipt['failed_requests'])
            browser.close(); browser = None
            receipt['complete'] = True
    except Exception:
        receipt['failure'] = traceback.format_exc()
        if page and not page.is_closed():
            try: page.screenshot(path=str(args.out / 'failure.png'), full_page=True)
            except Exception: pass
        if browser:
            try: browser.close()
            except Exception: pass
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    path = args.out / 'qa_diagnostics.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'complete': receipt['complete'], 'checks': len(receipt['checks']), 'receipt': str(path), 'failure': receipt.get('failure')}, ensure_ascii=False, indent=2))
    return 0 if receipt['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
