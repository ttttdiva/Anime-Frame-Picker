// Real accepted-catalogue UI interaction. No inference/extraction actions.
// NODE_PATH=<existing playwright-core parent> node ... <url> <evidence> <chrome>
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import childProcess from 'node:child_process';
// Apply the workspace's no-visible-helper-window rule to Playwright's children.
const originalSpawn = childProcess.spawn;
childProcess.spawn = (command, args, options) => originalSpawn(command, args, { ...options, windowsHide: true });
const require = createRequire(import.meta.url);
const { chromium } = require('playwright-core');
const [baseUrl, evidenceDir, executablePath] = process.argv.slice(2);
fs.mkdirSync(evidenceDir, { recursive: true });
const log = [], network = [], pending = [], errors = [];
const step = (name, data = {}) => { const entry = { step: name, ...data }; log.push(entry); console.log(JSON.stringify(entry)); };
const expect = (value, message) => { if (!value) throw new Error(message); };
const browser = await chromium.launch({ headless: true, executablePath, args: ['--disable-gpu'] });
let failure = null;
try {
  const context = await browser.newContext({ viewport: { width: 1500, height: 1000 } });
  context.on('page', (p) => p.on('pageerror', (e) => errors.push(String(e))));
  context.on('request', (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith('/api/')) {
      network.push({ method: request.method(), path: url.pathname + url.search });
      if (request.method() !== 'GET') errors.push(`unexpected mutation: ${request.method()} ${url.pathname}`);
      if (url.pathname.includes('character-suggestions') || url.pathname.startsWith('/api/jobs')) errors.push(`unexpected inference route: ${url.pathname}`);
    }
  });
  context.on('response', (response) => {
    if (new URL(response.url()).pathname.startsWith('/api/accepted-catalogue/frames/') && response.url().endsWith('/image')) {
      pending.push((async () => {
        const bytes = await response.body();
        step('native_network_response', { url: response.url(), status: response.status(), bytes: bytes.length,
          sha256: crypto.createHash('sha256').update(bytes).digest('hex'), contentType: response.headers()['content-type'],
          nosniff: response.headers()['x-content-type-options'] });
      })().catch((e) => step('thumbnail_response_unavailable', { reason: String(e) })));
    }
  });
  const page = await context.newPage();
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Accepted frames / 採用フレーム', exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll('.accepted-card').length > 0, null, { timeout: 120000 });
  const count = await page.textContent('#acceptedCount');
  expect(count.includes('23,792'), `exact accepted count: ${count}`);
  expect(await page.locator('.accepted-card').count() === 40, 'bounded initial page');
  step('accepted_entry', { count, status: await page.textContent('#acceptedStatus'), label: 'isolated localhost real app, pinned preview release' });
  await page.screenshot({ path: path.join(evidenceDir, '01-accepted.png'), fullPage: true });
  const initialIds = await page.locator('.accepted-card').evaluateAll((cards) => cards.map((c) => c.dataset.frameId));
  const move = async (button) => {
    const before = await page.locator('.accepted-card').first().getAttribute('data-frame-id');
    await page.locator(button).click();
    await page.waitForFunction((id) => document.querySelector('.accepted-card')?.dataset.frameId !== id && !document.querySelector('#acceptedSearchButton').disabled, before);
  };
  await move('#acceptedNext');
  const nextIds = await page.locator('.accepted-card').evaluateAll((cards) => cards.map((c) => c.dataset.frameId));
  expect(!nextIds.some((id) => initialIds.includes(id)), 'forward page has no duplicate IDs');
  await move('#acceptedPrevious');
  expect(JSON.stringify(initialIds) === JSON.stringify(await page.locator('.accepted-card').evaluateAll((cards) => cards.map((c) => c.dataset.frameId))), 'back page is stable');
  step('forward_back', { initialIds, nextIds, count: await page.textContent('#acceptedCount') });
  const openNative = async (card, label) => {
    const identity = await card.getAttribute('data-frame-id');
    const [popup] = await Promise.all([page.waitForEvent('popup'), card.locator('a').click()]);
    await popup.waitForLoadState('load');
    await popup.waitForFunction(() => { const i = document.querySelector('img'); return i && i.complete && i.naturalWidth > 0; });
    step('open_native', { label, identity, url: popup.url(), size: await popup.locator('img').evaluate((i) => [i.naturalWidth, i.naturalHeight]) });
    await popup.screenshot({ path: path.join(evidenceDir, `${label}-native.png`) });
    await popup.close();
  };
  await openNative(page.locator('.accepted-card[data-native-origin="existing_native"]').first(), '02-old');
  const suggestion = await page.locator('#acceptedSuggestions button').first().textContent();
  expect(suggestion, 'real cached character suggestions');
  const originalTag = suggestion.replace(/ \(\d+\)$/, '');
  const unicodeQuery = originalTag.replace(/[a-z]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 0xFEE0));
  await page.fill('#acceptedQuery', unicodeQuery);
  await page.selectOption('#acceptedCategory', 'character');
  const submit = async () => {
    const response = page.waitForResponse((r) => new URL(r.url()).pathname === '/api/accepted-catalogue/frames');
    await page.locator('#acceptedSearchButton').click();
    const r = await response; expect(r.status() === 200, 'search GET succeeded');
    await page.waitForFunction(() => !document.querySelector('#acceptedSearchButton').disabled);
    return r.json();
  };
  const found = await submit();
  expect(found.total > 0, 'Unicode-normalized real character tag search');
  step('unicode_character_search', { originalTag, unicodeQuery, total: found.total, visibleCount: await page.textContent('#acceptedCount') });
  await page.screenshot({ path: path.join(evidenceDir, '03-unicode-search.png'), fullPage: true });
  await page.fill('#acceptedQuery', '__definitely_no_tag_7e05e28d__');
  const absent = await submit(); expect(absent.total === 0 && await page.locator('.accepted-card').count() === 0, 'no-result search');
  step('no_result_search', { count: await page.textContent('#acceptedCount') });
  await page.fill('#acceptedQuery', '');
  await page.selectOption('#acceptedCategory', '');
  await page.selectOption('#acceptedRow', '02');
  await submit();
  let turns = 0;
  while (await page.locator('.accepted-card[data-native-origin="accepted_overlay_copy"]').count() === 0) {
    expect(await page.locator('#acceptedNext').isEnabled() && turns++ < 100, 'find new overlay through bounded UI pages');
    await move('#acceptedNext');
  }
  await page.screenshot({ path: path.join(evidenceDir, '04-overlay-page.png'), fullPage: true });
  await openNative(page.locator('.accepted-card[data-native-origin="accepted_overlay_copy"]').first(), '05-new');
  step('overlay_page', { turns, count: await page.textContent('#acceptedCount') });
  await Promise.all(pending);
  for (const opened of log.filter((entry) => entry.step === 'open_native')) {
    expect(log.some((entry) => entry.step === 'native_network_response' && entry.url === opened.url && entry.status === 200 && entry.bytes > 0 && entry.nosniff === 'nosniff'), 'opened native has captured PNG network bytes');
  }
  expect(errors.length === 0, `browser errors: ${errors.join('; ')}`);
} catch (error) { failure = String(error.stack || error); }
finally {
  await browser.close();
  fs.writeFileSync(path.join(evidenceDir, 'ui-result.json'), JSON.stringify({ pass: !failure, failure, errors, log, network }, null, 2));
}
if (failure) { console.error(failure); process.exitCode = 1; }
