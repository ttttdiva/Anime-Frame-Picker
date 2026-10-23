// Headless real-UI check: candidate selection -> extraction -> result display.
// Usage: node tests/ui/char_select_ui.mjs <baseUrl> <evidenceDir> <chromiumExecutable>
// Requires playwright-core on NODE_PATH; never opens a visible window.
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright-core');
const [baseUrl, evidenceDir, executablePath] = process.argv.slice(2);
fs.mkdirSync(evidenceDir, { recursive: true });
const log = [];
const step = (name, data = {}) => { const entry = { at: new Date().toISOString(), step: name, ...data }; log.push(entry); console.log(JSON.stringify(entry)); };
const shot = async (page, name) => { const file = path.join(evidenceDir, name); await page.screenshot({ path: file, fullPage: true }); step('screenshot', { file }); };
const expect = (condition, message) => { if (!condition) throw new Error(`assertion failed: ${message}`); };

const browser = await chromium.launch({ headless: true, executablePath });
let failed = null;
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  page.on('pageerror', (error) => step('pageerror', { message: String(error) }));
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  step('open', { url: baseUrl });

  // Default operation: the only local video is selected and candidates appear
  // without pressing any "detect" button.
  await page.waitForSelector('.character-card', { timeout: 30000 });
  await page.waitForFunction(() => document.querySelectorAll('.character-card').length >= 2, null, { timeout: 30000 });
  const cards = await page.$$eval('.character-card', (nodes) => nodes.map((node) => ({ key: node.dataset.characterKey, members: node.querySelectorAll('.character-member').length, title: node.querySelector('strong').textContent })));
  step('candidates_shown', { cards, status: await page.textContent('#suggestionStatus') });
  expect(cards.length === 2, 'two candidate characters');
  expect(await page.isChecked('#selectionModeReference'), 'reference selection is the default mode');
  await page.waitForFunction(() => [...document.querySelectorAll('.character-member img')].every((img) => img.complete && img.naturalWidth > 0), null, { timeout: 30000 });
  step('candidate_images_loaded', { images: await page.$$eval('.character-member img', (nodes) => nodes.length) });
  await shot(page, '01-candidates.png');

  const [first, second] = cards;
  const memberSel = (key, index) => `.character-card[data-character-key="${key}"] .character-member >> nth=${index}`;
  // Multi-select: two images of character 1 (same character references) + character 2.
  await page.click(memberSel(first.key, 0));
  await page.click(memberSel(first.key, 1));
  await page.click(memberSel(second.key, 0));
  let pressed = await page.$$eval('.character-member[aria-pressed="true"]', (nodes) => nodes.map((n) => n.dataset.characterKey));
  step('multi_selected', { pressed });
  expect(pressed.length === 3, 'three images selected');
  // Deselect character 2: it must not reach extraction.
  await page.click(memberSel(second.key, 0));
  pressed = await page.$$eval('.character-member[aria-pressed="true"]', (nodes) => nodes.map((n) => n.dataset.characterKey));
  step('after_deselect', { pressed, count: await page.textContent('#suggestedSelectionCount') });
  expect(pressed.length === 2 && pressed.every((key) => key === first.key), 'only character 1 remains selected');
  await shot(page, '02-selection.png');

  // Logo cleanup would load a GPU inpainting model at export: keep it off here.
  if (await page.isChecked('#logoRemovalToggle')) await page.click('label[for="logoRemovalToggle"]');
  await page.click('#analyzeButton');
  step('analyze_clicked');
  await page.waitForSelector('.candidate-card', { timeout: 120000 });
  await page.waitForFunction(() => document.getElementById('progressArea').classList.contains('hidden'), null, { timeout: 120000 });
  const results = await page.$$eval('.candidate-card', (nodes) => nodes.map((node) => ({ id: node.dataset.id, evidence: node.querySelector('.candidate-evidence').textContent, time: node.querySelector('.candidate-time').textContent })));
  const runInfo = await page.textContent('#runInfo');
  step('results_shown', { count: results.length, results, runInfo, references: await page.textContent('#referenceCount') });
  expect(results.length >= 1, 'results displayed');
  const label = first.title.replace(' の候補', '');
  expect(results.every((item) => item.evidence.startsWith(`${label} ·`)), `every result belongs to ${label}`);
  expect(runInfo.includes('新規解析'), 'first analysis is a new run');
  await shot(page, '03-results.png');

  // Export one result to outputs/<video>/images/.
  await page.click('.candidate-card >> nth=0 >> .candidate-check');
  await page.click('#exportButton');
  await page.waitForFunction(() => /書き出し|保存|export/i.test(document.getElementById('toast').textContent || '') || !document.getElementById('globalNotice').classList.contains('hidden'), null, { timeout: 60000 });
  step('exported', { toast: await page.textContent('#toast'), notice: await page.textContent('#globalNotice') });
  await shot(page, '04-exported.png');

  // Same selection + same settings: the completed run is reused.
  await page.click('#analyzeButton');
  await page.waitForFunction(() => /再利用/.test(document.getElementById('runInfo').textContent || ''), null, { timeout: 60000 });
  step('rerun_reused', { runInfo: await page.textContent('#runInfo'), count: await page.$$eval('.candidate-card', (n) => n.length) });
  await shot(page, '05-rerun-reused.png');
} catch (error) {
  failed = error;
  step('failure', { message: String(error && error.stack || error) });
} finally {
  await browser.close();
  fs.writeFileSync(path.join(evidenceDir, 'ui-oplog.json'), JSON.stringify(log, null, 2));
}
if (failed) process.exit(1);
