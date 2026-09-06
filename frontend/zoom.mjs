// Dev-only: screenshot just the Fig. 1 pipeline diagram at 3x scale, for
// inspecting fine detail (arrowheads, glow, edge colour) that a full-page
// screenshot compresses too much to see.
// Usage: node zoom.mjs [outfile] ["question to run first"]
import { chromium } from 'playwright'

const out = process.argv[2] || 'zoom.png'
const question = process.argv[3]

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 3 })
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.waitForTimeout(2000)

if (question) {
  await page.fill('.ask-input', question)
  await page.click('button[type=submit]')
  await page
    .waitForFunction(() => !document.querySelector('.btn-stop'), { timeout: 120000 })
    .catch(() => {})
  await page.waitForTimeout(1000)
}

await page.locator('.screen').screenshot({ path: out })
console.log('wrote', out)
await browser.close()
