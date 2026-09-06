// Dev-only: screenshot the running app so the UI can be reviewed visually.
// Usage: node shot.mjs [outfile] [--ask "question"]
import { chromium } from 'playwright'

const out = process.argv[2] || 'shot.png'
const askIndex = process.argv.indexOf('--ask')
const question = askIndex > -1 ? process.argv[askIndex + 1] : null

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })

page.on('console', (m) => {
  if (m.type() === 'error') console.log('CONSOLE ERROR:', m.text())
})
page.on('pageerror', (e) => console.log('PAGE ERROR:', e.message))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.waitForTimeout(2500) // let webfonts settle

if (question) {
  await page.fill('.ask-input', question)
  await page.click('button[type=submit]')
  // wait for the run to finish (Stop button disappears) or time out
  await page
    .waitForFunction(() => !document.querySelector('.btn-stop'), { timeout: 120000 })
    .catch(() => console.log('(still running at timeout)'))
  await page.waitForTimeout(1200)
}

await page.screenshot({ path: out, fullPage: true })
console.log('wrote', out)
await browser.close()
