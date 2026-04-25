import { expect, test, type Page } from '@playwright/test';

// --- helpers ---------------------------------------------------------------

/** Wait until the solver has loaded — the dice row appears. */
async function waitForLoaded(page: Page): Promise<void> {
  await page.locator('.die').first().waitFor({ timeout: 30_000 });
}

/**
 * Read a die's face by counting pip circles in its SVG. Each PIP_POS entry
 * has `face` circles, so circle count == face value (1..6).
 */
async function getDieFace(page: Page, i: number): Promise<number> {
  return page.locator('.die').nth(i).locator('circle').count();
}

async function getAllFaces(page: Page): Promise<number[]> {
  const out: number[] = [];
  for (let i = 0; i < 5; i++) out.push(await getDieFace(page, i));
  return out;
}

/** Click the i-th die forward until it shows `target`. */
async function setDie(page: Page, i: number, target: number): Promise<void> {
  const die = page.locator('.die').nth(i);
  for (let tries = 0; tries < 8; tries++) {
    const face = await die.locator('circle').count();
    if (face === target) return;
    await die.click();
  }
  throw new Error(`Failed to set die ${i} to ${target}`);
}

/** The currently active roll number (1, 2, or 3) per the segmented control. */
async function getRoll(page: Page): Promise<number> {
  const text = await page.locator('.seg.on').first().textContent();
  return parseInt(text?.trim() ?? '0', 10);
}

/** How many of the 13 score inputs currently have non-empty values. */
async function getFilledCount(page: Page): Promise<number> {
  return page.evaluate(
    () =>
      Array.from(
        document.querySelectorAll<HTMLInputElement>(
          'input.score-edit[type=text]',
        ),
      ).filter(el => el.value.trim() !== '').length,
  );
}

// --- tests -----------------------------------------------------------------

test('solver loads and the dice + scorecard render', async ({ page }) => {
  await page.goto('/');
  // "Loading solver…" should appear briefly while wasm + scores.bin fetch.
  // We don't assert on its visibility (it can flash too fast on a warm cache);
  // just wait for the loaded state.
  await waitForLoaded(page);
  await expect(page.getByText('Loading solver…')).toHaveCount(0);

  // 5 dice and 13 score-input rows render.
  await expect(page.locator('.die')).toHaveCount(5);
  await expect(page.locator('input.score-edit[type=text]')).toHaveCount(13);

  // Game starts on roll 1, scorecard empty, current total 0.
  expect(await getRoll(page)).toBe(1);
  expect(await getFilledCount(page)).toBe(0);
  await expect(page.getByText('Current score').locator('..').locator('strong')).toHaveText('0');

  // The recommendation panel renders some choices.
  await expect(page.locator('.choice-list li').first()).toBeVisible();
});

test('Enter applies the top recommendation through a complete game', async ({ page }) => {
  await page.goto('/');
  await waitForLoaded(page);

  // Each turn takes between 1 (score on roll 1) and 3 (keep, keep, score)
  // Enter presses to complete. 13 turns × 3 worst-case = 39 presses; we give
  // a generous 80-press budget to account for any reactive lag.
  const gameOver = page.locator('.game-over');
  for (let i = 0; i < 80; i++) {
    if (await gameOver.isVisible()) break;
    await page.keyboard.press('Enter');
    // Allow Svelte's reactive recompute to settle before the next press.
    await page.waitForTimeout(60);
  }

  await expect(gameOver).toBeVisible();

  // Final score should be plausible. Optimal play averages ~254.6; bad play
  // can go quite low; a multi-Yahtzee streak can push it well above 500.
  const scoreText = await gameOver.locator('strong').first().textContent();
  const score = parseInt(scoreText ?? '0', 10);
  expect(score).toBeGreaterThanOrEqual(50);
  expect(score).toBeLessThan(1500);

  // All 13 categories filled.
  expect(await getFilledCount(page)).toBe(13);
});

test('scoring while a Yahtzee is on the board auto-increments Extra Yahtzees', async ({ page }) => {
  await page.goto('/');
  await waitForLoaded(page);

  // 1. Manually set Yahtzee = 50 to enable the bonus chain.
  //    The 12th text input (idx 11) is the Yahtzee row.
  const yahtzeeInput = page.locator('input.score-edit[type=text]').nth(11);
  await yahtzeeInput.fill('50');
  await yahtzeeInput.press('Tab'); // commits via blur

  // 2. Set all five dice to face 5 (a Yahtzee of fives).
  for (let i = 0; i < 5; i++) await setDie(page, i, 5);

  // 3. Jump straight to roll 3 — only entry rows show in the choice list.
  await page.locator('.seg').nth(2).click(); // segment "3"
  expect(await getRoll(page)).toBe(3);

  // Bonus counter starts at 0. Note: the input is *disabled* before the
  // Sixes apply because turnsPlayed = 1 (only Yahtzee filled) → cap = 0.
  // It gets enabled once we fill a second category, bumping the cap to 1.
  const bonusInput = page.locator('input[type=number]');
  await expect(bonusInput).toHaveValue('0');

  // 4. Apply the Sixes row — scores 0 (no sixes), but triggers the bonus
  //    auto-increment because the dice are a Yahtzee and Yahtzee=50.
  const sixesRow = page.locator('.choice-list li', { hasText: 'Sixes' }).first();
  await sixesRow.getByRole('button', { name: 'Apply' }).click();

  // 5. Bonus counter is now 1, and the input is enabled (cap = turnsPlayed-1 = 1).
  await expect(bonusInput).toHaveValue('1');
  await expect(bonusInput).toBeEnabled();
});

test('undo reverses dice-cycle, roll, and apply actions', async ({ page }) => {
  await page.goto('/');
  await waitForLoaded(page);

  // --- 1. Dice cycle ---
  const initialFaces = await getAllFaces(page);
  await page.locator('.die').nth(0).click();
  expect(await getDieFace(page, 0)).toBe((initialFaces[0] % 6) + 1);
  await page.keyboard.press('u');
  expect(await getAllFaces(page)).toEqual(initialFaces);

  // --- 2. Roll (Space) ---
  expect(await getRoll(page)).toBe(1);
  await page.keyboard.press(' ');
  expect(await getRoll(page)).toBe(2);
  await page.keyboard.press('u');
  expect(await getRoll(page)).toBe(1);
  // Pre-roll dice are restored.
  expect(await getAllFaces(page)).toEqual(initialFaces);

  // --- 3. Apply (Enter) ---
  // Jump to roll 3 so Enter is guaranteed to apply a 'score' choice.
  await page.locator('.seg').nth(2).click();
  expect(await getRoll(page)).toBe(3);
  const preApplyFaces = await getAllFaces(page);
  const filledBefore = await getFilledCount(page);

  await page.keyboard.press('Enter');
  // After applying a score: roll resets to 1, dice are randomized, one new
  // entry filled.
  await expect.poll(() => getFilledCount(page)).toBe(filledBefore + 1);
  expect(await getRoll(page)).toBe(1);

  // Undo brings everything back.
  await page.keyboard.press('u');
  expect(await getRoll(page)).toBe(3);
  expect(await getAllFaces(page)).toEqual(preApplyFaces);
  expect(await getFilledCount(page)).toBe(filledBefore);
});
