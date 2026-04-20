<script lang="ts">
  import { onMount } from 'svelte';
  import init, { Solver, achievableScores, entryScore } from 'yahtzee-wasm';

  const ENTRY_LABELS = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    'Three of a kind', 'Four of a kind', 'Full house',
    'Small straight', 'Large straight', 'Yahtzee', 'Chance',
  ] as const;

  const HOW_TO_SCORE = [
    'count 1s', 'count 2s', 'count 3s', 'count 4s', 'count 5s', 'count 6s',
    'sum of dice', 'sum of dice', '25', '30', '40', '50', 'sum of dice',
  ] as const;

  const ENTRY_NAME_TO_IDX: Record<string, number> = {
    ones: 0, twos: 1, threes: 2, fours: 3, fives: 4, sixes: 5,
    three_of_a_kind: 6, four_of_a_kind: 7, full_house: 8,
    small_straight: 9, large_straight: 10, yahtzee: 11, chance: 12,
  };

  let solver: Solver | null = null;
  let loading = true;
  let loadError: string | null = null;

  let rawInputs: string[] = new Array(13).fill('');
  let yahtzeeBonuses = 0;

  type Snapshot = {
    rawInputs: string[];
    yahtzeeBonuses: number;
    dice: number[];
    kept: boolean[];
    roll: 1 | 2 | 3;
  };
  let history: Snapshot[] = [];

  function snapshot(): Snapshot {
    return {
      rawInputs: [...rawInputs],
      yahtzeeBonuses,
      dice: [...dice],
      kept: [...kept],
      roll,
    };
  }
  function pushHistory() {
    history = [...history, snapshot()];
    if (history.length > 50) history = history.slice(-50);
  }
  function undo() {
    if (history.length === 0) return;
    const last = history[history.length - 1];
    history = history.slice(0, -1);
    rawInputs = [...last.rawInputs];
    yahtzeeBonuses = last.yahtzeeBonuses;
    dice = [...last.dice];
    kept = [...last.kept];
    roll = last.roll;
  }

  let allowedByEntry: number[][] = [];
  let allowedSets: Set<number>[] = [];
  let allowedLabels: string[] = [];
  let wasmReady = false;

  function formatAllowed(vals: number[]): string {
    if (vals.length <= 6) return vals.join(', ');
    const contiguous = (xs: number[]) => xs.every((v, i) => i === 0 || v === xs[i - 1] + 1);
    if (contiguous(vals)) return `${vals[0]}–${vals[vals.length - 1]}`;
    if (vals[0] === 0) {
      const rest = vals.slice(1);
      if (contiguous(rest)) return `0 or ${rest[0]}–${rest[rest.length - 1]}`;
    }
    return vals.join(', ');
  }

  function parseRaw(raw: string, i: number): { value: number | null; error: string | null } {
    const t = raw.trim();
    if (t === '') return { value: null, error: null };
    if (!/^-?\d+$/.test(t)) return { value: null, error: 'whole number' };
    const n = parseInt(t, 10);
    if (!wasmReady) return { value: n, error: null };
    if (allowedSets[i].has(n)) return { value: n, error: null };
    return { value: null, error: allowedLabels[i] };
  }

  $: parsed = rawInputs.map((r, i) => parseRaw(r, i));
  $: scores = parsed.map(p => p.value);
  $: errors = parsed.map(p => p.error);
  let dice: number[] = [1, 1, 1, 1, 1];
  let kept: boolean[] = [false, false, false, false, false];
  let roll: 1 | 2 | 3 = 1;

  type KeeperRec = { dice: number[]; ev: number; turn_ev: number; best_entry: string | null };
  type EntryRec = { entry: string; ev: number; turn_ev: number };
  type Recommendation = {
    value: number;
    keepers: KeeperRec[] | null;
    entries: EntryRec[] | null;
  };
  let rec: Recommendation | null = null;
  let recError: string | null = null;

  onMount(async () => {
    try {
      await init();
      allowedByEntry = Array.from({ length: 13 }, (_, i) => Array.from(achievableScores(i)));
      allowedSets = allowedByEntry.map(vs => new Set(vs));
      allowedLabels = allowedByEntry.map(formatAllowed);
      wasmReady = true;
      const resp = await fetch('/scores.bin');
      if (!resp.ok) throw new Error(`fetch scores.bin: ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      solver = new Solver(bytes);
      dice = randomDice();
    } catch (e) {
      loadError = String(e);
    } finally {
      loading = false;
    }
  });

  $: entries = scores.map(s => s !== null);
  $: upperFilled = scores.slice(0, 6).reduce<number>((a, b) => a + (b ?? 0), 0);
  $: lowerFilled = scores.slice(6).reduce<number>((a, b) => a + (b ?? 0), 0);
  $: upperScoreRemaining = Math.max(0, 63 - upperFilled);
  $: upperBonus = upperFilled >= 63 ? 35 : 0;
  $: turnsPlayed = scores.filter(s => s !== null).length;
  $: gameOver = turnsPlayed === 13;
  $: maxYahtzeeBonuses = scores[11] === 50 ? Math.max(0, turnsPlayed - 1) : 0;
  $: {
    const clamped = Math.max(0, Math.min(yahtzeeBonuses | 0, maxYahtzeeBonuses));
    if (clamped !== yahtzeeBonuses) yahtzeeBonuses = clamped;
  }
  $: yahtzeeBonusPoints = yahtzeeBonuses * 100;
  $: currentTotal = upperFilled + upperBonus + lowerFilled + yahtzeeBonusPoints;
  $: yahtzeeBonusEligible = scores[11] === 50;

  $: stateInput = {
    entries,
    yahtzee_bonus_eligible: yahtzeeBonusEligible,
    upper_score_remaining: upperScoreRemaining,
  };

  $: if (solver) {
    try {
      rec = solver.recommend(stateInput, new Uint8Array(dice), roll) as Recommendation;
      recError = null;
    } catch (e) {
      rec = null;
      recError = String(e);
    }
  }

  function cycleDie(i: number, delta: number) {
    pushHistory();
    const next = ((dice[i] - 1 + delta + 6) % 6) + 1;
    dice = dice.map((v, j) => (j === i ? next : v));
  }

  $: jokerActive = (() => {
    const c = [0, 0, 0, 0, 0, 0];
    for (const v of dice) c[v - 1]++;
    const face = c.findIndex(x => x === 5) + 1;
    if (face === 0) return false;
    return scores[11] !== null && scores[face - 1] !== null;
  })();

  function isJokerEnabled(i: number): boolean {
    return jokerActive && (i === 8 || i === 9 || i === 10);
  }

  type Choice = {
    kind: 'keep' | 'score';
    dice: number[];
    entry: string | null;
    turn_ev: number;
    ev: number;
  };

  $: choices = ((): Choice[] => {
    if (rec?.keepers) {
      return rec.keepers.slice(0, 8).map(k => ({
        kind: k.best_entry ? ('score' as const) : ('keep' as const),
        dice: k.dice,
        entry: k.best_entry,
        turn_ev: k.turn_ev,
        ev: k.ev,
      }));
    }
    if (rec?.entries) {
      return rec.entries.map(e => ({
        kind: 'score' as const,
        dice: [],
        entry: e.entry,
        turn_ev: e.turn_ev,
        ev: e.ev,
      }));
    }
    return [];
  })();

  $: hasJokerChoice = choices.some(c => {
    if (!c.entry) return false;
    const idx = ENTRY_NAME_TO_IDX[c.entry];
    return idx !== undefined && isJokerEnabled(idx);
  });

  function keepMaskFor(faces: number[], currentDice: number[]): boolean[] {
    const mask = [false, false, false, false, false];
    const remaining = [0, 0, 0, 0, 0, 0];
    for (const v of faces) remaining[v - 1]++;
    for (let i = 0; i < 5; i++) {
      const f = currentDice[i];
      if (remaining[f - 1] > 0) {
        mask[i] = true;
        remaining[f - 1]--;
      }
    }
    return mask;
  }

  function toggleKeep(i: number) {
    kept = kept.map((v, j) => (j === i ? !v : v));
  }

  function randomDice(): number[] {
    return Array.from({ length: 5 }, () => 1 + Math.floor(Math.random() * 6));
  }

  function randomizeDice() {
    if (gameOver) return;
    pushHistory();
    dice = randomDice();
    kept = [false, false, false, false, false];
  }

  function rollDice() {
    if (gameOver || roll === 3) return;
    if (kept.every(k => k)) return;
    const next = dice.slice();
    for (let i = 0; i < 5; i++) {
      if (!kept[i]) next[i] = 1 + Math.floor(Math.random() * 6);
    }
    dice = next;
    kept = [false, false, false, false, false];
    roll = (roll + 1) as 1 | 2 | 3;
  }

  function handleRoll() {
    if (gameOver || roll === 3 || kept.every(k => k)) return;
    pushHistory();
    rollDice();
  }

  function applyKeepAndRoll(faces: number[]) {
    if (gameOver || roll === 3) return;
    pushHistory();
    kept = keepMaskFor(faces, dice);
    rollDice();
  }

  function isYahtzee(d: number[]): boolean {
    return d.every(v => v === d[0]);
  }

  function applyScore(entryName: string) {
    if (gameOver) return;
    const idx = ENTRY_NAME_TO_IDX[entryName];
    if (idx === undefined) return;
    pushHistory();
    const state = {
      entries,
      yahtzee_bonus_eligible: yahtzeeBonusEligible,
      upper_score_remaining: upperScoreRemaining,
    };
    const val = entryScore(state, new Uint8Array(dice), idx);
    rawInputs[idx] = String(val);
    if (isYahtzee(dice) && yahtzeeBonusEligible && idx !== 11) {
      yahtzeeBonuses = yahtzeeBonuses + 1;
    }
    const otherFilled = entries.reduce((a, b, i) => a + (i !== idx && b ? 1 : 0), 0);
    if (otherFilled + 1 >= 13) {
      kept = [false, false, false, false, false];
      return;
    }
    dice = randomDice();
    kept = [false, false, false, false, false];
    roll = 1;
  }

  function scoreContributionMask(entryIdx: number, d: number[]): boolean[] {
    if (entryIdx >= 0 && entryIdx <= 5) {
      const face = entryIdx + 1;
      return d.map(v => v === face);
    }
    if (entryIdx === 9) {
      const present = new Set(d);
      const runs = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]];
      let chosen: number[] | null = null;
      for (const r of runs) {
        if (r.every(x => present.has(x))) { chosen = r; break; }
      }
      if (!chosen) return [false, false, false, false, false];
      return keepMaskFor(chosen, d);
    }
    return [true, true, true, true, true];
  }

  $: recKeepMask = (() => {
    const top = choices[0];
    if (!top) return [false, false, false, false, false];
    if (top.kind === 'keep') {
      if (top.dice.length === 0) return [false, false, false, false, false];
      return keepMaskFor(top.dice, dice);
    }
    if (top.kind === 'score' && top.entry) {
      const idx = ENTRY_NAME_TO_IDX[top.entry];
      if (idx === undefined) return [false, false, false, false, false];
      return scoreContributionMask(idx, dice);
    }
    return [false, false, false, false, false];
  })();
  $: anyRecommended = recKeepMask.some(Boolean);

  $: scoreHighlightIdx = (() => {
    const top = choices[0];
    if (!top || top.kind !== 'score' || !top.entry) return -1;
    return ENTRY_NAME_TO_IDX[top.entry] ?? -1;
  })();

  function setRoll(r: number) {
    if (r === roll) return;
    pushHistory();
    roll = r as 1 | 2 | 3;
  }

  function resetGame() {
    pushHistory();
    rawInputs = new Array(13).fill('');
    yahtzeeBonuses = 0;
    dice = randomDice();
    kept = [false, false, false, false, false];
    roll = 1;
  }

  const PIP_POS: Record<number, [number, number][]> = {
    1: [[50, 50]],
    2: [[25, 25], [75, 75]],
    3: [[25, 25], [50, 50], [75, 75]],
    4: [[25, 25], [75, 25], [25, 75], [75, 75]],
    5: [[25, 25], [75, 25], [50, 50], [25, 75], [75, 75]],
    6: [[25, 25], [75, 25], [25, 50], [75, 50], [25, 75], [75, 75]],
  };
</script>

<main>
  <header>
    <h1>easy-yahtzee</h1>
    <div class="header-actions">
      <button class="reset" disabled={history.length === 0} on:click={undo}>Undo</button>
      <button class="reset" on:click={resetGame}>Reset</button>
    </div>
  </header>

  {#if loading}
    <p>Loading solver…</p>
  {:else if loadError}
    <p class="error">Failed to load: {loadError}</p>
  {:else}
    <div class="layout">
      <section class="scorecard">
        <div class="section-label">Upper</div>
        <table class="sheet">
          <colgroup>
            <col />
            <col class="col-howto" />
            <col class="col-score" />
          </colgroup>
          <tbody>
            {#each ENTRY_LABELS.slice(0, 6) as label, i}
              {@const filled = scores[i] !== null}
              {@const err = errors[i]}
              <tr class:filled class:invalid={err !== null} class:best={scoreHighlightIdx === i && !filled}>
                <td>{label}</td>
                <td class="howto">
                  <svg class="pip-icon" viewBox="0 0 100 100" aria-hidden="true">
                    <rect x="8" y="8" width="84" height="84" rx="16" ry="16" />
                    {#each PIP_POS[i + 1] as [cx, cy]}
                      <circle cx={cx} cy={cy} r="14" />
                    {/each}
                  </svg>
                  {HOW_TO_SCORE[i]}
                </td>
                <td class="score-cell">
                  <input
                    class="score-edit"
                    class:invalid={err !== null}
                    type="text"
                    inputmode="numeric"
                    bind:value={rawInputs[i]}
                  />
                  {#if err}<div class="err">{err}</div>{/if}
                </td>
              </tr>
            {/each}
            <tr class="computed">
              <td>Upper subtotal</td>
              <td class="howto arrow">→</td>
              <td>{upperFilled} / 63</td>
            </tr>
            <tr class="computed">
              <td>Upper bonus</td>
              <td class="howto">+35 at ≥63</td>
              <td>{upperBonus === 35 ? '+35' : '0'}</td>
            </tr>
          </tbody>
        </table>

        <div class="section-label">Lower</div>
        <table class="sheet">
          <colgroup>
            <col />
            <col class="col-howto" />
            <col class="col-score" />
          </colgroup>
          <tbody>
            {#each ENTRY_LABELS.slice(6) as label, j}
              {@const i = j + 6}
              {@const filled = scores[i] !== null}
              {@const err = errors[i]}
              {@const joker = isJokerEnabled(i)}
              <tr class:filled class:invalid={err !== null} class:best={scoreHighlightIdx === i && !filled}>
                <td>
                  {label}
                  {#if joker}<a
                    class="joker"
                    href="https://en.wikipedia.org/wiki/Yahtzee#Yahtzee_bonuses_and_Joker_rules"
                    target="_blank"
                    rel="noopener noreferrer"
                  >Joker Rule</a>{/if}
                </td>
                <td class="howto">{HOW_TO_SCORE[i]}</td>
                <td class="score-cell">
                  <input
                    class="score-edit"
                    class:invalid={err !== null}
                    type="text"
                    inputmode="numeric"
                    bind:value={rawInputs[i]}
                  />
                  {#if err}<div class="err">{err}</div>{/if}
                </td>
              </tr>
            {/each}
            <tr class="computed">
              <td>
                <label class="inline">
                  Extra Yahtzees
                  <input
                    type="number"
                    min="0"
                    max={maxYahtzeeBonuses}
                    class="score-edit"
                    disabled={maxYahtzeeBonuses === 0}
                    bind:value={yahtzeeBonuses}
                  />
                </label>
              </td>
              <td class="howto">+100 each</td>
              <td>
                {yahtzeeBonusPoints ? `+${yahtzeeBonusPoints}` : '0'}
                {#if yahtzeeBonusEligible}<span class="note">(eligible)</span>{/if}
              </td>
            </tr>
          </tbody>
        </table>

        <div class="totals">
          <div><span>Current score</span><strong>{currentTotal}</strong></div>
          <div><span>Projected final</span><strong>{rec ? (currentTotal + rec.value).toFixed(1) : '—'}</strong></div>
          <div><span>Turn</span><strong>{Math.min(turnsPlayed + (gameOver ? 0 : 1), 13)} / 13</strong></div>
        </div>
      </section>

      <section class="play">
        {#if gameOver}
          <div class="game-over">
            <h2>Game over</h2>
            <p>Final score <strong>{currentTotal}</strong></p>
            <button class="roll-btn primary" on:click={resetGame}>New game</button>
          </div>
        {:else}
        <div class="dice-row">
          {#each dice as face, i}
            <div class="die-cell" class:faded={anyRecommended && !recKeepMask[i]}>
              <button
                class="die"
                on:click={() => cycleDie(i, 1)}
                on:contextmenu|preventDefault={() => cycleDie(i, -1)}
                on:wheel|preventDefault={(e) => cycleDie(i, e.deltaY > 0 ? 1 : -1)}
                title="Click to advance face, right-click or scroll to reverse"
              >
                <svg viewBox="0 0 100 100">
                  <rect x="4" y="4" width="92" height="92" rx="14" ry="14" />
                  {#each PIP_POS[face] as [cx, cy]}
                    <circle cx={cx} cy={cy} r="9" />
                  {/each}
                </svg>
              </button>
              <label class="keep-toggle">
                <input type="checkbox" checked={kept[i]} on:change={() => toggleKeep(i)} />
                keep
              </label>
            </div>
          {/each}
        </div>

        <div class="roll-row">
          <span class="label">Roll</span>
          {#each [1, 2, 3] as r}
            <button class="seg" class:on={roll === r} on:click={() => setRoll(r)}>
              {r}
            </button>
          {/each}
          <button
            class="roll-btn"
            disabled={roll === 3 || kept.every(k => k)}
            on:click={handleRoll}
            title={roll === 3 ? 'No rolls left' : (kept.every(k => k) ? 'All dice kept' : 'Re-roll dice not marked keep')}
          >
            Roll
          </button>
          <button
            class="roll-btn"
            on:click={randomizeDice}
            title="Randomize all dice without advancing the roll"
          >
            Randomize
          </button>
        </div>

        <div class="rec">
          {#if recError}
            <p class="error">{recError}</p>
          {:else if rec}
            {#if choices.length > 0}
              <div class="entry-head">
                <span></span>
                <span>turn</span>
                <span>final</span>
                <span></span>
              </div>
              <ol class="choice-list">
                {#each choices as c, ci}
                  {@const entryIdx = c.entry ? ENTRY_NAME_TO_IDX[c.entry] : undefined}
                  {@const joker = entryIdx !== undefined && isJokerEnabled(entryIdx)}
                  <li class:top={ci === 0}>
                    <span class="choice-label">
                      {#if c.kind === 'score'}
                        <em>score in</em>
                        <span class="chip entry">
                          {entryIdx !== undefined ? ENTRY_LABELS[entryIdx] : c.entry}
                        </span>
                        {#if joker}<span class="joker-dot" aria-hidden="true">★</span>{/if}
                      {:else if c.dice.length === 0}
                        <em>re-roll all</em>
                      {:else}
                        <em>keep</em>
                        {#each c.dice as d}
                          <svg class="mini-die" viewBox="0 0 100 100" aria-label={String(d)}>
                            <rect x="6" y="6" width="88" height="88" rx="16" ry="16" />
                            {#each PIP_POS[d] as [cx, cy]}
                              <circle cx={cx} cy={cy} r="13" />
                            {/each}
                          </svg>
                        {/each}
                      {/if}
                    </span>
                    <code>{roll === 3 ? c.turn_ev.toFixed(0) : c.turn_ev.toFixed(2)}</code>
                    <code>{(currentTotal + c.ev).toFixed(1)}</code>
                    {#if c.kind === 'score' && c.entry}
                      {@const entryName = c.entry}
                      <button class="apply" class:primary={ci === 0} on:click={() => applyScore(entryName)}>Apply</button>
                    {:else if c.kind === 'keep'}
                      <button class="apply" class:primary={ci === 0} disabled={roll === 3} on:click={() => applyKeepAndRoll(c.dice)}>Apply</button>
                    {:else}
                      <span></span>
                    {/if}
                  </li>
                {/each}
              </ol>
              {#if hasJokerChoice}
                <p class="caption">
                  ★ =
                  <a href="https://en.wikipedia.org/wiki/Yahtzee#Yahtzee_bonuses_and_Joker_rules" target="_blank" rel="noopener noreferrer">Joker Rule</a>
                </p>
              {/if}
            {/if}
          {/if}
        </div>
        {/if}
      </section>
    </div>
  {/if}
</main>

<style>
  :global(body) {
    background: var(--bg-page, #f7f6f2);
    margin: 0;
    color: var(--fg, #222);
  }
  main {
    --text-xs: 0.85rem;
    --text-sm: 1rem;
    --text-md: 1.1rem;
    --text-lg: 1.75rem;

    --fg: #222;
    --fg-muted: #555;
    --fg-dim: #888;
    --fg-faint: #bbb;
    --fg-strike: #999;
    --fg-warm: #7a6f4f;

    --accent: #4a7fb0;
    --accent-dark: #2d4f6e;

    --border: #d4cdb8;
    --border-soft: #f1ecdc;
    --border-faint: #eee;
    --border-input: #ccc;

    --bg-page: #f7f6f2;
    --bg-card: #fff;
    --bg-muted: #f3f1ea;
    --bg-subtle: #f1efe7;

    --primary: #333;
    --primary-hover: #000;

    --err: #b00020;
    --err-bg: #fdecef;
    --err-row-bg: #fdf3f4;

    --eyebrow-spacing: 0.05em;

    max-width: 1080px;
    margin: 0 auto;
    padding: 1.25rem 1rem 3rem;
    font-family: system-ui, sans-serif;
  }
  header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 1rem;
  }
  h1 { margin: 0; font-size: var(--text-lg); }
  .header-actions { display: flex; gap: 0.5rem; }
  .reset {
    background: transparent;
    border: 1px solid var(--fg-faint);
    border-radius: 4px;
    padding: 0.25rem 0.75rem;
    cursor: pointer;
    font-size: var(--text-md);
  }
  .reset:disabled {
    color: var(--fg-faint);
    border-color: #ddd;
    cursor: not-allowed;
  }

  .layout {
    display: grid;
    grid-template-columns: minmax(420px, 1fr) minmax(320px, 460px);
    gap: 1.5rem;
    align-items: start;
  }
  @media (max-width: 760px) {
    .layout { grid-template-columns: 1fr; }
  }

  .scorecard {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .section-label {
    font-weight: 700;
    font-size: var(--text-md);
    letter-spacing: var(--eyebrow-spacing);
    text-transform: uppercase;
    color: var(--fg);
    margin: 0.75rem 0 0.25rem;
  }
  table.sheet {
    width: 100%;
    border-collapse: collapse;
    border: 1px solid var(--fg);
  }
  table.sheet td {
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--fg);
    font-size: var(--text-md);
  }
  table.sheet td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  table.sheet .col-howto { width: 5.5rem; }
  table.sheet .col-score { width: 4.75rem; }
  td.howto {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: left !important;
    white-space: nowrap;
  }
  td.howto.arrow {
    font-size: var(--text-lg);
    color: var(--fg);
    text-align: center !important;
    line-height: 1;
  }
  .pip-icon {
    width: 0.95rem;
    height: 0.95rem;
    display: inline-block;
    vertical-align: -0.18em;
    margin-right: 0.25rem;
  }
  .pip-icon rect { fill: var(--bg-card); stroke: var(--fg); stroke-width: 7; }
  .pip-icon circle { fill: var(--fg); }
  tr.filled td:first-child { color: var(--fg-strike); text-decoration: line-through; }
  tr.best td:first-child {
    font-weight: 600;
    box-shadow: inset 3px 0 0 var(--accent);
    padding-left: calc(0.5rem + 3px);
  }
  tr.computed td { color: var(--fg-muted); font-size: var(--text-sm); }
  tr.computed td:first-child { color: var(--fg); }
  .note { color: var(--fg-dim); font-size: var(--text-xs); font-weight: normal; }
  .joker {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: #6b4a06;
    background: #fff3c4;
    border: 1px solid #e2c868;
    border-radius: 3px;
    padding: 0 0.3rem;
    margin-left: 0.3rem;
    vertical-align: middle;
    text-decoration: none;
  }
  .joker:hover { background: #ffe899; }
  .joker-dot { color: #c39509; font-size: 0.75rem; margin-left: 0.2rem; }

  .score-edit {
    width: 3.4rem;
    text-align: right;
    padding: 0.15rem 0.3rem;
    border: 1px solid var(--border-input);
    border-radius: 3px;
    font-size: var(--text-md);
    font-variant-numeric: tabular-nums;
    background: var(--bg-card);
  }
  .score-edit::placeholder { color: var(--fg-faint); font-style: italic; }
  .score-edit:focus { outline: 2px solid #c9bf9c; outline-offset: 0; }
  .score-edit.invalid { border-color: var(--err); background: var(--err-bg); color: var(--err); }
  .score-edit.invalid:focus { outline-color: var(--err); }
  .err { color: var(--err); font-size: var(--text-xs); }
  tr.invalid { background: var(--err-row-bg); }
  .inline {
    display: inline-flex;
    gap: 0.4rem;
    align-items: center;
    font-size: var(--text-sm);
    color: var(--fg-muted);
  }

  .totals {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 2px solid var(--border);
  }
  .totals > div {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .totals span {
    font-size: var(--text-xs);
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: var(--eyebrow-spacing);
  }
  .totals strong {
    font-size: var(--text-lg);
    font-variant-numeric: tabular-nums;
  }

  .play { display: flex; flex-direction: column; gap: 1rem; }

  .dice-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.5rem;
  }
  .die-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    transition: opacity 0.15s;
  }
  .die-cell.faded { opacity: 0.35; }
  .die-cell.faded:hover { opacity: 0.7; }
  .keep-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    font-size: var(--text-xs);
    color: var(--fg-muted);
    cursor: pointer;
    user-select: none;
  }
  .keep-toggle input { margin: 0; cursor: pointer; accent-color: var(--accent); }
  .die {
    aspect-ratio: 1;
    padding: 0;
    border: none;
    background: transparent;
    cursor: pointer;
    width: 100%;
  }
  .die svg { width: 100%; height: 100%; display: block; }
  .die svg rect {
    fill: var(--bg-card);
    stroke: var(--primary);
    stroke-width: 3;
    transition: stroke 0.15s, stroke-width 0.15s;
  }
  .die svg circle { fill: var(--fg); }
  .die:hover svg rect { stroke: var(--primary-hover); }

  .roll-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
  }
  .roll-row .label { color: var(--fg-muted); font-size: var(--text-md); margin-right: 0.25rem; }
  .seg {
    width: 2.5rem;
    height: 2rem;
    border: 1px solid var(--fg-faint);
    background: var(--bg-card);
    cursor: pointer;
    font-size: var(--text-md);
  }
  .seg:first-of-type { border-radius: 4px 0 0 4px; }
  .seg:last-of-type { border-radius: 0 4px 4px 0; }
  .seg.on {
    background: var(--bg-subtle);
    color: var(--fg);
    font-weight: 600;
    box-shadow: inset 0 -3px 0 var(--accent);
  }
  .roll-btn {
    margin-left: 0.4rem;
    height: 2rem;
    padding: 0 0.9rem;
    border: 1px solid var(--fg-faint);
    background: var(--bg-card);
    color: var(--fg);
    border-radius: 4px;
    cursor: pointer;
    font-size: var(--text-md);
  }
  .roll-btn:hover:not(:disabled) { background: var(--bg-subtle); border-color: var(--fg-muted); }
  .roll-btn:disabled {
    background: #f0f0f0;
    border-color: #ddd;
    color: var(--fg-faint);
    cursor: not-allowed;
  }
  .roll-btn.primary {
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
  }
  .roll-btn.primary:hover:not(:disabled) { background: var(--primary-hover); border-color: var(--primary-hover); }

  .rec {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  .entry-head {
    display: grid;
    grid-template-columns: 1fr 3.5rem 4rem 4rem;
    gap: 0.5rem;
    padding: 0.2rem 0.3rem;
    font-size: var(--text-xs);
    color: var(--fg-dim);
    text-transform: uppercase;
    letter-spacing: var(--eyebrow-spacing);
    white-space: nowrap;
  }
  .entry-head > span:not(:first-child) { text-align: right; }
  .caption { color: var(--fg-dim); font-size: var(--text-xs); margin: 0.25rem 0 0.5rem; }
  .choice-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .choice-list li {
    display: grid;
    grid-template-columns: 1fr 3.5rem 4rem 4rem;
    gap: 0.5rem;
    align-items: center;
    padding: 0.3rem 0.4rem;
    border-bottom: 1px dashed var(--border-faint);
    border-radius: 4px;
    font-size: var(--text-sm);
  }
  .choice-list li.top {
    box-shadow: inset 3px 0 0 var(--accent);
    padding-left: calc(0.4rem + 3px);
  }
  .choice-list li.top .choice-label em { color: var(--accent-dark); font-weight: 600; }
  .choice-list code { text-align: right; }
  .apply {
    justify-self: end;
    padding: 0.2rem 0.55rem;
    border: 1px solid var(--fg-faint);
    background: var(--bg-card);
    color: var(--fg);
    border-radius: 4px;
    cursor: pointer;
    font-size: var(--text-sm);
  }
  .apply:hover:not(:disabled) { background: var(--bg-subtle); border-color: var(--fg-muted); }
  .apply:disabled { color: var(--fg-faint); cursor: not-allowed; }
  .apply.primary {
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
  }
  .apply.primary:hover:not(:disabled) { background: var(--primary-hover); border-color: var(--primary-hover); }
  .apply.primary:disabled { background: #ddd; border-color: #ccc; color: var(--fg-dim); }
  .game-over {
    text-align: center;
    padding: 2rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .game-over h2 { margin: 0 0 0.5rem; }
  .game-over p { margin: 0 0 1rem; color: var(--fg-muted); }
  .game-over strong { font-size: var(--text-lg); }
  .choice-label { display: inline-flex; gap: 0.3rem; align-items: center; flex-wrap: wrap; }
  .choice-label em { font-style: normal; color: var(--fg-dim); font-size: var(--text-sm); }
  .mini-die {
    width: 1.7rem;
    height: 1.7rem;
    display: inline-block;
    vertical-align: middle;
  }
  .mini-die rect { fill: var(--bg-card); stroke: var(--primary); stroke-width: 5; }
  .mini-die circle { fill: var(--fg); }
  .chip {
    display: inline-block;
    min-width: 1.3rem;
    text-align: center;
    background: #eee;
    border-radius: 3px;
    padding: 0.05rem 0.3rem;
    font-variant-numeric: tabular-nums;
  }
  .chip.entry { min-width: 0; padding: 0.05rem 0.4rem; font-variant-numeric: normal; }
  code {
    background: #f4f4f4;
    padding: 0.1rem 0.3rem;
    border-radius: 3px;
    font-size: var(--text-sm);
  }
  .error { color: var(--err); }
</style>
