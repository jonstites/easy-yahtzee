<script lang="ts">
  import { onMount } from 'svelte';
  import init, { Solver, achievableScores } from 'yahtzee-wasm';

  const ENTRY_LABELS = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    'Three of a kind', 'Four of a kind', 'Full house',
    'Small straight', 'Large straight', 'Yahtzee', 'Chance',
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

  $: expectedFinal = rec ? currentTotal + rec.value : null;

  let turnEv: number | null = null;
  $: {
    if (solver && !recError) {
      try {
        turnEv = solver.thisTurnEv(stateInput, new Uint8Array(dice), roll);
      } catch {
        turnEv = null;
      }
    } else {
      turnEv = null;
    }
  }

  function cycleDie(i: number, delta: number) {
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

  $: keepMask = (() => {
    const mask = [false, false, false, false, false];
    const top = choices[0];
    if (!top || top.kind !== 'keep' || top.dice.length === 0) return mask;
    const remaining = [0, 0, 0, 0, 0, 0];
    for (const v of top.dice) remaining[v - 1]++;
    for (let i = 0; i < 5; i++) {
      const f = dice[i];
      if (remaining[f - 1] > 0) {
        mask[i] = true;
        remaining[f - 1]--;
      }
    }
    return mask;
  })();

  $: scoreHighlightIdx = (() => {
    const top = choices[0];
    if (!top || top.kind !== 'score' || !top.entry) return -1;
    return ENTRY_NAME_TO_IDX[top.entry] ?? -1;
  })();

  function setRoll(r: number) {
    roll = r as 1 | 2 | 3;
  }

  function resetGame() {
    rawInputs = new Array(13).fill('');
    yahtzeeBonuses = 0;
    dice = [1, 1, 1, 1, 1];
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
    <button class="reset" on:click={resetGame}>Reset</button>
  </header>

  {#if loading}
    <p>Loading solver…</p>
  {:else if loadError}
    <p class="error">Failed to load: {loadError}</p>
  {:else}
    <div class="layout">
      <section class="scorecard">
        <div class="section-label">Upper</div>
        <table>
          <tbody>
            {#each ENTRY_LABELS.slice(0, 6) as label, i}
              {@const filled = scores[i] !== null}
              {@const err = errors[i]}
              <tr class:filled class:invalid={err !== null} class:best={scoreHighlightIdx === i && !filled}>
                <td>{label}</td>
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
              <td>{upperFilled} / 63</td>
            </tr>
            <tr class="computed">
              <td>Upper bonus</td>
              <td>{upperBonus === 35 ? '+35' : '0'}</td>
            </tr>
          </tbody>
        </table>

        <div class="section-label">Lower</div>
        <table>
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
              <td>
                {yahtzeeBonusPoints ? `+${yahtzeeBonusPoints}` : '0'}
                {#if yahtzeeBonusEligible}<span class="note">(eligible)</span>{/if}
              </td>
            </tr>
          </tbody>
        </table>

        <div class="totals">
          <div><span>Current score</span><strong>{currentTotal}</strong></div>
        </div>
      </section>

      <section class="play">
        <div class="dice-row">
          {#each dice as face, i}
            <button
              class="die"
              class:keep={keepMask[i]}
              on:click={() => cycleDie(i, 1)}
              on:contextmenu|preventDefault={() => cycleDie(i, -1)}
              on:wheel|preventDefault={(e) => cycleDie(i, e.deltaY > 0 ? 1 : -1)}
              title="Click to advance, right-click or scroll to reverse"
            >
              <svg viewBox="0 0 100 100">
                <rect x="4" y="4" width="92" height="92" rx="14" ry="14" />
                {#each PIP_POS[face] as [cx, cy]}
                  <circle cx={cx} cy={cy} r="9" />
                {/each}
              </svg>
            </button>
          {/each}
        </div>

        <div class="roll-row">
          <span class="label">Roll</span>
          {#each [1, 2, 3] as r}
            <button class="seg" class:on={roll === r} on:click={() => setRoll(r)}>
              {r}
            </button>
          {/each}
        </div>

        <div class="rec">
          {#if recError}
            <p class="error">{recError}</p>
          {:else if rec}
            <div class="ev-summary">
              {#if turnEv !== null}
                <div>
                  <span>EV this turn</span>
                  <strong>{roll === 3 ? turnEv.toFixed(0) : turnEv.toFixed(2)}</strong>
                </div>
              {/if}
              <div>
                <span>EV final</span>
                <strong>{expectedFinal !== null ? expectedFinal.toFixed(1) : '—'}</strong>
              </div>
            </div>
            {#if choices.length > 0}
              <div class="entry-head">
                <span></span>
                <span>this turn</span>
                <span>EV final</span>
              </div>
              <ol class="choice-list">
                {#each choices as c}
                  {@const entryIdx = c.entry ? ENTRY_NAME_TO_IDX[c.entry] : undefined}
                  {@const joker = entryIdx !== undefined && isJokerEnabled(entryIdx)}
                  <li>
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
                    <code>{c.ev.toFixed(2)}</code>
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
      </section>
    </div>
  {/if}
</main>

<style>
  :global(body) {
    background: #f7f6f2;
    margin: 0;
    color: #222;
  }
  main {
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
  h1 { margin: 0; font-size: 1.5rem; }
  .reset {
    background: transparent;
    border: 1px solid #bbb;
    border-radius: 4px;
    padding: 0.25rem 0.75rem;
    cursor: pointer;
  }

  .layout {
    display: grid;
    grid-template-columns: minmax(320px, 1fr) minmax(280px, 420px);
    gap: 1.5rem;
    align-items: start;
  }
  @media (max-width: 760px) {
    .layout { grid-template-columns: 1fr; }
  }

  .scorecard {
    background: #fff;
    border: 1px solid #d4cdb8;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .section-label {
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #7a6f4f;
    margin: 0.5rem 0 0.25rem;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  td {
    padding: 0.3rem 0.4rem;
    border-bottom: 1px solid #f1ecdc;
    font-size: 0.95rem;
  }
  td:nth-child(2) { text-align: right; font-variant-numeric: tabular-nums; }
  td.score-cell { width: 6rem; }
  tr.filled td:first-child { color: #999; text-decoration: line-through; }
  tr.best { background: #fff3c4; }
  tr.best td:first-child { font-weight: 600; }
  tr.computed td { color: #555; background: #faf7ec; font-size: 0.85rem; }
  .note { color: #999; font-size: 0.8rem; font-weight: normal; }
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
    border: 1px solid #ccc;
    border-radius: 3px;
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
    background: #fff;
  }
  .score-edit::placeholder { color: #bbb; font-style: italic; }
  .score-edit:focus { outline: 2px solid #c9bf9c; outline-offset: 0; }
  .score-edit.invalid { border-color: #b00020; background: #fdecef; color: #b00020; }
  .score-edit.invalid:focus { outline-color: #b00020; }
  .err { color: #b00020; font-size: 0.75rem; }
  tr.invalid { background: #fdf3f4; }
  .inline {
    display: inline-flex;
    gap: 0.4rem;
    align-items: center;
    font-size: 0.85rem;
    color: #555;
  }

  .totals {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 2px solid #d4cdb8;
  }
  .totals > div {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .totals span {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .totals strong {
    font-size: 1.5rem;
    font-variant-numeric: tabular-nums;
  }

  .play { display: flex; flex-direction: column; gap: 1rem; }

  .dice-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.5rem;
  }
  .die {
    aspect-ratio: 1;
    padding: 0;
    border: none;
    background: transparent;
    cursor: pointer;
  }
  .die svg { width: 100%; height: 100%; display: block; }
  .die svg rect {
    fill: #fff;
    stroke: #333;
    stroke-width: 3;
    transition: fill 0.15s;
  }
  .die svg circle { fill: #222; }
  .die.keep svg rect { fill: #d7d2c2; stroke: #6b6450; }
  .die:hover svg rect { stroke: #000; }

  .roll-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
  }
  .roll-row .label { color: #666; font-size: 0.9rem; margin-right: 0.25rem; }
  .seg {
    width: 2.5rem;
    height: 2rem;
    border: 1px solid #bbb;
    background: #fff;
    cursor: pointer;
    font-size: 1rem;
  }
  .seg:first-of-type { border-radius: 4px 0 0 4px; }
  .seg:last-of-type { border-radius: 0 4px 4px 0; }
  .seg.on { background: #333; color: #fff; border-color: #333; }

  .rec {
    background: #fff;
    border: 1px solid #d4cdb8;
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }
  .ev-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
    gap: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #eee;
  }
  .ev-summary > div { display: flex; flex-direction: column; gap: 0.1rem; }
  .ev-summary span {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .ev-summary strong { font-size: 1.35rem; font-variant-numeric: tabular-nums; }
  .entry-head {
    display: grid;
    grid-template-columns: 1fr 3rem 3.2rem;
    gap: 0.5rem;
    padding: 0.2rem 0.3rem;
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .entry-head > span:not(:first-child) { text-align: right; }
  .caption { color: #777; font-size: 0.8rem; margin: 0.25rem 0 0.5rem; }
  .choice-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .choice-list li {
    display: grid;
    grid-template-columns: 1fr 3rem 3.2rem;
    gap: 0.5rem;
    align-items: center;
    padding: 0.25rem 0.3rem;
    border-bottom: 1px dashed #eee;
  }
  .choice-list code { text-align: right; }
  .choice-label { display: inline-flex; gap: 0.3rem; align-items: center; flex-wrap: wrap; }
  .choice-label em { font-style: normal; color: #888; font-size: 0.8rem; }
  .mini-die {
    width: 1.25rem;
    height: 1.25rem;
    display: inline-block;
    vertical-align: middle;
  }
  .mini-die rect { fill: #fff; stroke: #333; stroke-width: 5; }
  .mini-die circle { fill: #222; }
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
    font-size: 0.85rem;
  }
  .error { color: #b00020; }
</style>
