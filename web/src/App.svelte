<script lang="ts">
  import { onMount } from 'svelte';
  import init, { Solver, achievableScores, entryScore } from 'yahtzee-wasm';
  import { ENTRY_LABELS, PIP_POS, YAHTZEE_IDX } from './constants';
  import {
    cycleFace,
    formatAllowed,
    isYahtzee,
    keepMaskFor,
    maxYahtzeeBonuses as maxYahtzeeBonusesFor,
    parseScoreInput,
    randomDice,
    scoreContributionMask,
  } from './scoring';
  import { buildChoices, type Recommendation } from './recommendation';
  import Scorecard from './Scorecard.svelte';
  import HelpModal from './HelpModal.svelte';

  // `$state.raw` for the wasm-bound Solver: the binding needs to be reactive
  // (so the recommend `$effect` re-fires once the solver loads), but we don't
  // want Svelte to wrap the wasm-bindgen object in a deep proxy — its methods
  // depend on internal slots that proxies don't preserve.
  let solver = $state.raw<Solver | null>(null);
  let loading = $state(true);
  let loadError: string | null = $state(null);
  let showHelp = $state(false);

  let rawInputs: string[] = $state(new Array(13).fill(''));
  let yahtzeeBonuses = $state(0);

  type Roll = 1 | 2 | 3;
  const ROLLS: Roll[] = [1, 2, 3];

  type Snapshot = {
    rawInputs: string[];
    yahtzeeBonuses: number;
    dice: number[];
    kept: boolean[];
    roll: Roll;
  };
  let history: Snapshot[] = $state([]);

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

  let allowedByEntry: number[][] = $state([]);
  let allowedSets: Set<number>[] = $state([]);
  let allowedLabels: string[] = $state([]);
  let wasmReady = $state(false);

  const parsed = $derived(rawInputs.map((r, i) =>
    parseScoreInput(
      r,
      wasmReady ? { values: allowedSets[i], label: allowedLabels[i] } : null,
    ),
  ));
  const scores = $derived(parsed.map(p => p.value));
  const errors = $derived(parsed.map(p => p.error));
  let dice: number[] = $state([1, 1, 1, 1, 1]);
  let kept: boolean[] = $state([false, false, false, false, false]);
  let roll: Roll = $state(1);

  // Plain data shape, but `$state.raw` since we always replace the whole
  // object — no need for a deep proxy.
  let rec = $state.raw<Recommendation | null>(null);
  let recError: string | null = $state(null);

  onMount(async () => {
    // Run wasm init and the scores.bin fetch concurrently — neither depends
    // on the other until we want to construct the Solver from the bytes.
    try {
      const [, bytes] = await Promise.all([
        init(),
        fetch('/scores.bin').then(async resp => {
          if (!resp.ok) throw new Error(`fetch scores.bin: ${resp.status}`);
          return new Uint8Array(await resp.arrayBuffer());
        }),
      ]);
      allowedByEntry = Array.from({ length: 13 }, (_, i) => Array.from(achievableScores(i)));
      allowedSets = allowedByEntry.map(vs => new Set(vs));
      allowedLabels = allowedByEntry.map(formatAllowed);
      wasmReady = true;
      solver = new Solver(bytes);
      dice = randomDice();
    } catch (e) {
      loadError = String(e);
    } finally {
      loading = false;
    }
  });

  // Which scorecard rows are filled (i.e. unavailable to score into again).
  // Distinct from `rec.entries`, which is the list of scoring recommendations.
  const filledMask = $derived(scores.map(s => s !== null));
  const upperFilled = $derived(scores.slice(0, 6).reduce<number>((a, b) => a + (b ?? 0), 0));
  const lowerFilled = $derived(scores.slice(6).reduce<number>((a, b) => a + (b ?? 0), 0));
  const upperScoreRemaining = $derived(Math.max(0, 63 - upperFilled));
  const upperBonus = $derived(upperFilled >= 63 ? 35 : 0);
  const turnsPlayed = $derived(scores.filter(s => s !== null).length);
  const gameOver = $derived(turnsPlayed === 13);
  const maxYahtzeeBonuses = $derived(maxYahtzeeBonusesFor(scores[YAHTZEE_IDX], turnsPlayed));
  $effect(() => {
    const clamped = Math.max(0, Math.min(yahtzeeBonuses | 0, maxYahtzeeBonuses));
    if (clamped !== yahtzeeBonuses) yahtzeeBonuses = clamped;
  });
  const yahtzeeBonusPoints = $derived(yahtzeeBonuses * 100);
  const currentTotal = $derived(upperFilled + upperBonus + lowerFilled + yahtzeeBonusPoints);
  const yahtzeeBonusEligible = $derived(scores[YAHTZEE_IDX] === 50);

  const stateInput = $derived({
    entries: filledMask,
    yahtzee_bonus_eligible: yahtzeeBonusEligible,
    upper_score_remaining: upperScoreRemaining,
  });

  $effect(() => {
    if (!solver) return;
    try {
      rec = solver.recommend(stateInput, new Uint8Array(dice), roll) as Recommendation;
      recError = null;
    } catch (e) {
      rec = null;
      recError = String(e);
    }
  });

  function cycleDie(i: number, delta: number) {
    pushHistory();
    dice[i] = cycleFace(dice[i], delta);
  }

  // The Joker rule fires when the dice are a Yahtzee, the Yahtzee box is
  // already filled (any value), AND the matching upper box is already filled.
  // Lower-row categories then accept fixed joker scores (full house = 25,
  // small straight = 30, large straight = 40).
  const jokerActive = $derived.by(() => {
    const c = [0, 0, 0, 0, 0, 0];
    for (const v of dice) c[v - 1]++;
    const face = c.findIndex(x => x === 5) + 1;
    if (face === 0) return false;
    return scores[YAHTZEE_IDX] !== null && scores[face - 1] !== null;
  });

  // Highlight the joker-rule indicator only on lower-row "shaped" categories
  // (full house, small straight, large straight) — those are the ones that
  // accept the special joker score.
  function isJokerEnabled(i: number): boolean {
    return jokerActive && (i === 8 || i === 9 || i === 10);
  }

  const choices = $derived(buildChoices(rec));

  const hasJokerChoice = $derived(choices.some(
    c => c.kind === 'score' && isJokerEnabled(c.entryIdx)
  ));

  function toggleKeep(i: number) {
    kept[i] = !kept[i];
  }

  function randomizeDice() {
    if (gameOver) return;
    pushHistory();
    dice = randomDice();
    kept.fill(false);
  }

  // Re-roll any die not marked kept and advance the roll counter. Caller is
  // responsible for the gameOver / roll / all-kept guards (see handleRoll and
  // applyKeepAndRoll) so callers can do pushHistory only when an action is
  // actually about to happen.
  function rollDice() {
    for (let i = 0; i < 5; i++) {
      if (!kept[i]) dice[i] = 1 + Math.floor(Math.random() * 6);
    }
    kept.fill(false);
    roll = (roll + 1) as Roll;
  }

  function handleRoll() {
    if (gameOver || roll === 3 || kept.every(k => k)) return;
    pushHistory();
    rollDice();
  }

  function applyKeepAndRoll(faces: number[]) {
    if (gameOver || roll === 3) return;
    const mask = keepMaskFor(faces, dice);
    if (mask.every(k => k)) return; // shouldn't happen — wasm omits the all-5 keeper
    pushHistory();
    kept = mask;
    rollDice();
  }

  function applyRerollAll() {
    if (gameOver || roll === 3) return;
    pushHistory();
    kept.fill(false);
    rollDice();
  }

  function applyTop() {
    const c = choices[0];
    if (!c) return;
    if (c.kind === 'score') applyScore(c.entryIdx);
    else if (c.kind === 'keep') applyKeepAndRoll(c.dice);
    else applyRerollAll();
  }

  function onKey(e: KeyboardEvent) {
    if (showHelp) {
      if (e.key === 'Escape' || e.key === '?') {
        e.preventDefault();
        showHelp = false;
      }
      return;
    }
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === ' ') { e.preventDefault(); handleRoll(); }
    else if (e.key === 'Enter') { e.preventDefault(); applyTop(); }
    else if (e.key === 'r' || e.key === 'R') { e.preventDefault(); randomizeDice(); }
    else if (e.key === 'u' || e.key === 'U') { e.preventDefault(); undo(); }
    else if (e.key === '?') { e.preventDefault(); showHelp = true; }
  }

  function applyScore(idx: number) {
    if (gameOver) return;
    if (idx < 0 || idx >= 13) return;
    pushHistory();
    const val = entryScore(stateInput, new Uint8Array(dice), idx);
    rawInputs[idx] = String(val);
    // Auto-track Yahtzee bonuses: a fresh Yahtzee scored anywhere except the
    // Yahtzee box itself, while already eligible (= Yahtzee box holds 50),
    // earns +100.
    if (isYahtzee(dice) && yahtzeeBonusEligible && idx !== YAHTZEE_IDX) {
      yahtzeeBonuses = yahtzeeBonuses + 1;
    }
    // If this fill completes the game, leave the dice alone so the
    // game-over screen can render against the final state.
    const otherFilled = filledMask.reduce((a, b, i) => a + (i !== idx && b ? 1 : 0), 0);
    if (otherFilled + 1 >= 13) {
      kept.fill(false);
      return;
    }
    dice = randomDice();
    kept.fill(false);
    roll = 1;
  }

  // Mask of dice the top recommendation cares about. Drives the fade on
  // non-recommended dice in the dice row. For 'reroll' nothing is highlighted
  // (we want the user to feel free to re-roll everything).
  const recKeepMask = $derived.by(() => {
    const top = choices[0];
    if (!top) return [false, false, false, false, false];
    if (top.kind === 'keep') return keepMaskFor(top.dice, dice);
    if (top.kind === 'score') return scoreContributionMask(top.entryIdx, dice);
    return [false, false, false, false, false];
  });
  const anyRecommended = $derived(recKeepMask.some(Boolean));

  // Which scorecard row to highlight as the top suggestion (-1 = none).
  const scoreHighlightIdx = $derived(choices[0]?.kind === 'score' ? choices[0].entryIdx : -1);

  function setRoll(r: Roll) {
    if (r === roll) return;
    pushHistory();
    roll = r;
  }

  function resetGame() {
    pushHistory();
    rawInputs.fill('');
    yahtzeeBonuses = 0;
    dice = randomDice();
    kept.fill(false);
    roll = 1;
  }

  const projectedFinal = $derived(rec ? (currentTotal + rec.value).toFixed(1) : '—');
  const turnText = $derived(`${Math.min(turnsPlayed + (gameOver ? 0 : 1), 13)} / 13`);
</script>

<svelte:window onkeydown={onKey} />

<main>
  <header>
    <h1>easy-yahtzee</h1>
    <div class="header-actions">
      <button class="reset" onclick={() => (showHelp = true)} title="Help (?)">?</button>
      <button class="reset" disabled={history.length === 0} onclick={undo} title="Undo (U)">Undo</button>
      <button class="reset" onclick={resetGame}>Reset</button>
    </div>
  </header>

  {#if loading}
    <p>Loading solver…</p>
  {:else if loadError}
    <p class="error">Failed to load: {loadError}</p>
  {:else}
    <div class="layout">
      <Scorecard
        bind:rawInputs
        bind:yahtzeeBonuses
        {errors}
        {scores}
        {scoreHighlightIdx}
        {isJokerEnabled}
        {maxYahtzeeBonuses}
        {yahtzeeBonusPoints}
        {yahtzeeBonusEligible}
        {upperFilled}
        {upperBonus}
        {currentTotal}
        {projectedFinal}
        {turnText}
      />

      <section class="play">
        {#if gameOver}
          <div class="game-over">
            <h2>Game over</h2>
            <p>Final score <strong>{currentTotal}</strong></p>
            <button class="roll-btn primary" onclick={resetGame}>New game</button>
          </div>
        {:else}
        <div class="dice-row">
          {#each dice as face, i}
            <div class="die-cell" class:faded={anyRecommended && !recKeepMask[i]}>
              <button
                class="die"
                onclick={() => cycleDie(i, 1)}
                oncontextmenu={(e) => { e.preventDefault(); cycleDie(i, -1); }}
                onwheel={(e) => { e.preventDefault(); cycleDie(i, e.deltaY > 0 ? 1 : -1); }}
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
                <input type="checkbox" checked={kept[i]} onchange={() => toggleKeep(i)} />
                keep
              </label>
            </div>
          {/each}
        </div>

        <div class="roll-row">
          {#each ROLLS as r}
            <button class="seg" class:on={roll === r} onclick={() => setRoll(r)}>
              {r}
            </button>
          {/each}
          <button
            class="roll-btn"
            disabled={roll === 3 || kept.every(k => k)}
            onclick={handleRoll}
            title={roll === 3 ? 'No rolls left' : (kept.every(k => k) ? 'All dice kept' : 'Re-roll dice not marked keep')}
          >
            Roll
          </button>
          <button
            class="roll-btn"
            onclick={randomizeDice}
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
                  <li class:top={ci === 0}>
                    <span class="choice-label">
                      {#if c.kind === 'score'}
                        <em>score in</em>
                        <span class="chip entry">{ENTRY_LABELS[c.entryIdx]}</span>
                        {#if isJokerEnabled(c.entryIdx)}
                          <span class="joker-dot" aria-hidden="true">★</span>
                        {/if}
                      {:else if c.kind === 'reroll'}
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
                    {#if c.kind === 'score'}
                      {@const idx = c.entryIdx}
                      <button class="apply" class:primary={ci === 0} onclick={() => applyScore(idx)}>Apply</button>
                    {:else if c.kind === 'keep'}
                      {@const faces = c.dice}
                      <button class="apply" class:primary={ci === 0} disabled={roll === 3} onclick={() => applyKeepAndRoll(faces)}>Apply</button>
                    {:else}
                      <button class="apply" class:primary={ci === 0} disabled={roll === 3} onclick={applyRerollAll}>Apply</button>
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

  <HelpModal bind:open={showHelp} />
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

  .joker-dot { color: #c39509; font-size: var(--text-xs); margin-left: 0.2rem; }

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
