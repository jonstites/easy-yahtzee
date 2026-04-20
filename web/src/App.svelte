<script lang="ts">
  import { onMount } from 'svelte';
  import init, { Solver } from 'yahtzee-wasm';

  const ENTRY_LABELS = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    'Three of a kind', 'Four of a kind', 'Full house',
    'Small straight', 'Large straight', 'Yahtzee', 'Chance',
  ] as const;

  let solver: Solver | null = null;
  let loading = true;
  let loadError: string | null = null;

  let entries: boolean[] = new Array(13).fill(false);
  let upperScoreRemaining = 63;
  let yahtzeeBonusEligible = false;
  let dice: number[] = [1, 1, 1, 1, 1];
  let roll: 1 | 2 | 3 = 1;

  type KeeperRec = { dice: number[]; ev: number };
  type EntryRec = { entry: string; ev: number };
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

  const ENTRY_NAME_TO_LABEL: Record<string, string> = {
    ones: 'Ones', twos: 'Twos', threes: 'Threes', fours: 'Fours', fives: 'Fives', sixes: 'Sixes',
    three_of_a_kind: 'Three of a kind', four_of_a_kind: 'Four of a kind',
    full_house: 'Full house', small_straight: 'Small straight',
    large_straight: 'Large straight', yahtzee: 'Yahtzee', chance: 'Chance',
  };
</script>

<main>
  <h1>easy-yahtzee</h1>

  {#if loading}
    <p>Loading solver…</p>
  {:else if loadError}
    <p class="error">Failed to load: {loadError}</p>
  {:else}
    <section class="panel">
      <h2>State</h2>
      <div class="grid">
        {#each ENTRY_LABELS as label, i}
          <label><input type="checkbox" bind:checked={entries[i]} /> {label}</label>
        {/each}
      </div>
      <div class="row">
        <label>
          Upper score remaining:
          <input type="number" min="0" max="63" bind:value={upperScoreRemaining} />
        </label>
        <label>
          <input type="checkbox" bind:checked={yahtzeeBonusEligible} />
          Yahtzee bonus eligible
        </label>
      </div>
    </section>

    <section class="panel">
      <h2>Dice</h2>
      <div class="row">
        {#each [0, 1, 2, 3, 4] as i}
          <label>
            Die {i + 1}
            <select bind:value={dice[i]}>
              {#each [1, 2, 3, 4, 5, 6] as v}
                <option value={v}>{v}</option>
              {/each}
            </select>
          </label>
        {/each}
      </div>
      <div class="row">
        <label>
          Roll:
          <select bind:value={roll}>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={3}>3</option>
          </select>
        </label>
      </div>
    </section>

    <section class="panel">
      <h2>Recommendation</h2>
      {#if recError}
        <p class="error">{recError}</p>
      {:else if rec}
        <p><strong>Total EV:</strong> {rec.value.toFixed(2)}</p>
        {#if rec.keepers}
          <h3>Best keepers (roll {roll})</h3>
          <ol>
            {#each rec.keepers.slice(0, 10) as k}
              <li>
                {k.dice.length === 0 ? '(keep nothing)' : k.dice.join(' ')}
                — <code>{k.ev.toFixed(4)}</code>
              </li>
            {/each}
          </ol>
        {:else if rec.entries}
          <h3>Best category to fill</h3>
          <ol>
            {#each rec.entries as e}
              <li>
                {ENTRY_NAME_TO_LABEL[e.entry] ?? e.entry}
                — <code>{e.ev.toFixed(4)}</code>
              </li>
            {/each}
          </ol>
        {/if}
      {/if}
    </section>
  {/if}
</main>

<style>
  main { max-width: 720px; margin: 2rem auto; padding: 0 1rem; font-family: system-ui, sans-serif; }
  h1 { margin-bottom: 1.5rem; }
  .panel { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 1rem; }
  .panel h2 { margin-top: 0; font-size: 1.1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem 1rem; margin-bottom: 0.5rem; }
  .row { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
  code { background: #f4f4f4; padding: 0 0.25rem; border-radius: 3px; }
  .error { color: #b00020; }
</style>
