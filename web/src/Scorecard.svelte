<script lang="ts">
  import { ENTRY_LABELS, HOW_TO_SCORE, PIP_POS } from './constants';

  export let rawInputs: string[];
  export let yahtzeeBonuses: number;
  export let errors: (string | null)[];
  export let scores: (number | null)[];
  export let scoreHighlightIdx: number;
  export let isJokerEnabled: (i: number) => boolean;
  export let maxYahtzeeBonuses: number;
  export let yahtzeeBonusPoints: number;
  export let yahtzeeBonusEligible: boolean;
  export let upperFilled: number;
  export let upperBonus: number;
  export let currentTotal: number;
  export let projectedFinal: string;
  export let turnText: string;
</script>

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
    <div><span>Projected final</span><strong>{projectedFinal}</strong></div>
    <div><span>Turn</span><strong>{turnText}</strong></div>
  </div>
</section>

<style>
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
  tr.invalid { background: var(--err-row-bg); }

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
  .inline {
    display: inline-flex;
    gap: 0.4rem;
    align-items: center;
    font-size: var(--text-sm);
    color: var(--fg-muted);
  }
  .note { color: var(--fg-dim); font-size: var(--text-xs); font-weight: normal; }

  .joker {
    display: inline-block;
    font-size: var(--text-xs);
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
</style>
