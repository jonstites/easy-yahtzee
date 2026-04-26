<script lang="ts">
  let { open = $bindable(false) } = $props();
  function close() { open = false; }
</script>

{#if open}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="modal-backdrop" onclick={close} role="presentation">
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_noninteractive_element_interactions -->
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog" tabindex="-1" aria-modal="true" aria-labelledby="help-title">
      <div class="modal-head">
        <h2 id="help-title">Help</h2>
        <button class="modal-close" onclick={close} aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <h3>Keyboard shortcuts</h3>
        <ul class="shortcuts">
          <li><kbd>Space</kbd><span>Roll non-kept dice</span></li>
          <li><kbd>Enter</kbd><span>Apply the top recommendation</span></li>
          <li><kbd>R</kbd><span>Randomize all dice (no advance)</span></li>
          <li><kbd>U</kbd><span>Undo</span></li>
          <li><kbd>?</kbd><span>Show / hide this help</span></li>
          <li><kbd>Esc</kbd><span>Close this help</span></li>
        </ul>
        <h3>Tips</h3>
        <ul>
          <li>Click a die to cycle the face — useful for querying the solver about a specific real-world roll. Right-click or scroll to go backward.</li>
          <li>Recommended dice stay full-contrast; non-recommended dice fade.</li>
          <li>The 1 / 2 / 3 segmented control sets which roll you're on, so you can ask the solver what to do mid-turn from a physical game.</li>
          <li>"Apply" on a "keep" choice marks those dice and rolls. "Apply" on a "score" choice writes to the scorecard and ends the turn.</li>
          <li>Undo covers rolls, applies, randomize, reset, dice-cycling, and roll-segment changes — but not typing into the scorecard.</li>
        </ul>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 1rem;
  }
  .modal {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    max-width: 520px;
    width: 100%;
    max-height: 85vh;
    overflow-y: auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  }
  .modal-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
  }
  .modal-head h2 { margin: 0; font-size: var(--text-lg); }
  .modal-close {
    background: transparent;
    border: none;
    font-size: 1.5rem;
    line-height: 1;
    cursor: pointer;
    color: var(--fg-muted);
    padding: 0 0.25rem;
  }
  .modal-close:hover { color: var(--fg); }
  .modal-body { padding: 1rem 1.25rem 1.5rem; }
  .modal-body h3 {
    margin: 1rem 0 0.5rem;
    font-size: var(--text-md);
    text-transform: uppercase;
    letter-spacing: var(--eyebrow-spacing);
    color: var(--fg-warm);
  }
  .modal-body h3:first-child { margin-top: 0; }
  .modal-body ul { margin: 0; padding-left: 1.25rem; }
  .modal-body li { margin: 0.3rem 0; }
  .shortcuts { list-style: none; padding-left: 0; }
  .shortcuts li {
    display: grid;
    grid-template-columns: 4.5rem 1fr;
    gap: 0.75rem;
    align-items: baseline;
  }
  kbd {
    display: inline-block;
    min-width: 1.5rem;
    text-align: center;
    padding: 0.1rem 0.4rem;
    border: 1px solid var(--fg-faint);
    border-bottom-width: 2px;
    border-radius: 3px;
    background: var(--bg-page);
    font-family: ui-monospace, monospace;
    font-size: var(--text-xs);
    color: var(--fg);
  }
</style>
