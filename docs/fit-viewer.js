const panels = Array.from(document.querySelectorAll('figure[data-stage]'));
const tabs = document.createElement('div');
tabs.setAttribute('role', 'tablist');
tabs.setAttribute('aria-label', 'Fit stage');
panels[0].before(tabs);

function selectStage(active) {
  for (const [index, panel] of panels.entries()) {
    const selected = index === active;
    const button = tabs.children[index];
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
    panel.hidden = !selected;
  }
}

for (const [index, panel] of panels.entries()) {
  const button = document.createElement('button');
  button.textContent = panel.dataset.stage;
  button.id = 'tab-' + index;
  button.setAttribute('role', 'tab');
  button.setAttribute('aria-controls', panel.id);
  panel.setAttribute('role', 'tabpanel');
  panel.setAttribute('aria-labelledby', button.id);
  panel.tabIndex = 0;
  button.addEventListener('click', () => selectStage(index));
  button.addEventListener('keydown', event => {
    const count = panels.length;
    const target = { ArrowRight: (index + 1) % count, ArrowLeft: (index + count - 1) % count,
                     Home: 0, End: count - 1 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    selectStage(target);
    tabs.children[target].focus();
  });
  tabs.append(button);
}
selectStage(0);
