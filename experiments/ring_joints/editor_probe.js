// Harness only. `check_against_editor.py` prepends the evaluator it lifts out
// of sdf_editor.html, so nothing here may redeclare any of it.
const fs = require('fs');
const doc = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

function buildPlan(list, bodies) {
  const live = list.filter(n => n.on);
  const plan = [];
  for (const b of bodies) {
    if (b.on === false) continue;
    const own = live.filter(n => n.op === 'add' ? n.b === b.id : hits(n, b.id));
    if (own.some(n => n.op === 'add')) plan.push({ id: b.id, nodes: own });
  }
  return plan;
}

const plan = buildPlan(doc.nodes, doc.bodies);
console.error('bodies: ' + plan.map(p => p.id).join(','));

// A fixed LCG, so the Python side compares against the same points every run.
let rng = 12345;
const rnd = () => (rng = (rng * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const out = [];
for (let i = 0; i < 4000; i++) {
  const x = -25 + 50 * rnd(), y = -40 + 120 * rnd(), z = 5 + 45 * rnd();
  const row = [x, y, z];
  for (const g of plan) row.push(sceneSDF([g], x, y, z));
  out.push(row.join(' '));
}
console.log(out.join('\n'));
