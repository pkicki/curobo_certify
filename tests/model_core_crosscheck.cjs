'use strict';
// Reads a JSON case from argv[2] and prints JS exact Bernstein residual coefficients.
const fs = require('fs');
const M = require('../web/model_core.js');
const P = require('../web/proof_core.js');
const c = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const model = M.build(c.path, c.table);
const out = c.queries.map(([si, fam, joint]) => {
  const r = M.residual(model, si, fam, joint, c.path.limits);
  return M.bernstein(r.power).map(P.repr);
});
process.stdout.write(JSON.stringify({spans: model.spans.length, coefficients: out}));
