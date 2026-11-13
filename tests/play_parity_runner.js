// Prints what docs/play/calc.js computes for every fixture in
// docs/play/data/parity.json, as one JSON document on stdout.
// tests/test_play_parity.py runs this under node and compares each number
// with the Python output stored in the same file. By hand:
//   node tests/play_parity_runner.js
"use strict";

var fs = require("fs");
var path = require("path");

var playDir = path.join(__dirname, "..", "docs", "play");
var calc = require(path.join(playDir, "calc.js"));

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(playDir, "data", name), "utf8"));
}

var tiers = readJson("tiers.json");
var parity = readJson("parity.json");

var fixtures = [];
for (var i = 0; i < parity.fixtures.length; i++) {
  var fixture = parity.fixtures[i];
  var companies = fixture.companies;
  // null means the shipped service data; copy it so no fixture can touch the next one's.
  if (companies === null) companies = JSON.parse(JSON.stringify(tiers.companies));
  var actual = calc.evaluate(fixture.inputs, companies);
  var rankings = {};
  var sampleIndexes = Object.keys(fixture.expected_rank_at_rate);
  for (var s = 0; s < sampleIndexes.length; s++) {
    rankings[sampleIndexes[s]] = calc.rankAtRate(actual.rate_grid, Number(sampleIndexes[s]));
  }
  fixtures.push({ name: fixture.name, actual: actual, actual_rank_at_rate: rankings });
}

var weeklyFromAnnual = [];
var annualSamples = parity.helpers.weekly_pct_from_annual;
for (var a = 0; a < annualSamples.length; a++) {
  weeklyFromAnnual.push(calc.weeklyPctFromAnnual(annualSamples[a].annual_pct));
}
var annualFromWeekly = [];
var weeklySamples = parity.helpers.annual_pct_from_weekly;
for (var w = 0; w < weeklySamples.length; w++) {
  annualFromWeekly.push(calc.annualPctFromWeekly(weeklySamples[w].weekly_pct));
}

process.stdout.write(JSON.stringify({
  fixtures: fixtures,
  helpers: { weekly_pct_from_annual: weeklyFromAnnual, annual_pct_from_weekly: annualFromWeekly }
}));
