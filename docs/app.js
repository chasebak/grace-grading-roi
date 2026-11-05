// Hand GRACE a Budget: a viewer for docs/data/*.json.
// Why this file computes nothing: every number is written by
// `python -m grace.viz build` from the tested Experiment 4 functions and
// checked in tests/test_viz.py. This script only looks values up by budget
// and rule and formats them, so the page cannot drift from the README.
"use strict";

(function () {
  var state = { summary: null, grid: null, index: 0, batchRule: "grace", chart: null };

  function byId(id) { return document.getElementById(id); }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // ---- formatting only -------------------------------------------------
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function money(v) {
    if (!isNum(v)) return "n/a";
    var text = "$" + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 });
    return v < 0 ? "-" + text : text;
  }
  function ratio(v) { return isNum(v) ? v.toFixed(3) : "n/a"; }
  function fixed2(v) { return isNum(v) ? v.toFixed(2) : "n/a"; }
  function signed(v, digits) {
    if (!isNum(v)) return "n/a";
    return (v >= 0 ? "+" : "") + v.toFixed(digits);
  }
  function share(v) { return isNum(v) ? (v * 100).toFixed(1) + "%" : "n/a"; }
  function signedShare(v) { return isNum(v) ? signed(v * 100, 1) + "%" : "n/a"; }
  function count(v) {
    if (!isNum(v)) return "n/a";
    return Number.isInteger(v) ? String(v) : v.toFixed(1);
  }
  // The "(mean)" label goes by rule: the random row is a mean over seeded
  // shuffles even when that mean happens to be a whole number.
  function picksText(entry, rule) {
    var text = count(entry.planted_picks) + "/" + count(entry.n_picks);
    if (rule === "random") text += " (mean)";
    return text;
  }

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) {
      var keys = Object.keys(attrs);
      for (var i = 0; i < keys.length; i++) node.setAttribute(keys[i], attrs[keys[i]]);
    }
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ---- lookups -----------------------------------------------------------
  function budgets() { return state.grid.budgets; }
  function budget() { return budgets()[state.index]; }
  function cell(rule, atBudget) {
    var b = atBudget === undefined ? budget() : atBudget;
    return state.grid.cells[String(b)][rule];
  }
  function label(rule) { return state.summary.rule_labels[rule]; }
  function shortLabel(rule) { return state.summary.rule_short_labels[rule]; }
  function lowerFirst(text) { return text.charAt(0).toLowerCase() + text.slice(1); }

  // ---- static text -------------------------------------------------------
  function fillWhyBoxes() {
    var boxes = document.querySelectorAll(".why[data-why]");
    for (var i = 0; i < boxes.length; i++) {
      var entry = state.summary.why[boxes[i].getAttribute("data-why")];
      boxes[i].textContent = "";
      boxes[i].appendChild(el("b", null, "Why this metric"));
      boxes[i].appendChild(el("p", null, entry.question));
      boxes[i].appendChild(el("p", null, "Kind: " + entry.kind));
      boxes[i].appendChild(el("p", null, "Could lie if: " + entry.lie));
    }
  }

  function fillProvenance() {
    var pills = document.querySelectorAll(".provenance[data-prov]");
    for (var i = 0; i < pills.length; i++) {
      pills[i].textContent = state.summary.provenance[pills[i].getAttribute("data-prov")];
    }
  }

  function fillHeadline() {
    var s = state.summary;
    var h = s.headline;
    byId("headline-label").textContent =
      "Oracle value captured at a " + money(s.default_budget) + " budget (" + s.settings.tier + " tier)";
    byId("headline-value").textContent = "GRACE " + ratio(h.grace.oracle_value_captured);
    byId("headline-note").textContent =
      "against " + ratio(h.heuristic.oracle_value_captured) + " for " + lowerFirst(label("heuristic")) +
      " and " + ratio(h.random.oracle_value_captured) + " for random liquid cards; the oracle is " +
      ratio(h.oracle.oracle_value_captured) + ".";
    var f = s.sweep_facts;
    byId("headline-sweep").textContent =
      "Across the sweep (" + money(f.min_budget) + " to " + money(f.max_budget) + "), GRACE ends with more true NPV than the heuristic at " +
      f.grace_beats_heuristic + " of " + f.n_budgets + " solved budgets, and its planted-error share sits below the liquid universe's base rate at " +
      f.grace_leakage_below_base_rate + " of " + f.n_budgets + ".";
    byId("a-one-note").textContent =
      label("gated_npv_per_dollar") + " (a = 1) has more true NPV than GRACE (a = 0.5) at " +
      f.npv_per_dollar_beats_grace + " of " + f.n_budgets + " solved budgets, and captures " +
      ratio(h.gated_npv_per_dollar.oracle_value_captured) + " against GRACE's " + ratio(h.grace.oracle_value_captured) +
      " at " + money(s.default_budget) + ". The shipped a stays 0.5 (DECISIONS.md).";
    byId("spread-note").textContent = s.spread_band;
    byId("footer-text").textContent = s.footer;
  }

  // ---- slider ------------------------------------------------------------
  function buildSlider() {
    var slider = byId("budget");
    var list = budgets();
    slider.max = String(list.length - 1);
    var ticks = byId("ticks");
    ticks.textContent = "";
    var labelled = state.summary.settings.tick_label_budgets;
    for (var i = 0; i < list.length; i++) {
      var stop = el("span", { "class": "stop", "data-index": String(i) });
      if (i === 0) stop.classList.add("first");
      if (i === list.length - 1) stop.classList.add("last");
      stop.style.left = (i / (list.length - 1)) * 100 + "%";
      stop.appendChild(el("span", { "class": "mark" }));
      if (labelled.indexOf(list[i]) !== -1) stop.appendChild(el("span", { "class": "tick-label" }, shortMoney(list[i])));
      ticks.appendChild(stop);
    }
    slider.addEventListener("input", function () { select(Number(slider.value)); });
    declutterTicks();
    window.addEventListener("resize", declutterTicks);
  }

  // At phone width the $250 and $1k labels sit about 30px apart and can
  // overlap. Keep the end labels; hide (not remove) any inner label that
  // would touch the label before it or the last one. Every stop keeps its mark.
  var TICK_GAP_PX = 4;
  function declutterTicks() {
    var labels = document.querySelectorAll("#ticks .tick-label");
    var n = labels.length;
    for (var i = 0; i < n; i++) labels[i].style.visibility = "";
    if (n < 3) return;
    var lastLeft = labels[n - 1].getBoundingClientRect().left;
    var shownRight = labels[0].getBoundingClientRect().right;
    for (var j = 1; j < n - 1; j++) {
      var box = labels[j].getBoundingClientRect();
      if (box.left < shownRight + TICK_GAP_PX || box.right + TICK_GAP_PX > lastLeft) {
        labels[j].style.visibility = "hidden";
      } else {
        shownRight = box.right;
      }
    }
  }

  function shortMoney(v) { return v >= 1000 ? "$" + v / 1000 + "k" : "$" + v; }

  function indexOfBudget(value) {
    var list = budgets();
    for (var i = 0; i < list.length; i++) if (list[i] === value) return i;
    return -1;
  }

  function select(index) {
    state.index = index;
    var slider = byId("budget");
    slider.value = String(index);
    slider.setAttribute("aria-valuetext", money(budget()));
    byId("budget-now").textContent = money(budget());
    var spans = document.querySelectorAll("[data-budget-now]");
    for (var i = 0; i < spans.length; i++) spans[i].textContent = money(budget());
    var stops = document.querySelectorAll("#ticks .stop");
    for (var j = 0; j < stops.length; j++) stops[j].classList.toggle("on", j === index);
    try { history.replaceState(null, "", "#budget=" + budget()); } catch (e) { /* file:// */ }
    renderTiles();
    renderRulesTable();
    renderBatch();
    markFallbackRow();
    if (state.chart) state.chart.update("none");
  }

  // ---- tiles -------------------------------------------------------------
  function tile(labelText, valueText, noteText) {
    var t = el("div", { "class": "tile" });
    t.appendChild(el("div", { "class": "label" }, labelText));
    t.appendChild(el("div", { "class": "value" }, valueText));
    t.appendChild(el("div", { "class": "note" }, noteText));
    return t;
  }

  function renderTiles() {
    var g = cell("grace"), hz = cell("heuristic"), r = cell("random"), o = cell("oracle");
    var u = state.summary.universe;
    var box = byId("tiles");
    box.textContent = "";
    box.appendChild(tile("Oracle value captured", "GRACE " + ratio(g.oracle_value_captured),
      "heuristic " + ratio(hz.oracle_value_captured) + " · random " + ratio(r.oracle_value_captured) + " · oracle " + ratio(o.oracle_value_captured)));
    box.appendChild(tile("Regret (oracle $ − rule $)", "GRACE " + money(g.regret),
      "heuristic " + money(hz.regret) + " · random " + money(r.regret)));
    box.appendChild(tile("True NPV per budget dollar", "GRACE " + fixed2(g.npv_per_budget_dollar),
      "heuristic " + fixed2(hz.npv_per_budget_dollar) + " · random " + fixed2(r.npv_per_budget_dollar) + " · oracle " + fixed2(o.npv_per_budget_dollar)));
    box.appendChild(tile("Pick optimism (predicted − true)", "GRACE p10 " + signed(g.pick_optimism_p10, 3),
      "batch $ GRACE " + signedShare(g.pick_optimism_share) + " · heuristic p10 " + signed(hz.pick_optimism_p10, 3) + ", batch $ " + signedShare(hz.pick_optimism_share)));
    box.appendChild(tile("Planted-error picks", "GRACE " + picksText(g, "grace"),
      share(g.leakage_capital_share) + " of capital · heuristic " + picksText(hz, "heuristic") + ", " + share(hz.leakage_capital_share) +
      " of capital · base rate " + u.n_planted + "/" + u.n_liquid));
  }

  // ---- every rule table --------------------------------------------------
  function headerRow(table, names, numericFrom) {
    var thead = el("thead");
    var tr = el("tr");
    for (var i = 0; i < names.length; i++) {
      tr.appendChild(el("th", i >= numericFrom ? { "class": "num", scope: "col" } : { scope: "col" }, names[i]));
    }
    thead.appendChild(tr);
    table.appendChild(thead);
  }

  function row(values, numericFrom, rowClass) {
    var tr = el("tr", rowClass ? { "class": rowClass } : null);
    for (var i = 0; i < values.length; i++) {
      tr.appendChild(el("td", i >= numericFrom ? { "class": "num" } : null, values[i]));
    }
    return tr;
  }

  function renderRulesTable() {
    var table = byId("rules-table");
    table.textContent = "";
    headerRow(table, ["rule", "oracle value captured", "true NPV", "regret", "NPV per budget $", "planted / picks", "spent"], 1);
    var tbody = el("tbody");
    var rules = state.summary.table_rules;
    for (var i = 0; i < rules.length; i++) {
      var c = cell(rules[i]);
      tbody.appendChild(row([label(rules[i]), ratio(c.oracle_value_captured), money(c.true_npv), money(c.regret),
        fixed2(c.npv_per_budget_dollar), picksText(c, rules[i]), money(c.spent)], 1, rules[i] === "grace" ? "chosen" : ""));
    }
    table.appendChild(tbody);
    var o = cell("oracle");
    byId("bracket-note").textContent =
      "Oracle bracket at this budget: feasible batch " + money(o.true_npv) + ", upper bound " + money(o.upper_bound) +
      " (width " + money(o.bracket_width) + "). Random is the mean over " + state.summary.settings.random_draws + " seeded shuffles.";
  }

  // ---- batch table -------------------------------------------------------
  function buildBatchButtons() {
    var group = byId("batch-rules");
    var rules = state.summary.batch_rules;
    for (var i = 0; i < rules.length; i++) {
      var button = el("button", { type: "button", "data-rule": rules[i], "aria-pressed": String(rules[i] === state.batchRule) },
        shortLabel(rules[i]));
      button.title = label(rules[i]);
      button.addEventListener("click", onBatchButton);
      group.appendChild(button);
    }
  }

  function onBatchButton(event) {
    state.batchRule = event.currentTarget.getAttribute("data-rule");
    var buttons = byId("batch-rules").querySelectorAll("button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", String(buttons[i].getAttribute("data-rule") === state.batchRule));
    }
    renderBatch();
  }

  function renderBatch() {
    var rule = state.batchRule;
    var c = cell(rule);
    var picks = state.grid.picks[String(budget())][rule];
    byId("batch-summary").textContent =
      label(rule) + " buys " + count(c.n_picks) + " cards for " + money(c.spent) + " of the " + money(budget()) +
      " budget: predicted NPV " + money(c.predicted_npv) + ", true NPV " + money(c.true_npv) + "; planted errors " +
      count(c.planted_picks) + " of " + count(c.n_picks) + " (red rows). Rows are in buy order.";
    var table = byId("batch-table");
    table.textContent = "";
    headerRow(table, ["#", "card", "what it is", "predicted NPV", "true NPV", "paid", "predicted p10", "true p10"], 3);
    var tbody = el("tbody");
    for (var i = 0; i < picks.length; i++) {
      var card = state.grid.cards[picks[i]];
      tbody.appendChild(row([String(i + 1), picks[i], state.summary.label_text[card.label], money(card.predicted_npv),
        money(card.true_npv), money(card.capital_paid), card.predicted_p10.toFixed(3), card.true_p10.toFixed(3)],
        3, card.planted ? "planted" : ""));
    }
    table.appendChild(tbody);
  }

  // ---- fallback table ----------------------------------------------------
  function renderFallbackTable() {
    var table = byId("fallback-table");
    table.textContent = "";
    var rules = state.summary.chart_rules;
    var names = ["budget"];
    for (var i = 0; i < rules.length; i++) names.push(shortLabel(rules[i]) + ": NPV per budget $");
    names.push("GRACE: oracle value captured");
    headerRow(table, names, 1);
    var tbody = el("tbody");
    var list = budgets();
    for (var j = 0; j < list.length; j++) {
      var values = [money(list[j])];
      for (var k = 0; k < rules.length; k++) values.push(fixed2(cell(rules[k], list[j]).npv_per_budget_dollar));
      values.push(ratio(cell("grace", list[j]).oracle_value_captured));
      var tr = row(values, 1, "");
      tr.setAttribute("data-index", String(j));
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
  }

  function markFallbackRow() {
    var rows = byId("fallback-table").querySelectorAll("tbody tr");
    for (var i = 0; i < rows.length; i++) rows[i].classList.toggle("chosen", i === state.index);
  }

  // ---- chart -------------------------------------------------------------
  var STYLE = {
    oracle: { color: "--series-3", dash: [6, 4], width: 2 },
    grace: { color: "--accent", dash: [], width: 3 },
    heuristic: { color: "--series-2", dash: [], width: 2 },
    random: { color: "--baseline", dash: [2, 3], width: 2 }
  };

  // Inline cursor (~20 lines, not a library plugin): a vertical line at the slider budget.
  var cursorPlugin = {
    id: "budgetCursor",
    afterDatasetsDraw: function (chart) {
      var x = chart.scales.x.getPixelForValue(budget());
      var area = chart.chartArea;
      var ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = cssVar("--text-muted");
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, area.bottom);
      ctx.stroke();
      ctx.restore();
    }
  };

  function buildChart() {
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    var rules = state.summary.chart_rules;
    var list = budgets();
    var datasets = [];
    for (var i = 0; i < rules.length; i++) {
      var points = [];
      for (var j = 0; j < list.length; j++) points.push({ x: list[j], y: cell(rules[i], list[j]).npv_per_budget_dollar });
      var style = STYLE[rules[i]];
      var color = cssVar(style.color);
      datasets.push({
        label: shortLabel(rules[i]), data: points, borderColor: color, backgroundColor: color,
        borderDash: style.dash, borderWidth: style.width, pointRadius: 2, pointHoverRadius: 5, tension: 0
      });
    }
    var text = cssVar("--text-muted");
    var grid = cssVar("--border");
    var tickBudgets = state.summary.settings.tick_label_budgets;
    state.chart = new window.Chart(byId("chart"), {
      type: "line",
      data: { datasets: datasets },
      plugins: [cursorPlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            type: "logarithmic", min: list[0], max: list[list.length - 1],
            title: { display: true, text: "budget (log scale)", color: text },
            ticks: { color: text, callback: function (v) { return shortMoney(v); } },
            afterBuildTicks: function (axis) {
              var ticks = [];
              for (var k = 0; k < tickBudgets.length; k++) ticks.push({ value: tickBudgets[k] });
              axis.ticks = ticks;
            },
            grid: { color: grid }
          },
          y: {
            title: { display: true, text: "true NPV per budget $", color: text },
            ticks: { color: text }, grid: { color: grid }
          }
        },
        plugins: {
          legend: { labels: { color: cssVar("--text"), boxWidth: 14 } },
          tooltip: {
            callbacks: {
              title: function (items) { return items.length ? money(items[0].parsed.x) + " budget" : ""; },
              label: function (item) { return item.dataset.label + ": " + fixed2(item.parsed.y); }
            }
          }
        },
        onClick: function (event, elements) {
          if (elements.length) select(elements[0].index);  // points are in budget order
        }
      }
    });
  }

  function showFallback() {
    byId("chart-box").hidden = true;
    byId("chart-fallback").hidden = false;
    byId("fallback-details").open = true;
  }

  // ---- start -------------------------------------------------------------
  function start(summary, grid) {
    state.summary = summary;
    state.grid = grid;
    fillWhyBoxes();
    fillProvenance();
    fillHeadline();
    buildSlider();
    buildBatchButtons();
    renderFallbackTable();
    var wanted = indexOfBudget(summary.default_budget);
    var match = /budget=(\d+)/.exec(location.hash);
    if (match && indexOfBudget(Number(match[1])) !== -1) wanted = indexOfBudget(Number(match[1]));
    if (window.Chart) {
      buildChart();
      var media = window.matchMedia("(prefers-color-scheme: dark)");
      var rebuild = function () { buildChart(); };
      if (media.addEventListener) media.addEventListener("change", rebuild); else media.addListener(rebuild);
    } else {
      showFallback();
    }
    select(wanted);
    // Why: a shared link or the back button changes only the hash, which
    // does not reload the page; follow it so the URL and the slider agree.
    window.addEventListener("hashchange", function () {
      var changed = /budget=(\d+)/.exec(location.hash);
      if (!changed) return;
      var index = indexOfBudget(Number(changed[1]));
      if (index !== -1 && index !== state.index) select(index);
    });
  }

  function getJson(path) {
    return fetch(path).then(function (response) {
      if (!response.ok) throw new Error(path + " " + response.status);
      return response.json();
    });
  }

  Promise.all([getJson("data/summary.json"), getJson("data/budget_grid.json")])
    .then(function (parts) { start(parts[0], parts[1]); })
    .catch(function (error) {
      byId("load-error").hidden = false;
      if (window.console) console.error(error);
    });
})();
