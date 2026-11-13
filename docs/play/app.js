// Grading Play Calculator: reads the inputs, calls calc.js, draws.
// Why this file holds no formulas: every number shown comes from
// window.PlayCalc (calc.js), which tests/test_play_parity.py checks against
// grace/play.py. This script parses what was typed, formats what calc.js
// returns, and draws it. Service tables, defaults and label text come from
// data/tiers.json, written by `python -m grace.play build`.
"use strict";

(function () {
  var calc = window.PlayCalc;
  // Company colours in table order, fixed (never cycled): a company keeps
  // its colour whatever else is on the chart.
  var COMPANY_COLORS = ["--series-1", "--series-2", "--series-4"];
  var EXTRA_COLOR = "--series-5";
  var BEST = "best";
  var COMPANY_FIELDS = ["p_top_pct", "p_second_pct", "price_top", "price_second", "price_lower"];
  var COMPANY_FIELD_LABELS = {
    p_top_pct: "Odds of the top grade (%)",
    p_second_pct: "Odds of the second grade (%)",
    price_top: "Sale price, top grade ($)",
    price_second: "Sale price, second grade ($)",
    price_lower: "Sale price, lower grade ($)"
  };
  var TIER_FIELDS = ["fee", "max_value", "days_min", "days_max", "min_cards"];
  var BAR_ROW_PX = 30;
  var BAR_CHROME_PX = 110;
  var TABLE_RATE_EVERY = 4;  // the fallback table shows every 4th grid rate (whole percents)
  var MAX_PCT = 100;  // a selling fee cannot take more than the whole sale
  var EXAMPLE_CLAUSE = "using the example odds and sale prices, the same for every company";

  var state = {
    data: null,
    rangeMode: null,
    weeklyRatePct: null,  // exact: a preset carries more digits than the box shows
    selected: BEST,
    inputs: null,
    companies: null,
    result: null,
    barKeys: [],
    barChart: null,
    rateChart: null,
    rankCache: {}
  };

  function byId(id) { return document.getElementById(id); }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
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

  // ---- formatting only -------------------------------------------------
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function dollars(v, digits) {
    return "$" + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }
  function money(v) {
    if (!isNum(v)) return "n/a";
    return (v < 0 ? "−" : "") + dollars(v, 2);
  }
  function signedMoney(v) {
    if (!isNum(v)) return "n/a";
    return (v < 0 ? "−" : "+") + dollars(v, 2);
  }
  function wholeMoney(v) {
    if (!isNum(v)) return "n/a";
    return (v < 0 ? "−" : "") + dollars(v, 0);
  }
  function wholeDays(v) { return isNum(v) ? v.toFixed(0) : "n/a"; }
  function trimmed(v, digits) { return String(Number(v.toFixed(digits))); }
  function pctText(v) { return isNum(v) ? trimmed(v, 4) + "%" : "n/a"; }
  function bigPct(v) {
    if (!isNum(v)) return "n/a";
    var digits = v < 100 ? 1 : 0;
    return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }) + "%";
  }
  function dayRange(tier) {
    if (!isNum(tier.days_min) && !isNum(tier.days_max)) return "–";
    if (tier.days_min === tier.days_max) return String(tier.days_min);
    return tier.days_min + "–" + tier.days_max;
  }

  // ---- lookups -----------------------------------------------------------
  function companyIndex(key) {
    var list = state.data.companies;
    for (var i = 0; i < list.length; i++) if (list[i].key === key) return i;
    return -1;
  }
  function companyByKey(key) { return state.data.companies[companyIndex(key)]; }
  function companyColor(key) {
    var index = companyIndex(key);
    var token = index >= 0 && index < COMPANY_COLORS.length ? COMPANY_COLORS[index] : EXTRA_COLOR;
    return cssVar(token);
  }
  function tierByKey(key) {
    var tiers = state.result.tiers;
    for (var i = 0; i < tiers.length; i++) if (tiers[i].key === key) return tiers[i];
    return null;
  }
  function labelOfKey(key) {
    if (key === "raw") return "sell raw today";
    var tier = tierByKey(key);
    return tier ? tier.label : key;
  }

  // ---- reading what was typed ---------------------------------------------
  function numberOf(input, fallback) {
    var value = parseFloat(input.value);
    if (!isFinite(value) || value < 0) {
      input.setAttribute("aria-invalid", "true");
      return fallback;
    }
    input.removeAttribute("aria-invalid");
    return value;
  }

  // A blank service-row cell is "not entered" (null), never a zero.
  function optionalNumberOf(input) {
    if (input.value.trim() === "") {
      input.removeAttribute("aria-invalid");
      return null;
    }
    var value = parseFloat(input.value);
    if (!isFinite(value) || value < 0) {
      input.setAttribute("aria-invalid", "true");
      return null;
    }
    input.removeAttribute("aria-invalid");
    return value;
  }

  function readInputs() {
    var defaults = state.data.defaults;
    var quantity = parseInt(byId("quantity").value, 10);
    var quantityBad = !isFinite(quantity) || quantity < 1;
    if (quantityBad) byId("quantity").setAttribute("aria-invalid", "true"); else byId("quantity").removeAttribute("aria-invalid");
    var sellFeeBox = byId("sell_fee_pct");
    var sellFeePct = numberOf(sellFeeBox, 0);
    if (sellFeePct > MAX_PCT) {
      sellFeeBox.setAttribute("aria-invalid", "true");
      sellFeePct = MAX_PCT;
    }
    var inputs = {
      raw_value_now: numberOf(byId("raw_value_now"), 0),
      quantity: quantityBad ? 1 : quantity,
      sell_fee_pct: sellFeePct,
      ship_to_buyer: numberOf(byId("ship_to_buyer"), 0),
      days_out: numberOf(byId("days_out"), 0),
      days_back: numberOf(byId("days_back"), 0),
      ship_out_batch: numberOf(byId("ship_out_batch"), 0),
      ship_back_batch: numberOf(byId("ship_back_batch"), 0),
      sell_days: numberOf(byId("sell_days"), 0),
      weekly_rate_pct: isNum(state.weeklyRatePct) ? state.weeklyRatePct : defaults.weekly_rate_pct,
      range_mode: state.rangeMode,
      companies: {}
    };
    var list = state.data.companies;
    for (var i = 0; i < list.length; i++) {
      var key = list[i].key;
      var entry = { fee_timing: byId(key + "-fee_timing").value };
      for (var f = 0; f < COMPANY_FIELDS.length; f++) {
        entry[COMPANY_FIELDS[f]] = numberOf(byId(key + "-" + COMPANY_FIELDS[f]), 0);
      }
      inputs.companies[key] = entry;
    }
    return inputs;
  }

  // The service data calc.js runs on: the shipped tables, with the typed
  // rows of an editable (unverified) company in place of its blank ones.
  function readCompanies() {
    var companies = JSON.parse(JSON.stringify(state.data.companies));
    for (var i = 0; i < companies.length; i++) {
      var company = companies[i];
      if (!company.editable) continue;
      for (var t = 0; t < company.tiers.length; t++) {
        var prefix = company.key + "-tier-" + t + "-";
        var row = { name: byId(prefix + "name").value.trim() };
        for (var f = 0; f < TIER_FIELDS.length; f++) {
          row[TIER_FIELDS[f]] = optionalNumberOf(byId(prefix + TIER_FIELDS[f]));
        }
        company.tiers[t] = row;
      }
    }
    return companies;
  }

  // ---- building the input widgets from tiers.json -------------------------
  function fieldBox(labelText, control) {
    var box = el("div", { "class": "field" });
    box.appendChild(el("label", { "for": control.id }, labelText));
    box.appendChild(control);
    return box;
  }

  function numberInput(id, value, step) {
    var input = el("input", { type: "number", id: id, min: "0", step: step, inputmode: "decimal" });
    if (value !== null && value !== undefined) input.value = String(value);
    input.addEventListener("input", recompute);
    return input;
  }

  function swatch(key) {
    var dot = el("span", { "class": "swatch", "data-company": key, "aria-hidden": "true" });
    dot.style.background = companyColor(key);
    return dot;
  }

  function buildCompanyPanels() {
    var holder = byId("company-panels");
    holder.textContent = "";
    var list = state.data.companies;
    for (var i = 0; i < list.length; i++) {
      var company = list[i];
      var defaults = state.data.defaults.companies[company.key];
      var panel = el("fieldset", { "data-company": company.key });
      var legend = el("legend");
      legend.appendChild(swatch(company.key));
      legend.appendChild(document.createTextNode(company.name));
      panel.appendChild(legend);
      var fields = el("div", { "class": "fields" });
      for (var f = 0; f < COMPANY_FIELDS.length; f++) {
        var name = COMPANY_FIELDS[f];
        var step = name.indexOf("pct") === -1 ? "10" : "1";
        fields.appendChild(fieldBox(COMPANY_FIELD_LABELS[name], numberInput(company.key + "-" + name, defaults[name], step)));
      }
      var readout = el("div", { "class": "field" });
      readout.appendChild(el("span", { "class": "label" }, "Odds of a lower grade"));
      readout.appendChild(el("div", { "class": "readout", id: company.key + "-lower" }));
      fields.appendChild(readout);
      panel.appendChild(fields);
      var timing = el("div", { "class": "fields" });
      var select = el("select", { id: company.key + "-fee_timing" });
      var timings = state.data.fee_timings;
      for (var m = 0; m < timings.length; m++) {
        var option = el("option", { value: timings[m] }, state.data.fee_timing_labels[timings[m]]);
        if (timings[m] === defaults.fee_timing) option.selected = true;
        select.appendChild(option);
      }
      select.addEventListener("change", recompute);
      var timingBox = fieldBox("When " + company.name + " charges the grading fee", select);
      timingBox.classList.add("wide");
      timing.appendChild(timingBox);
      panel.appendChild(timing);
      panel.appendChild(el("p", { "class": "source" }, "Where that default comes from: " + company.fee_timing_source + "."));
      if (!company.verified) {
        panel.appendChild(el("p", { "class": "source warn", id: company.key + "-note" }, company.note));
        var open = el("button", { type: "button", "class": "small" }, "Enter " + company.name + " prices");
        open.addEventListener("click", openServices);
        panel.appendChild(open);
      }
      var copy = el("button", { type: "button", "class": "small", "data-copy-from": company.key },
        "Use " + company.name + "’s odds and prices for every company");
      copy.addEventListener("click", onCopy);
      panel.appendChild(copy);
      holder.appendChild(panel);
    }
  }

  function openServices() {
    byId("services-details").open = true;
    byId("services").scrollIntoView();
  }

  function onCopy(event) {
    var from = event.currentTarget.getAttribute("data-copy-from");
    var list = state.data.companies;
    for (var i = 0; i < list.length; i++) {
      if (list[i].key === from) continue;
      for (var f = 0; f < COMPANY_FIELDS.length; f++) {
        byId(list[i].key + "-" + COMPANY_FIELDS[f]).value = byId(from + "-" + COMPANY_FIELDS[f]).value;
      }
    }
    recompute();
  }

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

  function sourceLine(company) {
    var line = el("p", { "class": "source" });
    line.appendChild(document.createTextNode(company.verified ? "Source: " : "Where to find them: "));
    line.appendChild(el("a", { href: company.source_url, rel: "noopener" }, company.source_url));
    var parts = [];
    if (company.as_of) parts.push("read " + company.as_of);
    for (var q = 0; q < company.source_quotes.length; q++) parts.push("“" + company.source_quotes[q] + "”");
    if (company.verified) parts.push(company.note);
    if (parts.length) line.appendChild(document.createTextNode(" · " + parts.join(" · ")));
    return line;
  }

  function editableRow(company, index) {
    var prefix = company.key + "-tier-" + index + "-";
    var tr = el("tr");
    var nameCell = el("td");
    var nameInput = el("input", { type: "text", id: prefix + "name", placeholder: "level name", "aria-label": company.name + " row " + (index + 1) + " level name" });
    nameInput.addEventListener("input", recompute);
    nameCell.appendChild(nameInput);
    tr.appendChild(nameCell);
    var labels = { fee: "price per card", max_value: "max value", days_min: "fastest business days", days_max: "slowest business days", min_cards: "minimum cards" };
    for (var f = 0; f < TIER_FIELDS.length; f++) {
      var cell = el("td", { "class": "num" });
      var input = numberInput(prefix + TIER_FIELDS[f], null, "1");
      input.setAttribute("aria-label", company.name + " row " + (index + 1) + " " + labels[TIER_FIELDS[f]]);
      cell.appendChild(input);
      tr.appendChild(cell);
    }
    return tr;
  }

  function buildServiceTables() {
    var holder = byId("service-tables");
    holder.textContent = "";
    var data = state.data;
    for (var i = 0; i < data.companies.length; i++) {
      var company = data.companies[i];
      var heading = el("h3");
      heading.appendChild(swatch(company.key));
      heading.appendChild(document.createTextNode(company.name + " "));
      heading.appendChild(el("span", { "class": "provenance" },
        company.verified ? "read from the official page on " + company.as_of : "not verified · your entries"));
      holder.appendChild(heading);
      if (!company.verified) holder.appendChild(el("p", { "class": "warn" }, company.note));
      holder.appendChild(sourceLine(company));

      var wrap = el("div", { "class": "table-wrap" });
      var table = el("table", { id: company.key + "-tiers" });
      var tbody = el("tbody");
      if (company.editable) {
        headerRow(table, ["level", "$ per card", company.cap_label + " ($)", "fastest (business days)", "slowest (business days)", "minimum cards"], 1);
        for (var e = 0; e < company.tiers.length; e++) tbody.appendChild(editableRow(company, e));
      } else {
        headerRow(table, ["level", "$ per card", company.cap_label, "est. turnaround (business days)", "minimum cards"], 1);
        for (var t = 0; t < company.tiers.length; t++) {
          var tier = company.tiers[t];
          tbody.appendChild(row([tier.name, money(tier.fee), isNum(tier.max_value) ? wholeMoney(tier.max_value) : "–",
            dayRange(tier), isNum(tier.min_cards) ? String(tier.min_cards) : "–"], 1, ""));
        }
      }
      table.appendChild(tbody);
      wrap.appendChild(table);
      holder.appendChild(wrap);
    }
  }

  function buildSegmented(holderId, values, labelOf, isOn, onPick) {
    var holder = byId(holderId);
    holder.textContent = "";
    for (var i = 0; i < values.length; i++) {
      var button = el("button", { type: "button", "data-index": String(i), "aria-pressed": String(isOn(values[i])) }, labelOf(values[i]));
      button.addEventListener("click", onPick);
      holder.appendChild(button);
    }
  }

  function markSegmented(holderId, values, isOn) {
    var buttons = byId(holderId).querySelectorAll("button");
    for (var i = 0; i < buttons.length; i++) buttons[i].setAttribute("aria-pressed", String(isOn(values[i])));
  }

  function isCurrentRange(mode) { return mode === state.rangeMode; }
  function isCurrentPreset(preset) { return preset.weekly_rate_pct === state.weeklyRatePct; }

  function onRangePick(event) {
    state.rangeMode = state.data.range_modes[Number(event.currentTarget.getAttribute("data-index"))];
    recompute();
  }

  function setRate(value, showInBox) {
    state.weeklyRatePct = value;
    if (showInBox) byId("weekly_rate_pct").value = trimmed(value, 4);
    byId("weekly_rate_slider").value = String(value);
    byId("weekly_rate_pct").removeAttribute("aria-invalid");
  }

  function onPresetPick(event) {
    var preset = state.data.presets[Number(event.currentTarget.getAttribute("data-index"))];
    setRate(preset.weekly_rate_pct, true);
    recompute();
  }

  function fillStaticText() {
    var data = state.data;
    var labels = data.provenance_labels;
    var pills = document.querySelectorAll(".provenance[data-prov]");
    for (var i = 0; i < pills.length; i++) pills[i].textContent = labels[pills[i].getAttribute("data-prov")];
    var boxes = document.querySelectorAll(".why[data-why]");
    for (var j = 0; j < boxes.length; j++) {
      var entry = data.why[boxes[j].getAttribute("data-why")];
      boxes[j].textContent = "";
      boxes[j].appendChild(el("b", null, "Why this number"));
      boxes[j].appendChild(el("p", null, entry.question));
      boxes[j].appendChild(el("p", null, "Kind: " + entry.kind));
      boxes[j].appendChild(el("p", null, "Could lie if: " + entry.lie));
    }
    byId("footer-text").textContent = data.footer;
    var links = byId("official-links");
    links.textContent = "";
    var linked = 0;
    for (var c = 0; c < data.companies.length; c++) {
      var company = data.companies[c];
      if (!company.verified) continue;
      links.appendChild(document.createTextNode(linked === 0 ? " Official pages: " : " · "));
      links.appendChild(el("a", { href: company.source_url, rel: "noopener" }, company.name));
      linked += 1;
    }
    if (linked > 0) links.appendChild(document.createTextNode("."));
  }

  function fillDefaults() {
    var defaults = state.data.defaults;
    var inputs = document.querySelectorAll("input[data-input]");
    for (var i = 0; i < inputs.length; i++) {
      inputs[i].value = String(defaults[inputs[i].getAttribute("data-input")]);
      if (inputs[i].id !== "weekly_rate_pct") inputs[i].addEventListener("input", recompute);
    }
    state.rangeMode = defaults.range_mode;
    setRate(defaults.weekly_rate_pct, true);
    byId("weekly_rate_pct").addEventListener("input", function () {
      var box = byId("weekly_rate_pct");
      var value = numberOf(box, state.weeklyRatePct);
      state.weeklyRatePct = value;
      byId("weekly_rate_slider").value = String(value);
      recompute();
    });
    byId("weekly_rate_slider").addEventListener("input", function () {
      setRate(Number(byId("weekly_rate_slider").value), true);
      recompute();
    });
  }

  // ---- drawing: text -------------------------------------------------------
  function renderOddsReadouts() {
    var list = state.data.companies;
    for (var i = 0; i < list.length; i++) {
      var entry = state.inputs.companies[list[i].key];
      var box = byId(list[i].key + "-lower");
      if (calc.oddsAreValid(entry)) {
        box.textContent = pctText(calc.lowerOddsPct(entry));
        box.classList.remove("bad");
      } else {
        box.textContent = "the odds add up to more than 100%";
        box.classList.add("bad");
      }
    }
  }

  function renderRateNote() {
    var r = state.result;
    byId("rate-note").textContent = pctText(r.weekly_rate_pct) + " a week compounds to about " + bigPct(r.annual_rate_pct) + " a year.";
    markSegmented("rate-presets", state.data.presets, isCurrentPreset);
    markSegmented("range-modes", state.data.range_modes, isCurrentRange);
  }

  // True while every company's odds and prices are still the shipped example.
  function usingExampleNumbers() {
    var list = state.data.companies;
    for (var i = 0; i < list.length; i++) {
      var typed = state.inputs.companies[list[i].key];
      var example = state.data.defaults.companies[list[i].key];
      for (var f = 0; f < COMPANY_FIELDS.length; f++) {
        if (typed[COMPANY_FIELDS[f]] !== example[COMPANY_FIELDS[f]]) return false;
      }
    }
    return true;
  }

  // A field marked invalid was replaced by a stand-in, so the result is not final.
  function answerNote(text) {
    var invalid = document.querySelectorAll('[aria-invalid="true"]').length;
    if (invalid === 0) return text;
    if (invalid === 1) return "Fix the highlighted field first. Until then the numbers here use a stand-in for it, so this is not your answer yet.";
    return "Fix the highlighted fields first. Until then the numbers here use stand-ins for them, so this is not your answer yet.";
  }

  function renderAnswer() {
    var r = state.result;
    var rate = pctText(r.weekly_rate_pct) + " a week";
    if (usingExampleNumbers()) rate += ", " + EXAMPLE_CLAUSE;
    var label = byId("answer-label");
    var value = byId("answer-value");
    var note = byId("answer-note");
    byId("live-bar").hidden = false;
    byId("live-label").textContent = "Best at " + rate + ":";
    if (r.best === null) {
      label.textContent = "No grading level to compare at " + rate;
      value.textContent = "Sell it raw today: " + money(r.sell_raw_today);
      note.textContent = answerNote("No service level is usable with these inputs; the table below says why.");
      byId("live-value").textContent = "sell it raw today, " + money(r.sell_raw_today) + " a card";
      return;
    }
    var best = tierByKey(r.best);
    label.textContent = "Best play at " + rate;
    if (r.play === "grade") {
      byId("live-value").textContent = best.label + ", " + signedMoney(best.value_of_play) + " a card";
      value.textContent = best.label + ": " + signedMoney(best.value_of_play) + " a card";
      var text = "against selling it raw today for " + money(r.sell_raw_today) + ". In today’s dollars the grading play is worth " +
        money(best.npv) + ", and the money is back in about " + wholeDays(best.day_sale) + " days.";
      if (r.quantity > 1) text += " For the batch of " + r.quantity + " cards: " + signedMoney(r.best_batch_value_of_play) + ".";
      note.textContent = answerNote(text);
    } else {
      byId("live-value").textContent = "sell it raw today, " + money(r.sell_raw_today) + " a card";
      value.textContent = "Sell it raw today: " + money(r.sell_raw_today) + " a card";
      note.textContent = answerNote("The best grading level, " + best.label + ", comes to " + signedMoney(best.value_of_play) +
        " a card against selling raw today, with the money back in about " + wholeDays(best.day_sale) + " days.");
    }
  }

  function flagText(tier) {
    var company = companyByKey(tier.company);
    var parts = [];
    for (var i = 0; i < tier.flags.length; i++) {
      if (tier.flags[i] === "below_minimum") {
        parts.push(state.data.flag_text.below_minimum.replace("{minimum}", String(tier.min_cards)));
      } else if (tier.flags[i] === "over_cap") {
        parts.push(company.cap_flag_text);
      }
    }
    if (!tier.eligible) parts.push("not charted or ranked");
    return parts.join("; ");
  }

  function whenPaid(tier) {
    var how = tier.fee_timing === "at_checkout" ? "up front" : "after grading";
    return "day " + wholeDays(tier.fee_day) + " (" + how + ")";
  }

  function renderTable() {
    var r = state.result;
    var table = byId("results-table");
    table.textContent = "";
    headerRow(table, ["level", "fee", "fee paid", "turnaround used (business days)", "money back (days)", "costs in today’s dollars", "sale in today’s dollars", "value of the play", "vs the best", "flags"], 1);
    var tbody = el("tbody");
    for (var i = 0; i < r.tiers.length; i++) {
      var tier = r.tiers[i];
      var rowClass = tier.key === r.best ? "chosen" : (tier.eligible ? "" : "flagged");
      var tr = row([tier.label, money(tier.fee), whenPaid(tier), trimmed(tier.turnaround_business_days, 1), wholeDays(tier.day_sale), money(tier.pv_costs),
        money(tier.pv_proceeds), signedMoney(tier.value_of_play), tier.gap_to_best === null ? "–" : money(tier.gap_to_best)], 1, rowClass);
      tr.appendChild(el("td", { "class": "wrap" }, flagText(tier)));
      tbody.appendChild(tr);
    }
    var rawRow = row(["Sell raw today", "–", "–", "–", "0", "–", money(r.sell_raw_today), signedMoney(0), "–"], 1, r.play === "grade" ? "" : "chosen");
    rawRow.appendChild(el("td", { "class": "wrap" }, "the zero line"));
    tbody.appendChild(rawRow);
    table.appendChild(tbody);

    var notes = [];
    var list = state.companies;
    for (var c = 0; c < list.length; c++) {
      var complete = 0;
      for (var t = 0; t < list[c].tiers.length; t++) if (calc.tierIsComplete(list[c].tiers[t])) complete += 1;
      if (complete === 0) notes.push(list[c].name + " has no prices entered, so it is in no chart or ranking (enter them under Service levels).");
    }
    for (var s = 0; s < r.skipped_companies.length; s++) {
      notes.push(companyByKey(r.skipped_companies[s]).name + " is left out: its odds add up to more than 100%.");
    }
    byId("table-notes").textContent = notes.join(" ");
  }

  function renderBarNotes() {
    var r = state.result;
    var hidden = [];
    for (var i = 0; i < r.tiers.length; i++) if (!r.tiers[i].eligible) hidden.push(r.tiers[i].label);
    var text = "";
    if (hidden.length) text = "Not charted, flagged in the table below: " + hidden.join(", ") + ".";
    if (state.barKeys.length === 0) text = "No level can be charted with these inputs. " + text;
    byId("bars-notes").textContent = text;
  }

  function rangeText(segment) {
    if (segment.from_pct === segment.to_pct) return "at " + pctText(segment.from_pct);
    return pctText(segment.from_pct) + " to " + pctText(segment.to_pct);
  }

  function renderSegments() {
    var segments = state.result.best_by_rate;
    var parts = [];
    for (var i = 0; i < segments.length; i++) parts.push(rangeText(segments[i]) + ": " + labelOfKey(segments[i].winner));
    byId("rates-segments").textContent = "Best play by weekly rate, in " + pctText(state.data.rate_grid.step_pct) + " steps — " + parts.join(" · ") + ".";
  }

  function renderRatesTable() {
    var grid = state.result.rate_grid;
    var table = byId("rates-table");
    table.textContent = "";
    var names = ["level"];
    var columns = [];
    for (var g = 0; g < grid.rates_pct.length; g += TABLE_RATE_EVERY) {
      columns.push(g);
      names.push(pctText(grid.rates_pct[g]) + " a week");
    }
    headerRow(table, names, 1);
    var tbody = el("tbody");
    for (var k = 0; k < grid.keys.length; k++) {
      var values = [labelOfKey(grid.keys[k])];
      for (var c = 0; c < columns.length; c++) values.push(signedMoney(grid.value_of_play[grid.keys[k]][columns[c]]));
      tbody.appendChild(row(values, 1, grid.keys[k] === state.result.best ? "chosen" : ""));
    }
    table.appendChild(tbody);
  }

  function selectedTier() {
    var r = state.result;
    if (state.selected !== BEST) {
      var picked = tierByKey(state.selected);
      if (picked) return picked;
      state.selected = BEST;
    }
    if (r.best !== null) return tierByKey(r.best);
    return r.tiers.length ? r.tiers[0] : null;
  }

  function renderFlowPicker() {
    var r = state.result;
    var select = byId("flow-tier");
    select.textContent = "";
    var bestLabel = r.best === null ? "Best level (none usable)" : "Best level (" + tierByKey(r.best).label + ")";
    select.appendChild(el("option", { value: BEST }, bestLabel));
    for (var i = 0; i < r.tiers.length; i++) {
      select.appendChild(el("option", { value: r.tiers[i].key }, r.tiers[i].label + (r.tiers[i].eligible ? "" : " (flagged)")));
    }
    select.value = state.selected;
  }

  function flowMoney(v) {
    if (v === 0) return money(0);
    return signedMoney(v);
  }

  // A cost is shown negative and the sale as computed (it can come out
  // negative), so the tiles always add up to the net value today beside them.
  function flowTile(flow) {
    var cost = flow.kind === "cost";
    var amount = cost ? -flow.amount : flow.amount;
    var valueToday = cost ? -flow.value_today : flow.value_today;
    var tile = el("div", { "class": cost ? "tile cost" : "tile" });
    tile.appendChild(el("div", { "class": "label" }, "Day " + wholeDays(flow.day) + " · " + state.data.flow_labels[flow.what]));
    tile.appendChild(el("div", { "class": "value" }, flowMoney(amount)));
    tile.appendChild(el("div", { "class": "note" }, "worth " + flowMoney(valueToday) + " today"));
    return tile;
  }

  function renderFlows() {
    renderFlowPicker();
    var strip = byId("flow-strip");
    strip.textContent = "";
    var tier = selectedTier();
    if (!tier) {
      byId("flow-summary").textContent = "No level to show with these inputs.";
      return;
    }
    for (var i = 0; i < tier.flows.length; i++) strip.appendChild(flowTile(tier.flows[i]));
    var total = el("div", { "class": "tile headline" });
    total.appendChild(el("div", { "class": "label" }, tier.label + " · net value today (NPV)"));
    total.appendChild(el("div", { "class": "value" }, money(tier.npv)));
    total.appendChild(el("div", { "class": "note" }, "sticker profit " + money(tier.undiscounted_profit)));
    strip.appendChild(total);
    byId("flow-summary").textContent =
      "Per card. The fee is " + money(tier.fee) + " on the price list and " + money(tier.fee_value_today) +
      " in today’s dollars, because it is paid on day " + wholeDays(tier.fee_day) + ". Selling raw today is " +
      money(state.result.sell_raw_today) + ", so this play is " + signedMoney(tier.value_of_play) +
      " (without discounting it would look like " + signedMoney(tier.undiscounted_value_of_play) + ").";
  }

  // ---- drawing: charts -------------------------------------------------------
  function shownCompanies(keys) {
    var seen = {};
    var labels = [];
    var text = cssVar("--text");
    for (var i = 0; i < keys.length; i++) {
      var tier = tierByKey(keys[i]);
      if (!tier || seen[tier.company]) continue;
      seen[tier.company] = true;
      var color = companyColor(tier.company);
      labels.push({ text: companyByKey(tier.company).name, fillStyle: color, strokeStyle: color, lineWidth: 0, fontColor: text, hidden: false });
    }
    return labels;
  }

  function zeroAwareGrid() {
    var strong = cssVar("--text-muted");
    var faint = cssVar("--border");
    return {
      color: function (ctx) { return ctx.tick && ctx.tick.value === 0 ? strong : faint; },
      lineWidth: function (ctx) { return ctx.tick && ctx.tick.value === 0 ? 2 : 1; }
    };
  }

  function barTooltipLines(item) {
    var r = state.result;
    var tier = tierByKey(state.barKeys[item.dataIndex]);
    var lines = [
      "Value of the play: " + signedMoney(tier.value_of_play),
      "  (selling raw today: " + money(r.sell_raw_today) + ")",
      "Net value today (NPV): " + money(tier.npv)
    ];
    if (tier.key === r.best) {
      lines.push("This is the best level");
    } else {
      lines.push("vs the best, " + tierByKey(r.best).label + ": " + money(tier.gap_to_best));
    }
    lines.push("Money back in about " + wholeDays(tier.day_sale) + " days");
    lines.push("Fee " + money(tier.fee) + ", paid day " + wholeDays(tier.fee_day));
    lines.push("  = " + money(tier.fee_value_today) + " in today’s dollars");
    return lines;
  }

  function buildBarChart() {
    var muted = cssVar("--text-muted");
    var text = cssVar("--text");
    state.barChart = new window.Chart(byId("bars-chart"), {
      type: "bar",
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 4, maxBarThickness: 22 }] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false, animation: false,
        // the whole row is the hit target, so a bar too short to aim at still answers
        interaction: { mode: "index", axis: "y", intersect: false },
        scales: {
          x: {
            title: { display: true, text: "value of the play, $ per card (0 = sell raw today)", color: muted },
            ticks: { color: muted, callback: function (v) { return wholeMoney(v); } },
            grid: zeroAwareGrid()
          },
          y: {
            ticks: {
              color: text, autoSkip: false,
              font: function (ctx) { return { weight: state.barKeys[ctx.index] === state.result.best ? "700" : "400" }; }
            },
            grid: { display: false }
          }
        },
        plugins: {
          legend: {
            onClick: function () {},
            labels: { color: text, boxWidth: 14, generateLabels: function () { return shownCompanies(state.barKeys); } }
          },
          tooltip: {
            displayColors: false,
            callbacks: {
              title: function (items) { return items.length ? tierByKey(state.barKeys[items[0].dataIndex]).label : ""; },
              label: barTooltipLines
            }
          }
        },
        onClick: function (event, elements) {
          if (!elements.length) return;
          state.selected = state.barKeys[elements[0].index];
          renderFlows();
        },
        onHover: function (event, elements) {
          event.native.target.style.cursor = elements.length ? "pointer" : "default";
        }
      }
    });
  }

  function renderBars() {
    var r = state.result;
    var labels = [];
    var values = [];
    var colors = [];
    state.barKeys = [];
    for (var i = 0; i < r.tiers.length; i++) {
      var tier = r.tiers[i];
      if (!tier.eligible) continue;
      state.barKeys.push(tier.key);
      labels.push(tier.label + (tier.key === r.best ? " (best)" : ""));
      values.push(tier.value_of_play);
      colors.push(companyColor(tier.company));
    }
    renderBarNotes();
    if (!window.Chart) return;
    byId("bars-box").hidden = state.barKeys.length === 0;
    byId("bars-box").style.height = (state.barKeys.length * BAR_ROW_PX + BAR_CHROME_PX) + "px";
    if (!state.barChart) buildBarChart();
    state.barChart.data.labels = labels;
    state.barChart.data.datasets[0].data = values;
    state.barChart.data.datasets[0].backgroundColor = colors;
    state.barChart.resize();
    state.barChart.update("none");
  }

  // Inline cursor: a dashed line at the chosen weekly rate, and a hairline
  // under the pointer so the reader aims at a rate, not at a thin line.
  var cursorPlugin = {
    id: "rateCursor",
    afterDatasetsDraw: function (chart) {
      var area = chart.chartArea;
      var ctx = chart.ctx;
      var scale = chart.scales.x;
      var rate = state.result.weekly_rate_pct;
      ctx.save();
      ctx.lineWidth = 1;
      ctx.strokeStyle = cssVar("--text-muted");
      if (rate >= scale.min && rate <= scale.max) {
        var x = scale.getPixelForValue(rate);
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(x, area.top);
        ctx.lineTo(x, area.bottom);
        ctx.stroke();
        var nearRight = x > (area.left + area.right) / 2;
        ctx.fillStyle = cssVar("--text-muted");
        ctx.font = "12px " + cssVar("--font");
        ctx.textAlign = nearRight ? "right" : "left";
        ctx.textBaseline = "top";
        ctx.fillText("your rate, " + pctText(rate), nearRight ? x - 6 : x + 6, area.top + 4);
      }
      var active = chart.getActiveElements();
      if (active.length) {
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(active[0].element.x, area.top);
        ctx.lineTo(active[0].element.x, area.bottom);
        ctx.stroke();
      }
      ctx.restore();
    }
  };

  // The ranking at one grid rate, from calc.js (cached per render).
  function rankingAt(gridIndex) {
    if (!state.rankCache[gridIndex]) {
      var ranked = calc.rankAtRate(state.result.rate_grid, gridIndex);
      var byKey = {};
      for (var i = 0; i < ranked.length; i++) byKey[ranked[i].key] = { place: i, gap: ranked[i].gap_to_best, value: ranked[i].value_of_play };
      state.rankCache[gridIndex] = { byKey: byKey, bestValue: ranked.length ? ranked[0].value_of_play : null };
    }
    return state.rankCache[gridIndex];
  }

  // One short line per level so the list fits a 375 px screen: the value,
  // then the gap to the best level in brackets (the title says so).
  function rateTooltipLabel(item) {
    var entry = rankingAt(item.dataIndex).byKey[item.dataset.tierKey];
    var tail = entry.place === 0 ? " (best)" : " (" + money(entry.gap) + ")";
    return item.dataset.label + ": " + signedMoney(entry.value) + tail;
  }

  function buildRateChart() {
    var muted = cssVar("--text-muted");
    var text = cssVar("--text");
    var grid = state.data.rate_grid;
    state.rateChart = new window.Chart(byId("rates-chart"), {
      type: "line",
      data: { datasets: [] },
      plugins: [cursorPlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            type: "linear", min: grid.min_pct, max: grid.max_pct,
            title: { display: true, text: "weekly opportunity rate (%)", color: muted },
            ticks: { color: muted, callback: function (v) { return v + "%"; } },
            grid: { color: cssVar("--border") }
          },
          y: {
            suggestedMin: 0, suggestedMax: 0,
            title: { display: true, text: "value of the play, $ per card", color: muted },
            ticks: { color: muted, callback: function (v) { return wholeMoney(v); } },
            grid: zeroAwareGrid()
          }
        },
        plugins: {
          legend: {
            onClick: function () {},
            labels: { color: text, boxWidth: 14, generateLabels: function () { return shownCompanies(state.result.rate_grid.keys); } }
          },
          tooltip: {
            itemSort: function (a, b) {
              var places = rankingAt(a.dataIndex).byKey;
              return places[a.dataset.tierKey].place - places[b.dataset.tierKey].place;
            },
            callbacks: {
              title: function (items) {
                if (!items.length) return "";
                return ["At " + pctText(items[0].parsed.x) + " a week", "value of the play (gap to the best)"];
              },
              label: rateTooltipLabel,
              footer: function (items) {
                if (!items.length) return "";
                var best = rankingAt(items[0].dataIndex).bestValue;
                return best !== null && best <= 0 ? "Every level is under zero: selling raw today wins" : "";
              }
            }
          }
        }
      }
    });
  }

  function renderOffChartNote() {
    var note = byId("rates-off-chart");
    var rate = state.result.weekly_rate_pct;
    var top = state.data.rate_grid.max_pct;
    var off = rate > top;
    note.hidden = !off;
    note.textContent = off ? "Your rate, " + pctText(rate) + " a week, is off this chart, which stops at " + pctText(top) +
      ". The answer and the table still use your rate." : "";
  }

  function renderRates() {
    var r = state.result;
    var grid = r.rate_grid;
    state.rankCache = {};
    renderSegments();
    renderOffChartNote();
    renderRatesTable();
    if (!window.Chart) return;
    byId("rates-box").hidden = grid.keys.length === 0;
    if (!state.rateChart) buildRateChart();
    var datasets = [];
    for (var k = 0; k < grid.keys.length; k++) {
      var key = grid.keys[k];
      var tier = tierByKey(key);
      var points = [];
      for (var g = 0; g < grid.rates_pct.length; g++) points.push({ x: grid.rates_pct[g], y: grid.value_of_play[key][g] });
      var color = companyColor(tier.company);
      var isBest = key === r.best;
      datasets.push({
        label: tier.label, tierKey: key, data: points, borderColor: color, backgroundColor: color,
        borderWidth: isBest ? 3 : 1.5, pointRadius: 0, pointHoverRadius: 4, pointHitRadius: 12, tension: 0,
        order: isBest ? 0 : 1
      });
    }
    state.rateChart.data.datasets = datasets;
    state.rateChart.update("none");
  }

  function showChartFallback() {
    byId("bars-box").hidden = true;
    byId("rates-box").hidden = true;
    byId("bars-fallback").hidden = false;
    byId("rates-fallback").hidden = false;
    byId("rates-details").open = true;
  }

  // ---- the loop: read, compute in calc.js, draw ---------------------------
  function recompute() {
    state.inputs = readInputs();
    state.companies = readCompanies();
    state.result = calc.evaluate(state.inputs, state.companies);
    renderOddsReadouts();
    renderRateNote();
    renderAnswer();
    renderBars();
    renderRates();
    renderTable();
    renderFlows();
  }

  function rebuildCharts() {
    if (state.barChart) { state.barChart.destroy(); state.barChart = null; }
    if (state.rateChart) { state.rateChart.destroy(); state.rateChart = null; }
    var swatches = document.querySelectorAll(".swatch[data-company]");
    for (var i = 0; i < swatches.length; i++) swatches[i].style.background = companyColor(swatches[i].getAttribute("data-company"));
    recompute();
  }

  function start(data) {
    state.data = data;
    fillStaticText();
    buildCompanyPanels();
    buildServiceTables();
    fillDefaults();
    buildSegmented("rate-presets", data.presets, function (preset) { return preset.label; }, isCurrentPreset, onPresetPick);
    buildSegmented("range-modes", data.range_modes, function (mode) { return data.range_mode_labels[mode]; }, isCurrentRange, onRangePick);
    byId("flow-tier").addEventListener("change", function () {
      state.selected = byId("flow-tier").value;
      renderFlows();
    });
    if (window.Chart) {
      window.Chart.defaults.font.family = cssVar("--font");
      var media = window.matchMedia("(prefers-color-scheme: dark)");
      if (media.addEventListener) media.addEventListener("change", rebuildCharts); else media.addListener(rebuildCharts);
    } else {
      showChartFallback();
    }
    recompute();
  }

  fetch("data/tiers.json")
    .then(function (response) {
      if (!response.ok) throw new Error("data/tiers.json " + response.status);
      return response.json();
    })
    .then(start)
    .catch(function (error) {
      byId("load-error").hidden = false;
      if (window.console) console.error(error);
    });
})();
