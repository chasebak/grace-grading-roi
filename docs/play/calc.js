// Grading Play Calculator: every formula, as pure functions with no DOM access.
// Why this file computes: a calculator has to answer inputs nobody precomputed.
// The source of truth is grace/play.py. This file mirrors it one function for
// one function (same names, same order of arithmetic), and
// tests/test_play_parity.py runs it under node on the fixtures in
// data/parity.json and compares every number with the Python output.
// Loadable by the page (window.PlayCalc) and by node (module.exports).
"use strict";

(function (root) {
  var FEE_AT_CHECKOUT = "at_checkout";
  var RANGE_FASTEST = "fastest";
  var RANGE_SLOWEST = "slowest";
  var CAP_MOVES_UP = "moves_up";
  var FLAG_BELOW_MINIMUM = "below_minimum";
  var FLAG_OVER_CAP = "over_cap";
  var PLAY_GRADE = "grade";
  var PLAY_SELL_RAW = "sell_raw";
  var RAW_KEY = "raw";
  var BUSINESS_DAYS_PER_WEEK = 5;
  var CALENDAR_DAYS_PER_WEEK = 7;
  var WEEKS_PER_YEAR = 52;
  var PERCENT = 100;
  // Typed odds like 33.3 + 66.7 can land a hair above 100 in floating point.
  var ODDS_SLACK_PCT = 1e-9;
  // Chart 2's grid: the weekly opportunity rate from 0% to 10% in 0.25% steps.
  var RATE_GRID_STEP_PCT = 0.25;
  var RATE_GRID_STEPS = 40;

  function isNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  // [fastest, slowest] stated business days, or null when neither is given.
  function tierDayRange(tier) {
    var daysMin = tier.days_min;
    var daysMax = tier.days_max;
    var hasMin = isNumber(daysMin) && daysMin >= 0;
    var hasMax = isNumber(daysMax) && daysMax >= 0;
    if (hasMin && hasMax) return [Math.min(daysMin, daysMax), Math.max(daysMin, daysMax)];
    if (hasMin) return [daysMin, daysMin];
    if (hasMax) return [daysMax, daysMax];
    return null;
  }

  // A row is usable only with a fee and a turnaround; blank rows are not.
  function tierIsComplete(tier) {
    var fee = tier.fee;
    if (!isNumber(fee) || fee < 0) return false;
    return tierDayRange(tier) !== null;
  }

  function turnaroundBusinessDays(tier, rangeMode) {
    var range = tierDayRange(tier);
    var fastest = range[0];
    var slowest = range[1];
    if (rangeMode === RANGE_FASTEST) return fastest;
    if (rangeMode === RANGE_SLOWEST) return slowest;
    return (fastest + slowest) / 2;
  }

  function businessToCalendarDays(businessDays) {
    return businessDays * CALENDAR_DAYS_PER_WEEK / BUSINESS_DAYS_PER_WEEK;
  }

  // amount / (1 + weeklyRate) ^ (day / 7)
  function presentValue(amount, day, weeklyRate) {
    var weeks = day / CALENDAR_DAYS_PER_WEEK;
    return amount / Math.pow(1 + weeklyRate, weeks);
  }

  function weeklyPctFromAnnual(annualPct) {
    var growth = 1 + annualPct / PERCENT;
    var weeklyGrowth = Math.pow(growth, 1 / WEEKS_PER_YEAR);
    return (weeklyGrowth - 1) * PERCENT;
  }

  function annualPctFromWeekly(weeklyPct) {
    var growth = Math.pow(1 + weeklyPct / PERCENT, WEEKS_PER_YEAR);
    return (growth - 1) * PERCENT;
  }

  function cleanQuantity(inputs) {
    var quantity = inputs.quantity;
    if (!isNumber(quantity) || quantity < 1) return 1;
    return quantity;
  }

  function oddsAreValid(companyInputs) {
    var pTopPct = companyInputs.p_top_pct;
    var pSecondPct = companyInputs.p_second_pct;
    if (pTopPct < 0 || pSecondPct < 0) return false;
    return pTopPct + pSecondPct <= PERCENT + ODDS_SLACK_PCT;
  }

  function lowerOddsPct(companyInputs) {
    var lowerPct = PERCENT - companyInputs.p_top_pct - companyInputs.p_second_pct;
    if (lowerPct < 0) return 0;
    return lowerPct;
  }

  function netSale(price, sellFee, shipToBuyer) {
    return price * (1 - sellFee) - shipToBuyer;
  }

  function expectedNetSale(companyInputs, sellFee, shipToBuyer) {
    var pTop = companyInputs.p_top_pct / PERCENT;
    var pSecond = companyInputs.p_second_pct / PERCENT;
    var pLower = lowerOddsPct(companyInputs) / PERCENT;
    var netTop = netSale(companyInputs.price_top, sellFee, shipToBuyer);
    var netSecond = netSale(companyInputs.price_second, sellFee, shipToBuyer);
    var netLower = netSale(companyInputs.price_lower, sellFee, shipToBuyer);
    return pTop * netTop + pSecond * netSecond + pLower * netLower;
  }

  function sellRawToday(inputs) {
    var sellFee = inputs.sell_fee_pct / PERCENT;
    return netSale(inputs.raw_value_now, sellFee, inputs.ship_to_buyer);
  }

  // One complete tier's dated sticker cash flows, before any discounting.
  function tierPlan(company, tier, inputs) {
    var companyInputs = inputs.companies[company.key];
    var sellFee = inputs.sell_fee_pct / PERCENT;
    var quantity = cleanQuantity(inputs);
    var businessDays = turnaroundBusinessDays(tier, inputs.range_mode);
    var calendarDays = businessToCalendarDays(businessDays);
    var dayReceived = inputs.days_out;
    var dayGraded = dayReceived + calendarDays;
    var dayBack = dayGraded + inputs.days_back;
    var daySale = dayBack + inputs.sell_days;
    var feeTiming = companyInputs.fee_timing;
    var feeDay = dayGraded;
    if (feeTiming === FEE_AT_CHECKOUT) feeDay = 0;
    var shipOut = inputs.ship_out_batch / quantity;
    var shipBack = inputs.ship_back_batch / quantity;
    var sale = expectedNetSale(companyInputs, sellFee, inputs.ship_to_buyer);
    var flows = [
      { what: "ship_out", kind: "cost", day: 0, amount: shipOut },
      { what: "grading_fee", kind: "cost", day: feeDay, amount: tier.fee },
      { what: "ship_back", kind: "cost", day: dayGraded, amount: shipBack },
      { what: "sale", kind: "proceeds", day: daySale, amount: sale }
    ];
    return {
      fee_timing: feeTiming,
      turnaround_business_days: businessDays,
      turnaround_calendar_days: calendarDays,
      day_received: dayReceived,
      day_graded: dayGraded,
      day_back: dayBack,
      day_sale: daySale,
      flows: flows
    };
  }

  // Each flow's value today, and the totals by kind (costs added in order).
  function valueFlows(flows, weeklyRate) {
    var valued = [];
    var pvCosts = 0;
    var pvProceeds = 0;
    var stickerCosts = 0;
    var stickerProceeds = 0;
    for (var i = 0; i < flows.length; i++) {
      var flow = flows[i];
      var valueToday = presentValue(flow.amount, flow.day, weeklyRate);
      valued.push({
        what: flow.what, kind: flow.kind, day: flow.day, amount: flow.amount, value_today: valueToday
      });
      if (flow.kind === "cost") {
        pvCosts = pvCosts + valueToday;
        stickerCosts = stickerCosts + flow.amount;
      } else {
        pvProceeds = pvProceeds + valueToday;
        stickerProceeds = stickerProceeds + flow.amount;
      }
    }
    return {
      flows: valued,
      pv_costs: pvCosts,
      pv_proceeds: pvProceeds,
      sticker_costs: stickerCosts,
      sticker_proceeds: stickerProceeds
    };
  }

  // below_minimum: fewer cards than the level's minimum. over_cap: the
  // top-grade price is above the level's value cap; with a "moves up" cap the
  // level is not a play, with an informational cap it still is.
  function tierFlags(company, tier, inputs) {
    var companyInputs = inputs.companies[company.key];
    var quantity = cleanQuantity(inputs);
    var minimum = tier.min_cards;
    var belowMinimum = isNumber(minimum) && quantity < minimum;
    var cap = tier.max_value;
    var overCap = isNumber(cap) && companyInputs.price_top > cap;
    var flags = [];
    if (belowMinimum) flags.push(FLAG_BELOW_MINIMUM);
    if (overCap) flags.push(FLAG_OVER_CAP);
    var blockedByCap = overCap && company.cap_rule === CAP_MOVES_UP;
    var eligible = !belowMinimum && !blockedByCap;
    return { below_minimum: belowMinimum, over_cap: overCap, flags: flags, eligible: eligible };
  }

  function tierKey(company, tierIndex) {
    return company.key + "/" + String(tierIndex);
  }

  function tierLabel(company, tier, tierIndex) {
    var name = tier.name;
    if (!name) name = "row " + String(tierIndex + 1);
    return company.name + " " + name;
  }

  // Everything the page shows for one complete tier, per card.
  function evaluateTier(company, tierIndex, inputs) {
    var tier = company.tiers[tierIndex];
    var weeklyRate = inputs.weekly_rate_pct / PERCENT;
    var plan = tierPlan(company, tier, inputs);
    var valued = valueFlows(plan.flows, weeklyRate);
    var rawToday = sellRawToday(inputs);
    var npv = valued.pv_proceeds - valued.pv_costs;
    var undiscountedProfit = valued.sticker_proceeds - valued.sticker_costs;
    var feeFlow = valued.flows[1];
    var flags = tierFlags(company, tier, inputs);
    return {
      key: tierKey(company, tierIndex),
      company: company.key,
      label: tierLabel(company, tier, tierIndex),
      fee: tier.fee,
      fee_timing: plan.fee_timing,
      fee_day: feeFlow.day,
      fee_value_today: feeFlow.value_today,
      turnaround_business_days: plan.turnaround_business_days,
      turnaround_calendar_days: plan.turnaround_calendar_days,
      day_received: plan.day_received,
      day_graded: plan.day_graded,
      day_back: plan.day_back,
      day_sale: plan.day_sale,
      expected_net_sale: valued.sticker_proceeds,
      sticker_costs: valued.sticker_costs,
      pv_costs: valued.pv_costs,
      pv_proceeds: valued.pv_proceeds,
      npv: npv,
      value_of_play: npv - rawToday,
      undiscounted_profit: undiscountedProfit,
      undiscounted_value_of_play: undiscountedProfit - rawToday,
      min_cards: tier.min_cards,
      max_value: tier.max_value,
      below_minimum: flags.below_minimum,
      over_cap: flags.over_cap,
      flags: flags.flags,
      eligible: flags.eligible,
      gap_to_best: null,
      flows: valued.flows
    };
  }

  function valueOfPlayAtRate(company, tierIndex, inputs, weeklyRate) {
    var tier = company.tiers[tierIndex];
    var plan = tierPlan(company, tier, inputs);
    var valued = valueFlows(plan.flows, weeklyRate);
    var npv = valued.pv_proceeds - valued.pv_costs;
    return npv - sellRawToday(inputs);
  }

  function compareRanked(a, b) {
    if (a.value_of_play !== b.value_of_play) return b.value_of_play - a.value_of_play;
    return a.position - b.position;
  }

  // [{key, value_of_play, gap_to_best}] best first; ties keep table order.
  function rankValues(keys, values) {
    var entries = [];
    for (var position = 0; position < keys.length; position++) {
      entries.push({ key: keys[position], value_of_play: values[position], position: position });
    }
    entries.sort(compareRanked);
    var ranking = [];
    for (var i = 0; i < entries.length; i++) {
      var gap = entries[i].value_of_play - entries[0].value_of_play;
      ranking.push({ key: entries[i].key, value_of_play: entries[i].value_of_play, gap_to_best: gap });
    }
    return ranking;
  }

  function rateGridPcts() {
    var pcts = [];
    for (var step = 0; step <= RATE_GRID_STEPS; step++) pcts.push(step * RATE_GRID_STEP_PCT);
    return pcts;
  }

  // The eligible tiers ranked at one grid rate (chart 2's tooltip).
  function rankAtRate(rateGrid, gridIndex) {
    var keys = rateGrid.keys;
    var values = [];
    for (var i = 0; i < keys.length; i++) values.push(rateGrid.value_of_play[keys[i]][gridIndex]);
    return rankValues(keys, values);
  }

  // Runs of grid rates with one winner: a tier key, or "raw" when no
  // eligible tier beats selling raw today at that rate.
  function bestByRate(rateGrid) {
    var rates = rateGrid.rates_pct;
    var keys = rateGrid.keys;
    var segments = [];
    for (var gridIndex = 0; gridIndex < rates.length; gridIndex++) {
      var winner = RAW_KEY;
      var bestValue = 0;
      for (var k = 0; k < keys.length; k++) {
        var value = rateGrid.value_of_play[keys[k]][gridIndex];
        if (value > bestValue) {
          winner = keys[k];
          bestValue = value;
        }
      }
      var last = segments.length ? segments[segments.length - 1] : null;
      if (last && last.winner === winner) {
        last.to_pct = rates[gridIndex];
      } else {
        segments.push({ winner: winner, from_pct: rates[gridIndex], to_pct: rates[gridIndex] });
      }
    }
    return segments;
  }

  // Every complete tier of every company for one card, plus the ranking.
  // A blank or half-filled row is skipped. A company whose odds add up to
  // more than 100% is listed in skipped_companies. Flagged tiers stay in
  // "tiers"; only eligible tiers enter the ranking and the rate grid.
  function evaluate(inputs, companies) {
    var rawToday = sellRawToday(inputs);
    var tiers = [];
    var skippedCompanies = [];
    var eligibleKeys = [];
    var eligibleValues = [];
    var eligibleSources = [];
    for (var c = 0; c < companies.length; c++) {
      var company = companies[c];
      var companyInputs = inputs.companies[company.key];
      var validOdds = oddsAreValid(companyInputs);
      var tierRows = company.tiers;
      var completeCount = 0;
      for (var tierIndex = 0; tierIndex < tierRows.length; tierIndex++) {
        if (!tierIsComplete(tierRows[tierIndex])) continue;
        completeCount += 1;
        if (!validOdds) continue;
        var result = evaluateTier(company, tierIndex, inputs);
        tiers.push(result);
        if (result.eligible) {
          eligibleKeys.push(result.key);
          eligibleValues.push(result.value_of_play);
          eligibleSources.push({ company: company, tierIndex: tierIndex });
        }
      }
      if (completeCount > 0 && !validOdds) skippedCompanies.push(company.key);
    }

    var ranking = rankValues(eligibleKeys, eligibleValues);
    var gapByKey = {};
    for (var r = 0; r < ranking.length; r++) gapByKey[ranking[r].key] = ranking[r].gap_to_best;
    for (var t = 0; t < tiers.length; t++) {
      if (Object.prototype.hasOwnProperty.call(gapByKey, tiers[t].key)) {
        tiers[t].gap_to_best = gapByKey[tiers[t].key];
      }
    }

    var quantity = cleanQuantity(inputs);
    var best = null;
    var bestValue = null;
    var bestBatchValue = null;
    var play = PLAY_SELL_RAW;
    if (ranking.length) {
      best = ranking[0].key;
      bestValue = ranking[0].value_of_play;
      bestBatchValue = bestValue * quantity;
      if (bestValue > 0) play = PLAY_GRADE;
    }

    var ratesPct = rateGridPcts();
    var curves = {};
    for (var e = 0; e < eligibleKeys.length; e++) {
      var source = eligibleSources[e];
      var curve = [];
      for (var g = 0; g < ratesPct.length; g++) {
        var weeklyRate = ratesPct[g] / PERCENT;
        curve.push(valueOfPlayAtRate(source.company, source.tierIndex, inputs, weeklyRate));
      }
      curves[eligibleKeys[e]] = curve;
    }
    var rateGrid = { rates_pct: ratesPct, keys: eligibleKeys, value_of_play: curves };

    return {
      weekly_rate_pct: inputs.weekly_rate_pct,
      annual_rate_pct: annualPctFromWeekly(inputs.weekly_rate_pct),
      quantity: quantity,
      sell_raw_today: rawToday,
      tiers: tiers,
      skipped_companies: skippedCompanies,
      ranking: ranking,
      best: best,
      best_value_of_play: bestValue,
      best_batch_value_of_play: bestBatchValue,
      play: play,
      rate_grid: rateGrid,
      best_by_rate: bestByRate(rateGrid)
    };
  }

  var api = {
    isNumber: isNumber,
    tierDayRange: tierDayRange,
    tierIsComplete: tierIsComplete,
    turnaroundBusinessDays: turnaroundBusinessDays,
    businessToCalendarDays: businessToCalendarDays,
    presentValue: presentValue,
    weeklyPctFromAnnual: weeklyPctFromAnnual,
    annualPctFromWeekly: annualPctFromWeekly,
    cleanQuantity: cleanQuantity,
    oddsAreValid: oddsAreValid,
    lowerOddsPct: lowerOddsPct,
    netSale: netSale,
    expectedNetSale: expectedNetSale,
    sellRawToday: sellRawToday,
    tierPlan: tierPlan,
    valueFlows: valueFlows,
    tierFlags: tierFlags,
    tierKey: tierKey,
    tierLabel: tierLabel,
    evaluateTier: evaluateTier,
    valueOfPlayAtRate: valueOfPlayAtRate,
    rankValues: rankValues,
    rateGridPcts: rateGridPcts,
    rankAtRate: rankAtRate,
    bestByRate: bestByRate,
    evaluate: evaluate
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.PlayCalc = api;
  }
})(typeof window !== "undefined" ? window : this);
