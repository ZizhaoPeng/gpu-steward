(function () {
  "use strict";

  var dateInput = document.getElementById("report-date");
  var dateForm = document.getElementById("date-form");
  var timeline = document.getElementById("timeline");
  var detail = document.getElementById("detail");
  var reportMeta = document.getElementById("report-meta");
  var errorBox = document.getElementById("error");
  var timezone = "Asia/Singapore";

  function pad(value) { return String(value).padStart(2, "0"); }
  function todayInSingapore() {
    var parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: timezone, year: "numeric", month: "2-digit", day: "2-digit"
    }).formatToParts(new Date());
    var values = {};
    parts.forEach(function (part) { values[part.type] = part.value; });
    return values.year + "-" + values.month + "-" + values.day;
  }
  function escapeTime(value) {
    if (value === null || value === undefined || value === "") return "—";
    var parsed = typeof value === "number" ? new Date(value * 1000) : new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    var parts = new Intl.DateTimeFormat("en-GB", {
      timeZone: timezone, hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
    }).formatToParts(parsed);
    var values = {};
    parts.forEach(function (part) { values[part.type] = part.value; });
    return values.hour + ":" + values.minute + ":" + values.second;
  }
  function dayFraction(value, reportDate) {
    if (typeof value === "number") {
      var parsed = new Date(value * 1000);
      var parts = new Intl.DateTimeFormat("en-CA", {
        timeZone: timezone, year: "numeric", month: "2-digit", day: "2-digit",
        hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
      }).formatToParts(parsed);
      var values = {};
      parts.forEach(function (part) { values[part.type] = part.value; });
      var currentDate = values.year + "-" + values.month + "-" + values.day;
      if (currentDate < reportDate) return 0;
      if (currentDate > reportDate) return 1;
      return (Number(values.hour) * 3600 + Number(values.minute) * 60 + Number(values.second)) / 86400;
    }
    var parsedDate = new Date(value);
    if (Number.isNaN(parsedDate.getTime())) return 0;
    return dayFraction(parsedDate.getTime() / 1000, reportDate);
  }
  function seconds(value) { return Math.max(0, Number(value) || 0); }
  function durationLabel(value) {
    var total = Math.round(seconds(value));
    var hours = Math.floor(total / 3600);
    var minutes = Math.floor((total % 3600) / 60);
    var secs = total % 60;
    if (hours) return hours + "小时 " + pad(minutes) + "分";
    if (minutes) return minutes + "分 " + pad(secs) + "秒";
    return secs + "秒";
  }
  function summaryLabel(value) { return durationLabel(value); }
  function segmentStart(segment) { return segment.start !== undefined ? segment.start : segment.start_at; }
  function segmentEnd(segment) { return segment.end !== undefined ? segment.end : segment.end_at; }
  function segmentDuration(segment) {
    if (segment.duration_seconds !== undefined) return segment.duration_seconds;
    return seconds(segmentEnd(segment)) - seconds(segmentStart(segment));
  }
  function segmentClass(lane, segment) {
    var state = String(segment.state || segment.phase || "active-unspecified").toLowerCase().replace(/_/g, "-");
    var classes = ["segment", lane.kind === "gpu" ? "gpu-" + state : "codex"];
    if (segment.source === "inferred") classes.push("inferred");
    return classes.join(" ");
  }
  function text(value) { return value === null || value === undefined || value === "" ? "—" : String(value); }
  function field(label, value) {
    var wrapper = document.createElement("div");
    wrapper.className = "detail-item";
    var dt = document.createElement("dt"); dt.textContent = label;
    var dd = document.createElement("dd"); dd.textContent = text(value);
    wrapper.appendChild(dt); wrapper.appendChild(dd); return wrapper;
  }
  function showDetail(lane, segment, reportDate) {
    detail.textContent = "";
    var heading = document.createElement("h3");
    heading.textContent = text(segment.label || segment.task_name || segment.phase || lane.label);
    var grid = document.createElement("dl"); grid.className = "detail-grid";
    grid.appendChild(field("泳道", lane.label));
    grid.appendChild(field("任务", segment.task_name || segment.task || "—"));
    grid.appendChild(field("来源", segment.source || "—"));
    grid.appendChild(field("置信度", segment.confidence === undefined ? "—" : Number(segment.confidence).toFixed(2)));
    grid.appendChild(field("状态", segment.state || segment.phase || "—"));
    grid.appendChild(field("开始", escapeTime(segmentStart(segment))));
    grid.appendChild(field("结束", escapeTime(segmentEnd(segment))));
    grid.appendChild(field("时间跨度", durationLabel(segment.duration_seconds !== undefined ? segment.duration_seconds : segment.duration)));
    if (seconds(segment.gap_seconds) > 0) {
      grid.appendChild(field("实际观察", durationLabel(segment.observed_seconds)));
      grid.appendChild(field("已合并短间隙", text(segment.gap_count) + " 段 · " + durationLabel(segment.gap_seconds)));
    }
    detail.appendChild(heading); detail.appendChild(grid);
  }
  function renderAxis() {
    var axis = document.createElement("div"); axis.className = "axis";
    var blank = document.createElement("div"); axis.appendChild(blank);
    var track = document.createElement("div"); track.className = "axis-track";
    for (var hour = 0; hour <= 24; hour += 4) {
      var label = document.createElement("span"); label.className = "axis-label";
      label.style.left = (hour / 24 * 100) + "%"; label.textContent = pad(hour % 24) + ":00";
      track.appendChild(label);
    }
    axis.appendChild(track); return axis;
  }
  function renderReport(report) {
    timezone = report.timezone || timezone;
    dateInput.value = report.date;
    var mergeMinutes = Math.round(seconds(report.display_merge_gap_seconds) / 60);
    reportMeta.textContent = report.date + " · " + timezone + (mergeMinutes ? " · 合并≤" + mergeMinutes + "分钟短抖动" : "");
    Object.keys(report.summary || {}).forEach(function (key) {
      var target = document.querySelector('[data-summary="' + key + '"]');
      if (target) target.textContent = summaryLabel(report.summary[key]);
    });
    timeline.textContent = "";
    timeline.appendChild(renderAxis());
    if (!report.lanes || !report.lanes.length) {
      var empty = document.createElement("div"); empty.className = "empty"; empty.textContent = "该日没有可展示的活动。";
      timeline.appendChild(empty); return;
    }
    report.lanes.forEach(function (lane) {
      var row = document.createElement("div"); row.className = "lane";
      var label = document.createElement("div"); label.className = "lane-label";
      var name = document.createElement("strong"); name.textContent = lane.label;
      var kind = document.createElement("span"); kind.textContent = lane.kind;
      label.appendChild(name); label.appendChild(kind);
      var track = document.createElement("div"); track.className = "lane-track";
      (lane.segments || []).forEach(function (segment) {
        var start = Math.max(0, Math.min(1, dayFraction(segmentStart(segment), report.date)));
        var end = Math.max(start, Math.min(1, dayFraction(segmentEnd(segment), report.date)));
        var bar = document.createElement("button"); bar.type = "button"; bar.className = segmentClass(lane, segment);
        if (seconds(segment.gap_seconds) > 0) bar.className += " merged-gap";
        bar.style.left = (start * 100) + "%"; bar.style.width = Math.max(.18, (end - start) * 100) + "%";
        bar.textContent = text(segment.label || segment.task_name || segment.phase || segment.state || "活动");
        bar.title = bar.textContent + (seconds(segment.gap_seconds) > 0 ? " · 含" + text(segment.gap_count) + "段短间隙" : "");
        bar.addEventListener("click", function () { showDetail(lane, segment, report.date); });
        track.appendChild(bar);
      });
      row.appendChild(label); row.appendChild(track); timeline.appendChild(row);
    });
  }
  function loadReport(reportDate) {
    errorBox.hidden = true; timeline.textContent = "";
    var loading = document.createElement("div"); loading.className = "loading"; loading.textContent = "正在读取本机日报…"; timeline.appendChild(loading);
    fetch("/api/report?date=" + encodeURIComponent(reportDate), { credentials: "same-origin" })
      .then(function (response) { return response.json().then(function (payload) { if (!response.ok) throw new Error(payload.error || "日报读取失败"); return payload; }); })
      .then(renderReport)
      .catch(function (reason) { timeline.textContent = ""; errorBox.textContent = reason.message || "日报读取失败"; errorBox.hidden = false; });
  }
  dateInput.value = todayInSingapore();
  dateForm.addEventListener("submit", function (event) { event.preventDefault(); loadReport(dateInput.value); });
  loadReport(dateInput.value);
}());
