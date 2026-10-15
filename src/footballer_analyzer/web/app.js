const element = (id) => document.getElementById(id);
const state = { runId: null, data: null, frame: 0, selected: null, image: null, playing: false, pointMode: false, roiMode: false, roi: [], points: [], requestId: 0, runRequestId: 0, frameRows: new Map(), detectionRows: new Map(), motionRows: new Map(), draftDirty: false };
const colors = { home: "#28cfac", away: "#f9b26a", cluster_0: "#28cfac", cluster_1: "#f9b26a", unknown: "#a6b3b2" };
const reasons = { unbound: "身份尚未确认", excluded: "人工排除", outside_roi: "超出可靠统计区域", outside_pitch: "投影超出球场", clipped_feet: "脚点被画面裁切", uncalibrated: "尚未标定", outside_calibration_window: "超出标定有效区间", invalid_projection: "投影无效", speed_jump: "异常速度跳变，已切段" };

function notify(message, error = false) {
  element("notice").textContent = message;
  element("notice").classList.toggle("error", error);
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "请求失败");
  return result;
}

function api(action = "") {
  return `/api/runs/${encodeURIComponent(state.runId)}${action ? `/${action}` : ""}`;
}

function make(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function option(value, label) {
  const node = make("option", label);
  node.value = value;
  return node;
}

function bindingFor(row) {
  return state.data.bindings.bindings.find((entry) => entry.track_id === row.track_id && entry.start_frame <= row.frame_index && row.frame_index < entry.end_frame);
}

function project(row) {
  const calibration = state.data.calibration;
  const binding = bindingFor(row);
  const base = { row, binding, color: colors[binding?.team_id || row.team_id] || colors.unknown, position: null, reason: null };
  if (!calibration) return { ...base, reason: "uncalibrated" };
  if (row.frame_index < calibration.start_frame || row.frame_index >= calibration.end_frame) return { ...base, reason: "outside_calibration_window" };
  const footX = (row.bbox_x1 + row.bbox_x2) / 2;
  const footY = row.bbox_y2;
  const matrix = calibration.homography;
  const divisor = matrix[2][0] * footX + matrix[2][1] * footY + matrix[2][2];
  if (Math.abs(divisor) < 1e-8) return { ...base, reason: "invalid_projection" };
  const worldX = (matrix[0][0] * footX + matrix[0][1] * footY + matrix[0][2]) / divisor;
  const worldY = (matrix[1][0] * footX + matrix[1][1] * footY + matrix[1][2]) / divisor;
  if (!Number.isFinite(worldX) || !Number.isFinite(worldY)) return { ...base, reason: "invalid_projection" };
  base.position = [worldX, worldY];
  if (footY >= calibration.height - 2) base.reason = "clipped_feet";
  else if (!pointInPolygon(footX, footY, calibration.roi)) base.reason = "outside_roi";
  else if (worldX < 0 || worldX > calibration.pitch_length_m || worldY < 0 || worldY > calibration.pitch_width_m) base.reason = "outside_pitch";
  else if (binding?.action === "exclude") base.reason = "excluded";
  else if (!binding) base.reason = "unbound";
  const motion = state.motionRows.get(`${row.frame_index}:${row.track_id}`);
  if (motion && !motion.valid && !state.data.latest_report?.stale) base.reason = motion.reason;
  return base;
}

function pointInPolygon(horizontal, vertical, polygon) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [currentX, currentY] = polygon[index];
    const [previousX, previousY] = polygon[previous];
    const cross = (horizontal - currentX) * (previousY - currentY) - (vertical - currentY) * (previousX - currentX);
    if (Math.abs(cross) < 1e-6 && horizontal >= Math.min(currentX, previousX) && horizontal <= Math.max(currentX, previousX) && vertical >= Math.min(currentY, previousY) && vertical <= Math.max(currentY, previousY)) return true;
    if ((currentY > vertical) !== (previousY > vertical) && horizontal < (previousX - currentX) * (vertical - currentY) / (previousY - currentY) + currentX) inside = !inside;
  }
  return inside;
}

function svgNode(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function pitchShape(tag, attributes, text) {
  element("pitch").append(svgNode(tag, attributes, text));
}

function drawPitch(observations) {
  const calibration = state.data.calibration;
  const length = calibration?.pitch_length_m || state.data.manifest.config.pitch_length_m;
  const width = calibration?.pitch_width_m || state.data.manifest.config.pitch_width_m;
  const pitch = element("pitch");
  pitch.replaceChildren();
  pitch.setAttribute("viewBox", `-4 -4 ${length + 8} ${width + 8}`);
  pitchShape("rect", { x: -4, y: -4, width: length + 8, height: width + 8, fill: "#214e37" });
  for (let index = 0; index < 10; index++) pitchShape("rect", { x: index * length / 10, y: 0, width: length / 10, height: width, fill: index % 2 ? "#2d5a3d" : "#285539" });
  const line = { fill: "none", stroke: "#83a17c", "stroke-width": 0.32 };
  pitchShape("rect", { x: 0, y: 0, width: length, height: width, ...line });
  pitchShape("line", { x1: length / 2, y1: 0, x2: length / 2, y2: width, ...line });
  pitchShape("circle", { cx: length / 2, cy: width / 2, r: Math.min(9.15, width / 5), ...line });
  pitchShape("circle", { cx: length / 2, cy: width / 2, r: 0.6, fill: "#83a17c" });
  for (const side of [0, 1]) {
    const boxLength = Math.min(16.5, length * 0.16);
    const boxWidth = Math.min(40.32, width * 0.65);
    pitchShape("rect", { x: side ? length - boxLength : 0, y: (width - boxWidth) / 2, width: boxLength, height: boxWidth, ...line });
    pitchShape("rect", { x: side ? length - boxLength / 3 : 0, y: width * 0.36, width: boxLength / 3, height: width * 0.28, ...line });
  }
  if (!calibration) {
    pitchShape("text", { x: length / 2, y: width / 2 + 18, "text-anchor": "middle", fill: "#b6ccab", "font-size": 3 }, "先在“场地标定”中建立对应点");
  }
  for (const observation of observations) {
    if (!observation.position || observation.reason === "excluded") continue;
    const [horizontal, vertical] = observation.position;
    if (horizontal < 0 || horizontal > length || vertical < 0 || vertical > width) continue;
    const selected = observation.row.track_id === state.selected;
    const group = svgNode("g", { class: "pitch-point", "data-track-id": observation.row.track_id, tabindex: 0, role: "button", "aria-label": `选中轨迹 ${observation.row.track_id}` });
    if (selected) group.append(svgNode("circle", { cx: horizontal, cy: vertical, r: 2.8, fill: "none", stroke: "white", "stroke-width": 0.4 }));
    group.append(svgNode("circle", { cx: horizontal, cy: vertical, r: selected ? 1.8 : 1.4, fill: observation.reason ? "#214e37" : observation.color, stroke: observation.color, "stroke-width": 0.6 }));
    group.append(svgNode("text", { x: horizontal + 2, y: vertical - 1.2, fill: "#e5efdd", "font-size": 2.6, "font-family": "monospace" }, observation.row.track_id));
    group.addEventListener("click", () => selectTrack(observation.row.track_id));
    group.addEventListener("keydown", (event) => { if (event.key === "Enter") selectTrack(observation.row.track_id); });
    pitch.append(group);
  }
  pitchShape("text", { x: 0, y: width + 2.8, fill: "#89a47e", "font-size": 1.7 }, `0,0  → x / ${length} × ${width} m`);
}

function detectionKey(row) {
  return [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2, row.confidence].join(":");
}

function drawVideo(observations) {
  const canvas = element("video-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (state.image) context.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  const rawDetections = state.detectionRows.get(state.frame) || [];
  const matched = new Set(observations.map((observation) => detectionKey(observation.row)));
  const unmatched = rawDetections.filter((row) => !matched.has(detectionKey(row)));
  element("detection-summary").textContent = state.data.detections == null ? "旧运行没有原始检测缓存；重新执行 track 后可区分漏检与关联失败。" : `检测 ${rawDetections.length} · 已关联 ${observations.length} · 未关联 ${unmatched.length}（包含场外人员与噪声）。红色虚线：有检测但未形成轨迹；可见人物无任何框：检测阶段漏检。`;
  if (element("show-unmatched").checked) {
    context.strokeStyle = "#ff7575"; context.fillStyle = "#ff7575";
    context.lineWidth = 2; context.setLineDash([6, 4]); context.font = "17px sans-serif";
    for (const row of unmatched) {
      context.strokeRect(row.bbox_x1, row.bbox_y1, row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1);
      context.fillText(`未关联 ${row.confidence.toFixed(2)}`, row.bbox_x1, Math.max(18, row.bbox_y1 - 5));
    }
    context.setLineDash([]);
  }
  const calibration = state.data.calibration;
  if (calibration && state.frame >= calibration.start_frame && state.frame < calibration.end_frame) {
    context.strokeStyle = "#d4eac488";
    context.lineWidth = 2;
    context.setLineDash([10, 8]);
    context.beginPath();
    calibration.roi.forEach(([horizontal, vertical], index) => index ? context.lineTo(horizontal, vertical) : context.moveTo(horizontal, vertical));
    context.closePath(); context.stroke(); context.setLineDash([]);
    const inverse = invert3(calibration.homography);
    if (inverse) {
      const length = calibration.pitch_length_m, width = calibration.pitch_width_m;
      const lines = [[[0, 0], [length, 0]], [[0, width], [length, width]], [[length / 2, 0], [length / 2, width]]];
      const circle = Array.from({ length: 65 }, (_, index) => [length / 2 + 9.15 * Math.cos(index / 64 * Math.PI * 2), width / 2 + 9.15 * Math.sin(index / 64 * Math.PI * 2)]);
      context.strokeStyle = "#a3f4ce99";
      for (const points of [...lines, circle]) {
        context.beginPath();
        points.forEach(([horizontal, vertical], index) => {
          const divisor = inverse[2][0] * horizontal + inverse[2][1] * vertical + inverse[2][2];
          const pixelX = (inverse[0][0] * horizontal + inverse[0][1] * vertical + inverse[0][2]) / divisor;
          const pixelY = (inverse[1][0] * horizontal + inverse[1][1] * vertical + inverse[1][2]) / divisor;
          if (index) context.lineTo(pixelX, pixelY); else context.moveTo(pixelX, pixelY);
        });
        context.stroke();
      }
    }
  }
  if (element("show-boxes").checked) for (const observation of observations) {
    const row = observation.row;
    const selected = row.track_id === state.selected;
    context.strokeStyle = observation.reason === "excluded" ? "#fb7474" : observation.color;
    context.lineWidth = selected ? 5 : 2;
    context.strokeRect(row.bbox_x1, row.bbox_y1, row.bbox_x2 - row.bbox_x1, row.bbox_y2 - row.bbox_y1);
    const label = `#${row.track_id}${observation.binding?.player_id ? ` · ${observation.binding.player_id}` : ""}${element("show-unmatched").checked ? ` ${row.confidence.toFixed(2)}` : ""}`;
    context.font = "19px sans-serif";
    const labelWidth = context.measureText(label).width + 12;
    context.fillStyle = selected ? "#f3ffed" : "#092d25dd";
    context.fillRect(row.bbox_x1, Math.max(0, row.bbox_y1 - 27), labelWidth, 25);
    context.fillStyle = selected ? "#123d2f" : observation.color;
    context.fillText(label, row.bbox_x1 + 6, Math.max(20, row.bbox_y1 - 8));
    context.beginPath(); context.arc((row.bbox_x1 + row.bbox_x2) / 2, row.bbox_y2, selected ? 6 : 4, 0, Math.PI * 2); context.fill();
  }
  if (state.pointMode || !element("calibration-tab").hidden) state.points.forEach((point, index) => {
    context.fillStyle = "#ffd58f"; context.beginPath(); context.arc(point.image[0], point.image[1], 7, 0, Math.PI * 2); context.fill();
    context.font = "bold 24px sans-serif"; context.fillText(String(index + 1), point.image[0] + 10, point.image[1] - 10);
  });
  if (!element("calibration-tab").hidden && state.roi.length) {
    context.strokeStyle = "#ffe48e"; context.lineWidth = 3; context.beginPath();
    state.roi.forEach(([horizontal, vertical], index) => index ? context.lineTo(horizontal, vertical) : context.moveTo(horizontal, vertical));
    if (!state.roiMode) context.closePath();
    context.stroke();
  }
}

function invert3(matrix) {
  const determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  if (Math.abs(determinant) < 1e-12) return null;
  const inverse = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) inverse[column][row] = (matrix[(row + 1) % 3][(column + 1) % 3] * matrix[(row + 2) % 3][(column + 2) % 3] - matrix[(row + 1) % 3][(column + 2) % 3] * matrix[(row + 2) % 3][(column + 1) % 3]) / determinant;
  return inverse;
}

function displayName(playerId) {
  return state.data.players.find((player) => player.player_id === playerId)?.display_name || playerId;
}

function draw() {
  if (!state.data) return;
  element("export-download").hidden = true;
  const observations = (state.frameRows.get(state.frame) || []).map(project);
  drawVideo(observations);
  drawPitch(observations);
  element("stat-observations").textContent = observations.length;
  element("visible-count").textContent = observations.length;
  element("stat-bound").textContent = observations.filter((observation) => observation.binding?.action === "bind").length;
  element("track-list").replaceChildren(...observations.map((observation) => {
    const button = make("button", undefined, `track-item${observation.row.track_id === state.selected ? " selected" : ""}`);
    const marker = make("span", undefined, "marker"); marker.style.background = observation.color;
    button.append(marker, make("span", observation.binding?.player_id ? displayName(observation.binding.player_id) : `轨迹 #${observation.row.track_id}`), make("small", `#${observation.row.track_id}`));
    button.addEventListener("click", () => selectTrack(observation.row.track_id));
    return button;
  }));
  const selected = observations.find((observation) => observation.row.track_id === state.selected);
  element("selected-name").textContent = selected?.binding?.player_id ? displayName(selected.binding.player_id) : state.selected ? `轨迹 #${state.selected}` : "选择一名球员";
  element("selected-id").textContent = state.selected ? `#${state.selected}` : "—";
  element("selected-position").textContent = selected?.position ? selected.position.map((coordinate) => coordinate.toFixed(2)).join(" / ") : "—";
  element("selected-reason").textContent = selected ? (selected.reason ? reasons[selected.reason] || selected.reason : "当前点有效；报告仅累加连续有效观测。") : state.selected ? "该轨迹在当前帧没有观测。" : "从画面、场地圆点或左侧列表选择。";
  const manifest = state.data.manifest;
  element("frame-label").textContent = `FRAME ${String(state.frame).padStart(5, "0")} / ${manifest.end_frame - 1}`;
  element("frame-slider").value = state.frame;
  element("frame-number").value = state.frame;
  element("time-label").textContent = `${((state.frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} / ${((manifest.end_frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} s`;
}

async function showFrame(frameIndex) {
  if (!state.data) return;
  const manifest = state.data.manifest;
  const bounded = Math.min(manifest.end_frame - 1, Math.max(manifest.start_frame, frameIndex));
  const aligned = manifest.start_frame + Math.floor((bounded - manifest.start_frame) / manifest.config.stride) * manifest.config.stride;
  const requestId = ++state.requestId;
  const image = new Image();
  const loaded = new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error("无法读取视频帧，请检查源视频是否仍存在")); });
  image.src = api(`frames/${aligned}`);
  await loaded;
  if (requestId !== state.requestId) return;
  state.frame = aligned; state.image = image; draw();
}

function selectTrack(trackId) {
  state.selected = trackId;
  element("binding-track").value = trackId;
  const catalog = state.data.catalog.find((entry) => entry.track_id === trackId);
  if (catalog) {
    element("binding-start").value = catalog.start_frame;
    element("binding-end").value = catalog.end_frame;
  }
  draw();
}

function stopPlayback() {
  state.playing = false; element("play").textContent = "▶"; element("play").setAttribute("aria-label", "播放");
}

async function play() {
  if (state.playing) return stopPlayback();
  if (!state.data) return;
  state.playing = true; element("play").textContent = "Ⅱ"; element("play").setAttribute("aria-label", "暂停");
  while (state.playing) {
    const manifest = state.data.manifest;
    const step = Math.max(manifest.config.stride, Math.round(manifest.video.fps / 10 * Number(element("playback-rate").value) / manifest.config.stride) * manifest.config.stride);
    if (state.frame + step >= manifest.end_frame) { stopPlayback(); break; }
    try { await showFrame(state.frame + step); } catch (error) { notify(error.message, true); stopPlayback(); }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

function renderBindings() {
  element("binding-player").replaceChildren(option("", "选择球员档案"), ...state.data.players.map((player) => option(player.player_id, `${player.display_name} / ${player.player_id}`)));
  const entries = state.data.bindings.bindings;
  const nodes = entries.map((entry, index) => {
    const row = make("tr");
    row.append(make("td", `#${entry.track_id}`), make("td", `[${entry.start_frame}, ${entry.end_frame})`), make("td", entry.player_id ? displayName(entry.player_id) : "—"), make("td", entry.action === "exclude" ? "已排除" : entry.team_id));
    const cell = make("td"); const remove = make("button", "移除", "remove-button");
    remove.addEventListener("click", async () => {
      try { await request(api("bindings"), { run_id: state.runId, bindings: entries.filter((_, entryIndex) => entryIndex !== index) }); await loadRun(state.runId, true); notify("区间已移除；重新生成报告可更新统计。"); } catch (error) { notify(error.message, true); }
    });
    cell.append(remove); row.append(cell); return row;
  });
  if (!nodes.length) { const row = make("tr"); const cell = make("td", "尚未绑定。先建立档案，再从画面选择轨迹。", "empty-row"); cell.colSpan = 5; row.append(cell); nodes.push(row); }
  element("binding-table").replaceChildren(...nodes);
}

function markDraftDirty() {
  state.draftDirty = true;
  element("calibration-reviewed").checked = false;
}

function renderCalibrationPoints() {
  element("calibration-points").replaceChildren(...state.points.map((point, index) => {
    const row = make("tr"); row.append(make("td", String(index + 1)), make("td", point.image.map(Math.round).join(" / ")));
    const targets = make("td");
    point.pitch.forEach((coordinate, axis) => {
      const input = make("input"); input.type = "number"; input.step = "0.01"; input.value = coordinate; input.setAttribute("aria-label", `点 ${index + 1} 的 ${axis ? "y" : "x"} 米制坐标`);
      input.addEventListener("input", () => { point.pitch[axis] = Number(input.value); markDraftDirty(); }); targets.append(input);
    });
    const cell = make("td"); const remove = make("button", "×", "remove-button");
    remove.addEventListener("click", () => { state.points.splice(index, 1); markDraftDirty(); renderCalibrationPoints(); draw(); });
    cell.append(remove); row.append(targets, cell); return row;
  }));
}

function renderStatistics() {
  const report = state.data.latest_report;
  element("open-report").hidden = !report;
  if (report) element("open-report").href = `/reports/${state.runId}/${report.report_id}/report.html`;
  element("report-note").textContent = report ? report.stale ? "身份、标定或档案已修改，以下为旧版结果，请重新生成报告。" : `已生成 ${report.report_id}。结果为已分析片段的有效观测。` : "完成场地标定确认和球员绑定后，即可生成报告。";
  element("statistics-table").replaceChildren(...(report?.summary.players || []).map((summary) => {
    const row = make("tr");
    row.append(make("td", displayName(summary.player_id)), make("td", summary.distance_m === null ? "数据不足" : `${summary.distance_m.toFixed(2)} m`), make("td", `${summary.valid_seconds.toFixed(2)} s`), make("td", `${(summary.coverage * 100).toFixed(1)}%`), make("td", summary.avg_speed_mps === null ? "—" : `${(summary.avg_speed_mps * 3.6).toFixed(2)} km/h`));
    row.addEventListener("click", () => {
      const chart = make("img"); chart.alt = `${displayName(summary.player_id)} 的速度曲线与热图`;
      const encoded = [...new TextEncoder().encode(summary.player_id)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
      chart.src = `/reports/${state.runId}/${report.report_id}/${encoded}.png`; element("player-chart").replaceChildren(chart);
    }); return row;
  }));
  element("player-chart").replaceChildren();
}

async function loadRun(runId, preserve = false) {
  stopPlayback();
  const oldFrame = state.frame;
  state.requestId++;
  state.runId = runId;
  const runRequestId = ++state.runRequestId;
  const data = await request(api());
  if (runRequestId !== state.runRequestId) return;
  state.data = data;
  state.frameRows.clear(); state.detectionRows.clear(); state.motionRows.clear();
  for (const row of data.tracks) { if (!state.frameRows.has(row.frame_index)) state.frameRows.set(row.frame_index, []); state.frameRows.get(row.frame_index).push(row); }
  for (const row of data.detections || []) { if (!state.detectionRows.has(row.frame_index)) state.detectionRows.set(row.frame_index, []); state.detectionRows.get(row.frame_index).push(row); }
  element("show-unmatched").disabled = data.detections == null;
  if (data.detections == null) element("show-unmatched").checked = false;
  for (const row of data.latest_report?.motion || []) state.motionRows.set(`${row.frame_index}:${row.track_id}`, row);
  const manifest = data.manifest;
  const calibration = data.calibration;
  const slider = element("frame-slider"); slider.min = manifest.start_frame; slider.max = manifest.end_frame - 1; slider.step = manifest.config.stride;
  element("frame-number").min = manifest.start_frame; element("frame-number").max = manifest.end_frame - 1; element("frame-number").step = manifest.config.stride;
  element("video-canvas").width = manifest.video.width; element("video-canvas").height = manifest.video.height;
  element("source-description").textContent = `${manifest.config.clip_id} · ${manifest.config.match_id} · ${data.catalog.length} 条轨迹`;
  element("video-spec").textContent = `${manifest.video.width}×${manifest.video.height} · ${manifest.video.fps.toFixed(0)} FPS`;
  element("stat-duration").textContent = `${((manifest.end_frame - manifest.start_frame) / manifest.video.fps).toFixed(2)} s`;
  element("stat-frame-info").textContent = `源帧 [${manifest.start_frame}, ${manifest.end_frame}) · stride ${manifest.config.stride}`;
  element("stat-calibration").textContent = calibration ? calibration.reviewed ? "已确认" : "草稿预览" : "未标定";
  element("calibration-detail").textContent = calibration ? `${calibration.image_points.length} 对点 · 拟合残差 ${calibration.fit_rmse_m.toFixed(3)} m` : "先建立图像点与场地对应关系";
  element("pitch-mode").textContent = calibration ? calibration.reviewed ? "米制投影" : "草稿 · 未确认" : "等待标定";
  element("pitch-length").value = calibration?.pitch_length_m || manifest.config.pitch_length_m;
  element("pitch-width").value = calibration?.pitch_width_m || manifest.config.pitch_width_m;
  element("calibration-reference").value = calibration?.reference_frame ?? manifest.start_frame;
  element("calibration-start").value = calibration?.start_frame ?? manifest.start_frame;
  element("calibration-end").value = calibration?.end_frame ?? manifest.end_frame;
  element("calibration-reviewed").checked = calibration?.reviewed || false;
  element("calibration-fit").textContent = calibration ? `拟合点残差 ${calibration.fit_rmse_m.toFixed(3)} m，仅用于检查点的拟合情况，不能证明轨迹精度。${calibration.note || ""}` : "尚无拟合结果。";
  state.points = calibration ? calibration.image_points.map((image, index) => ({ image: [...image], pitch: [...calibration.pitch_points[index]] })) : [];
  state.roi = calibration ? calibration.roi.map((point) => [...point]) : [];
  state.draftDirty = false;
  if (!preserve) { state.selected = null; element("binding-start").value = manifest.start_frame; element("binding-end").value = manifest.end_frame; }
  renderBindings(); renderCalibrationPoints(); renderStatistics();
  await showFrame(preserve ? oldFrame : manifest.start_frame);
  notify(calibration ? calibration.reviewed ? "运行已加载。点击球员，对照原图脚点与场地位置；有问题可按区间修正身份。" : "当前为演示标定草稿，场地位置仅供核对。检查白线、尺寸和可靠区域后，再确认生成报告。" : "轨迹已加载。先绑定球员，并在“场地标定”中点选对应点；米制统计在标定确认后启用。");
}

async function initialize() {
  const runs = await request("/api/runs");
  const completed = runs.filter((run) => run.status === "complete");
  element("run-select").replaceChildren(...completed.map((run) => option(run.run_id, `${run.clip_id} · ${run.run_id.slice(-8)}`)));
  if (!completed.length) { element("run-select").append(option("", "尚无已完成运行")); notify("先在终端执行：footballer-analyzer track --config configs/demo.yaml，完成后刷新。"); return; }
  const selectedId = completed.some((run) => run.run_id === state.runId) ? state.runId : completed[0].run_id;
  element("run-select").value = selectedId;
  await loadRun(selectedId);
}

element("run-select").addEventListener("change", () => loadRun(element("run-select").value).catch((error) => notify(error.message, true)));
element("reload").addEventListener("click", () => initialize().catch((error) => notify(error.message, true)));
element("show-boxes").addEventListener("change", draw);
element("show-unmatched").addEventListener("change", draw);
element("export-frame").addEventListener("click", async () => {
  if (!state.data || !state.image) return;
  stopPlayback();
  const frameIndex = state.frame;
  const runId = state.runId;
  const selectedName = element("selected-name").textContent;
  const selectedDetails = `轨迹 ${element("selected-id").textContent} · 坐标 ${element("selected-position").textContent} m`;
  const calibrationNote = state.data.calibration?.reviewed ? "标定已确认 · 精度待独立验证" : "标定草稿 · 仅供位置核对";
  const canvas = document.createElement("canvas"); canvas.width = 1600; canvas.height = 760;
  const context = canvas.getContext("2d");
  context.fillStyle = "#f2f5f3"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#193632"; context.font = "bold 30px sans-serif"; context.fillText("TOUCHLINE / 球员追踪与位置对拍", 40, 55);
  context.font = "18px sans-serif"; context.fillStyle = "#668273";
  context.fillText(`${state.data.manifest.config.clip_id} · 源帧 ${state.frame} · 时间 ${(state.frame / state.data.manifest.video.fps).toFixed(2)} s`, 40, 92);
  context.drawImage(element("video-canvas"), 40, 130, 1000, 562.5);
  const pitch = element("pitch").cloneNode(true);
  pitch.setAttribute("width", "980"); pitch.setAttribute("height", "680");
  const serialized = new XMLSerializer().serializeToString(pitch);
  const image = new Image();
  try {
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("图片生成超时")), 5000);
      image.onload = () => { clearTimeout(timeout); resolve(); };
      image.onerror = () => { clearTimeout(timeout); reject(new Error("球场图片生成失败")); };
      image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(serialized)}`;
    });
    context.drawImage(image, 1070, 190, 490, 340);
    context.fillStyle = "#193632"; context.font = "20px sans-serif";
    context.fillText(selectedName, 1080, 590);
    context.font = "17px sans-serif"; context.fillStyle = "#668273";
    context.fillText(selectedDetails, 1080, 625);
    context.fillText(calibrationNote, 1080, 660);
    context.font = "16px sans-serif"; context.fillText("同帧对应；颜色为球队或待确认颜色簇，数字为本次运行的轨迹 ID。", 40, 735);
    if (state.frame !== frameIndex || state.runId !== runId || element("selected-name").textContent !== selectedName) return;
    const anchor = element("export-download"); anchor.download = `touchline-frame-${frameIndex}.png`; anchor.href = canvas.toDataURL("image/png"); anchor.hidden = false;
    notify(`源帧 ${frameIndex} 的对拍图已生成，点击“保存 PNG”下载。`);
  } catch { notify("导出失败，请重试。", true); }
});
element("play").addEventListener("click", play);
for (const [id, direction] of [["previous-frame", -1], ["next-frame", 1]]) element(id).addEventListener("click", () => { stopPlayback(); if (state.data) showFrame(state.frame + direction * state.data.manifest.config.stride).catch((error) => notify(error.message, true)); });
element("frame-slider").addEventListener("input", () => { stopPlayback(); showFrame(Number(element("frame-slider").value)).catch((error) => notify(error.message, true)); });
element("frame-number").addEventListener("change", () => { stopPlayback(); showFrame(Number(element("frame-number").value)).catch((error) => notify(error.message, true)); });
element("jump-frame").addEventListener("click", () => { stopPlayback(); showFrame(Number(element("frame-number").value)).catch((error) => notify(error.message, true)); });
element("frame-number").addEventListener("keydown", (event) => { if (event.key === "Enter") element("jump-frame").click(); });
document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-tab]").forEach((tab) => { tab.classList.toggle("active", tab === button); tab.setAttribute("aria-selected", tab === button ? "true" : "false"); });
  for (const name of ["identity", "calibration", "statistics"]) element(`${name}-tab`).hidden = button.dataset.tab !== name;
  if (button.dataset.tab !== "calibration") { state.pointMode = false; state.roiMode = false; element("point-mode").textContent = "开始点选图像点"; element("roi-mode").textContent = "点选可靠区域"; }
  draw();
}));
element("point-mode").addEventListener("click", async () => {
  if (!state.data) return;
  state.roiMode = false; element("roi-mode").textContent = "点选可靠区域";
  stopPlayback(); state.pointMode = !state.pointMode;
  element("point-mode").textContent = state.pointMode ? "结束点选" : "开始点选图像点";
  if (state.pointMode) {
    try { await showFrame(Number(element("calibration-reference").value)); notify("点选模式：先填写下一点米制 x/y，再在原始画面点击相应白线交点。所有点须来自同一参考帧。"); } catch (error) { state.pointMode = false; notify(error.message, true); }
  }
});
element("roi-mode").addEventListener("click", async () => {
  if (!state.data) return;
  stopPlayback(); state.pointMode = false; element("point-mode").textContent = "开始点选图像点";
  state.roiMode = !state.roiMode; element("roi-mode").textContent = state.roiMode ? "结束区域点选" : "点选可靠区域";
  if (state.roiMode) {
    state.roi = []; markDraftDirty();
    try { await showFrame(Number(element("calibration-reference").value)); notify("按顺序点击可靠地面区域的边界（至少三点），完成后点“结束区域点选”。区域之外不计入统计。"); } catch (error) { state.roiMode = false; notify(error.message, true); }
  }
  draw();
});
element("video-canvas").addEventListener("click", (event) => {
  if (!state.data) return;
  const canvas = element("video-canvas"), rectangle = canvas.getBoundingClientRect();
  const horizontal = (event.clientX - rectangle.left) / rectangle.width * canvas.width;
  const vertical = (event.clientY - rectangle.top) / rectangle.height * canvas.height;
  if (state.roiMode) {
    if (state.frame !== Number(element("calibration-reference").value)) { notify("请回到参考帧后点选可靠区域。", true); return; }
    state.roi.push([Number(horizontal.toFixed(2)), Number(vertical.toFixed(2))]); markDraftDirty(); draw(); return;
  }
  if (state.pointMode) {
    if (state.frame !== Number(element("calibration-reference").value)) { notify("请回到参考帧后点选，不能混用不同帧的图像点。", true); return; }
    state.points.push({ image: [Number(horizontal.toFixed(2)), Number(vertical.toFixed(2))], pitch: [Number(element("target-x").value), Number(element("target-y").value)] });
    markDraftDirty(); renderCalibrationPoints(); draw(); return;
  }
  const hits = (state.frameRows.get(state.frame) || []).filter((row) => row.bbox_x1 <= horizontal && row.bbox_x2 >= horizontal && row.bbox_y1 <= vertical && row.bbox_y2 >= vertical).sort((first, second) => (first.bbox_x2 - first.bbox_x1) * (first.bbox_y2 - first.bbox_y1) - (second.bbox_x2 - second.bbox_x1) * (second.bbox_y2 - second.bbox_y1));
  if (hits.length) selectTrack(hits[0].track_id);
});
element("player-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (!state.data) return;
  const playerId = element("player-id").value.trim();
  try { await request(api("players"), { player_id: playerId, display_name: element("player-name").value.trim() }); await loadRun(state.runId, true); element("binding-player").value = playerId; notify("球员档案已保存，可用于此后比赛。"); } catch (error) { notify(error.message, true); }
});
async function addBinding(action) {
  if (!state.data) return;
  const entry = { track_id: Number(element("binding-track").value), start_frame: Number(element("binding-start").value), end_frame: Number(element("binding-end").value), action };
  if (action === "bind") { entry.player_id = element("binding-player").value; entry.team_id = element("binding-team").value; }
  const bindings = [...state.data.bindings.bindings, entry];
  await request(api("bindings"), { run_id: state.runId, bindings });
  await loadRun(state.runId, true); notify("区间已保存。修改身份后只需重算报告，原始轨迹保持不变。");
}
element("binding-form").addEventListener("submit", (event) => { event.preventDefault(); addBinding("bind").catch((error) => notify(error.message, true)); });
element("exclude-track").addEventListener("click", () => addBinding("exclude").catch((error) => notify(error.message, true)));
for (const id of ["pitch-length", "pitch-width", "calibration-reference", "calibration-start", "calibration-end"]) element(id).addEventListener("input", markDraftDirty);
element("clear-points").addEventListener("click", () => { state.points = []; markDraftDirty(); renderCalibrationPoints(); draw(); });
element("save-calibration").addEventListener("click", async () => {
  if (!state.data) return;
  const payload = { reference_frame: Number(element("calibration-reference").value), start_frame: Number(element("calibration-start").value), end_frame: Number(element("calibration-end").value), image_points: state.points.map((point) => point.image), pitch_points: state.points.map((point) => point.pitch), pitch_length_m: Number(element("pitch-length").value), pitch_width_m: Number(element("pitch-width").value), reviewed: element("calibration-reviewed").checked, note: state.data.calibration?.note || "人工点选标定" };
  if (state.roi.length) payload.roi = state.roi;
  try { await request(api("calibration"), payload); state.pointMode = false; state.roiMode = false; element("point-mode").textContent = "开始点选图像点"; element("roi-mode").textContent = "点选可靠区域"; await loadRun(state.runId, true); notify(payload.reviewed ? "标定已确认，可以生成统计报告。" : "标定草稿已保存，可在首、中、末帧对照白线叠加。"); } catch (error) { notify(error.message, true); }
});
element("generate-report").addEventListener("click", async () => {
  if (!state.data) return;
  const button = element("generate-report"); button.disabled = true; button.classList.add("busy"); button.textContent = "正在计算与渲染…"; stopPlayback();
  try { await request(api("report"), {}); await loadRun(state.runId, true); notify("新版本报告已生成。点击统计表中的球员查看速度曲线与热图。"); } catch (error) { notify(error.message, true); }
  finally { button.disabled = false; button.classList.remove("busy"); button.textContent = "生成 / 重算报告"; }
});
initialize().catch((error) => notify(error.message, true));
