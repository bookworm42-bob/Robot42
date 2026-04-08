from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from typing import Any, Protocol

from .exploration import ExplorationBackend
from .offload import OffloadClient


class ExplorationUIController(Protocol):
    def snapshot(self) -> dict[str, Any]:
        ...

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        ...

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        ...

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        ...

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        ...

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        ...

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        ...

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        ...

    def split_region(self, region_id: str, *, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        ...

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any] | None:
        ...

    def approve_map(self) -> dict[str, Any] | None:
        ...


class LocalExplorationUIController:
    def __init__(self, backend: ExplorationBackend) -> None:
        self.backend = backend

    def snapshot(self) -> dict[str, Any]:
        return self.backend.snapshot()

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        return self.backend.start_explore(area=area, session=session, source=source)

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        return self.backend.start_create_map(area=area, session=session, source=source)

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        return self.backend.pause_task(task_id)

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        return self.backend.resume_task(task_id)

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        return self.backend.cancel_task(task_id)

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        return self.backend.update_region(
            region_id,
            label=label,
            polygon_2d=polygon_2d,
            default_waypoints=default_waypoints,
        )

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        return self.backend.merge_regions(region_ids, new_label=new_label)

    def split_region(self, region_id: str, *, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        return self.backend.split_region(region_id, polygons)

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any] | None:
        return self.backend.set_named_place(name, pose, region_id=region_id)

    def approve_map(self) -> dict[str, Any] | None:
        return self.backend.approve_current_map()


class RemoteExplorationUIController:
    def __init__(self, client: OffloadClient) -> None:
        self.client = client
        self.client.ensure_registered()

    def snapshot(self) -> dict[str, Any]:
        return self.client.mapping_snapshot()

    def start_explore(self, *, area: str, session: str | None = None, source: str = "operator") -> dict[str, Any]:
        return self.client.start_explore(area=area, session=session, source=source)

    def start_create_map(self, *, area: str, session: str, source: str = "operator") -> dict[str, Any]:
        return self.client.start_create_map(area=area, session=session, source=source)

    def pause_task(self, task_id: str) -> dict[str, Any] | None:
        return self.client.pause_mapping_task(task_id)

    def resume_task(self, task_id: str) -> dict[str, Any] | None:
        return self.client.resume_mapping_task(task_id)

    def cancel_task(self, task_id: str | None = None) -> dict[str, Any] | None:
        return self.client.cancel_mapping_task(task_id)

    def update_region(
        self,
        region_id: str,
        *,
        label: str | None = None,
        polygon_2d: list[list[float]] | None = None,
        default_waypoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        return self.client.update_mapping_region(
            region_id,
            label=label,
            polygon_2d=polygon_2d,
            default_waypoints=default_waypoints,
        )

    def merge_regions(self, region_ids: list[str], *, new_label: str | None = None) -> dict[str, Any] | None:
        return self.client.merge_mapping_regions(region_ids, new_label=new_label)

    def split_region(self, region_id: str, *, polygons: list[list[list[float]]] | None = None) -> list[dict[str, Any]]:
        return list(self.client.split_mapping_region(region_id, polygons=polygons).get("regions", []))

    def set_named_place(self, *, name: str, pose: dict[str, Any], region_id: str | None = None) -> dict[str, Any] | None:
        return self.client.set_named_place(name=name, pose=pose, region_id=region_id)

    def approve_map(self) -> dict[str, Any] | None:
        return self.client.approve_mapping_map()


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XLeRobot Exploration Review</title>
  <style>
    :root {
      --bg: #f7f4ec;
      --panel: rgba(255,255,255,0.8);
      --line: rgba(15, 23, 42, 0.12);
      --text: #172033;
      --muted: #52606d;
      --accent: #0f766e;
      --accent-2: #a16207;
      --danger: #b91c1c;
      --shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 30%),
        radial-gradient(circle at right, rgba(161,98,7,0.16), transparent 24%),
        linear-gradient(180deg, #faf7f0 0%, #efe7d8 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }
    .hero {
      display: grid;
      gap: 8px;
      margin-bottom: 20px;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      color: var(--muted);
    }
    h1 {
      margin: 0;
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(30px, 5vw, 52px);
      line-height: 0.98;
      max-width: 13ch;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px 1fr 340px;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      box-shadow: var(--shadow);
    }
    .stack { display: grid; gap: 16px; }
    label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, textarea, button {
      font: inherit;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.92);
      color: var(--text);
    }
    textarea { min-height: 90px; resize: vertical; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 15px;
      cursor: pointer;
      font-weight: 600;
    }
    .primary { background: var(--accent); color: white; }
    .secondary { background: #fff7ed; color: var(--accent-2); }
    .danger { background: #fee2e2; color: var(--danger); }
    .button-row { display: flex; flex-wrap: wrap; gap: 8px; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .meta-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.68);
    }
    .meta-key {
      font-size: 11px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .meta-value {
      font-size: 15px;
      font-weight: 600;
    }
    .map-shell {
      display: grid;
      gap: 12px;
    }
    #map-canvas {
      width: 100%;
      height: 720px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.84);
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: middle;
    }
    .free::before { background: rgba(148, 163, 184, 0.45); }
    .traj::before { background: #0f766e; }
    .region::before { background: rgba(180, 83, 9, 0.35); }
    .list {
      display: grid;
      gap: 10px;
      max-height: 260px;
      overflow: auto;
    }
    .list-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.74);
      cursor: pointer;
    }
    .list-card.active {
      border-color: rgba(15,118,110,0.35);
      background: rgba(15,118,110,0.08);
    }
    .thumbs {
      display: grid;
      gap: 10px;
    }
    .thumbs img {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }
    .muted { color: var(--muted); }
    @media (max-width: 1180px) {
      .grid { grid-template-columns: 1fr; }
      #map-canvas { height: 520px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Exploration Review</div>
      <h1>Review, correct, and approve semantic maps.</h1>
      <div class="muted">__HERO_SUBTITLE__</div>
    </section>

    <div class="grid">
      <div class="stack">
        <section class="panel" __CONTROL_PANEL_ATTR__>
          <div class="eyebrow">Control</div>
          <label for="area">Area</label>
          <input id="area" value="downstairs" />
          <label for="session">Session</label>
          <input id="session" value="house_v1" />
          <div class="button-row" style="margin-top:10px;">
            <button class="primary" id="start-explore">Start Explore</button>
            <button class="secondary" id="start-map">Create Map</button>
          </div>
          <div class="button-row" style="margin-top:10px;">
            <button class="secondary" id="pause-task">Pause</button>
            <button class="secondary" id="resume-task">Resume</button>
            <button class="danger" id="cancel-task">Cancel</button>
          </div>
          <div class="button-row" style="margin-top:10px;">
            <button class="primary" id="approve-map">Approve Map</button>
          </div>
        </section>

        <section class="panel">
          <div class="eyebrow">Status</div>
          <div id="meta-grid" class="meta-grid"></div>
        </section>

        <section class="panel">
          <div class="eyebrow">Regions</div>
          <div id="region-list" class="list"></div>
        </section>
      </div>

      <section class="panel map-shell">
        <div class="eyebrow">Map</div>
        <svg id="map-canvas" viewBox="0 0 1000 700"></svg>
        <div class="legend">
          <span class="free">occupancy</span>
          <span class="traj">trajectory</span>
          <span class="region">region overlay</span>
        </div>
      </section>

      <div class="stack">
        <section class="panel">
          <div class="eyebrow">Selected Region</div>
          <div id="selected-summary" class="muted">Select a region from the map or list.</div>
          <label for="region-label">Label</label>
          <input id="region-label" />
          <label for="region-polygon">Polygon JSON</label>
          <textarea id="region-polygon"></textarea>
          <label for="region-waypoints">Waypoints JSON</label>
          <textarea id="region-waypoints"></textarea>
          <div class="button-row" style="margin-top:10px;">
            <button class="primary" id="save-region">Save Region</button>
            <button class="secondary" id="split-region">Split Region</button>
          </div>
        </section>

        <section class="panel">
          <div class="eyebrow">Merge Regions</div>
          <label for="merge-ids">Region IDs JSON</label>
          <textarea id="merge-ids">[]</textarea>
          <label for="merge-label">Merged Label</label>
          <input id="merge-label" />
          <div class="button-row" style="margin-top:10px;">
            <button class="secondary" id="merge-regions">Merge</button>
          </div>
        </section>

        <section class="panel">
          <div class="eyebrow">Named Place</div>
          <label for="place-name">Name</label>
          <input id="place-name" placeholder="kitchen_entry" />
          <label for="place-pose">Pose JSON</label>
          <textarea id="place-pose">{"x":0,"y":0,"yaw":0}</textarea>
          <div class="button-row" style="margin-top:10px;">
            <button class="secondary" id="save-place">Save Place</button>
          </div>
        </section>

        <section class="panel">
          <div class="eyebrow">Keyframes</div>
          <div id="keyframes" class="thumbs"></div>
        </section>
      </div>
    </div>
  </div>

  <script>
    let selectedRegionId = null;
    let currentState = null;

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {})
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || ('Request failed: ' + response.status));
      }
      return response.json();
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function mapBounds(map) {
      if (map && map.occupancy && map.occupancy.bounds) {
        return map.occupancy.bounds;
      }
      const points = [];
      (map?.regions || []).forEach((region) => (region.polygon_2d || []).forEach((point) => points.push(point)));
      if (!points.length) {
        return {min_x: 0, max_x: 10, min_y: 0, max_y: 8};
      }
      return {
        min_x: Math.min(...points.map((point) => point[0])),
        max_x: Math.max(...points.map((point) => point[0])),
        min_y: Math.min(...points.map((point) => point[1])),
        max_y: Math.max(...points.map((point) => point[1]))
      };
    }

    function makeProjector(bounds) {
      const width = Math.max(bounds.max_x - bounds.min_x, 1);
      const height = Math.max(bounds.max_y - bounds.min_y, 1);
      const pad = 36;
      return function(point) {
        const x = pad + ((point.x - bounds.min_x) / width) * (1000 - pad * 2);
        const y = 700 - pad - ((point.y - bounds.min_y) / height) * (700 - pad * 2);
        return {x, y};
      };
    }

    function renderMeta(state) {
      const task = state.active_task;
      const map = state.current_map;
      const items = [
        ['Mode', state.mode || 'unknown'],
        ['Map', map ? map.map_id : 'none'],
        ['Approved', map && map.approved ? 'yes' : 'no'],
        ['Regions', map ? (map.regions || []).length : 0],
        ['Coverage', map ? String(map.coverage || 0) : '0'],
        ['Task', task ? (task.tool_id + ' ' + task.state) : 'idle']
      ];
      document.getElementById('meta-grid').innerHTML = items.map(([key, value]) => `
        <div class="meta-card">
          <div class="meta-key">${escapeHtml(key)}</div>
          <div class="meta-value">${escapeHtml(value)}</div>
        </div>
      `).join('');
    }

    function renderRegions(state) {
      const regions = state.current_map?.regions || [];
      document.getElementById('region-list').innerHTML = regions.map((region) => `
        <div class="list-card ${region.region_id === selectedRegionId ? 'active' : ''}" data-region-id="${escapeHtml(region.region_id)}">
          <strong>${escapeHtml(region.label)}</strong><br/>
          <span class="muted">${escapeHtml(region.region_id)} · ${escapeHtml(String(region.confidence))}</span>
        </div>
      `).join('');
      for (const element of document.querySelectorAll('[data-region-id]')) {
        element.addEventListener('click', () => {
          selectedRegionId = element.getAttribute('data-region-id');
          refreshSelectedRegion();
          renderRegions(currentState);
          renderMap(currentState);
        });
      }
    }

    function renderMap(state) {
      const svg = document.getElementById('map-canvas');
      const map = state.current_map;
      if (!map) {
        svg.innerHTML = '<text x="40" y="60" fill="#52606d" font-size="22">No map yet.</text>';
        return;
      }
      const bounds = mapBounds(map);
      const project = makeProjector(bounds);
      const occupancy = (map.occupancy?.cells || []).map((cell) => {
        const p = project({x: cell.x, y: cell.y});
        const p2 = project({x: cell.x + (map.occupancy.resolution || 0.5), y: cell.y + (map.occupancy.resolution || 0.5)});
        const fill = cell.state === 'occupied'
          ? 'rgba(15,23,42,0.55)'
          : cell.state === 'free'
            ? 'rgba(148,163,184,0.22)'
            : 'rgba(148,163,184,0.10)';
        return `<rect x="${p.x}" y="${p2.y}" width="${Math.max(2, p2.x - p.x)}" height="${Math.max(2, p.y - p2.y)}" fill="${fill}" />`;
      }).join('');
      const trajectory = (map.trajectory || []).map((point) => {
        const p = project(point);
        return `${p.x},${p.y}`;
      }).join(' ');
      const regions = (map.regions || []).map((region, index) => {
        const polygon = (region.polygon_2d || []).map((point) => {
          const p = project({x: point[0], y: point[1]});
          return `${p.x},${p.y}`;
        }).join(' ');
        const centroid = project(region.centroid || {x: 0, y: 0});
        const active = region.region_id === selectedRegionId;
        const hue = 24 + index * 47;
        const fill = active ? `hsla(${hue}, 72%, 45%, 0.35)` : `hsla(${hue}, 62%, 52%, 0.18)`;
        const stroke = active ? '#0f766e' : `hsla(${hue}, 62%, 42%, 0.62)`;
        return `
          <polygon data-region-id="${escapeHtml(region.region_id)}" points="${polygon}" fill="${fill}" stroke="${stroke}" stroke-width="${active ? 4 : 2}" />
          <text x="${centroid.x}" y="${centroid.y}" text-anchor="middle" font-size="16" fill="#172033" font-weight="600">${escapeHtml(region.label)}</text>
        `;
      }).join('');
      const namedPlaces = (map.named_places || []).map((place) => {
        const p = project(place.pose || {x: 0, y: 0});
        return `
          <circle cx="${p.x}" cy="${p.y}" r="5" fill="#0f766e" />
          <text x="${p.x + 8}" y="${p.y - 8}" font-size="12" fill="#0f766e">${escapeHtml(place.name)}</text>
        `;
      }).join('');
      svg.innerHTML = `
        <rect x="0" y="0" width="1000" height="700" fill="rgba(255,255,255,0.92)" />
        ${occupancy}
        <polyline points="${trajectory}" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        ${regions}
        ${namedPlaces}
      `;
      for (const element of svg.querySelectorAll('[data-region-id]')) {
        element.addEventListener('click', () => {
          selectedRegionId = element.getAttribute('data-region-id');
          refreshSelectedRegion();
          renderRegions(currentState);
          renderMap(currentState);
        });
      }
    }

    function refreshSelectedRegion() {
      const map = currentState?.current_map;
      const region = (map?.regions || []).find((item) => item.region_id === selectedRegionId);
      const summary = document.getElementById('selected-summary');
      const keyframes = document.getElementById('keyframes');
      if (!region) {
        summary.textContent = 'Select a region from the map or list.';
        document.getElementById('region-label').value = '';
        document.getElementById('region-polygon').value = '';
        document.getElementById('region-waypoints').value = '';
        keyframes.innerHTML = '';
        return;
      }
      summary.textContent = `${region.label} (${region.region_id})`;
      document.getElementById('region-label').value = region.label || '';
      document.getElementById('region-polygon').value = JSON.stringify(region.polygon_2d || [], null, 2);
      document.getElementById('region-waypoints').value = JSON.stringify(region.default_waypoints || [], null, 2);
      const framesById = Object.fromEntries((map.keyframes || []).map((item) => [item.frame_id, item]));
      keyframes.innerHTML = (region.representative_keyframes || []).map((frameId) => {
        const frame = framesById[frameId];
        if (!frame) return '';
        return `
          <div>
            <img src="${frame.thumbnail_data_url}" alt="${escapeHtml(frame.frame_id)}" />
            <div class="muted">${escapeHtml(frame.description)}</div>
          </div>
        `;
      }).join('');
      if (region.centroid) {
        document.getElementById('place-pose').value = JSON.stringify({x: region.centroid.x, y: region.centroid.y, yaw: 0}, null, 2);
      }
    }

    async function refresh() {
      const response = await fetch('/api/state');
      currentState = await response.json();
      renderMeta(currentState);
      renderRegions(currentState);
      renderMap(currentState);
      refreshSelectedRegion();
    }

    document.getElementById('start-explore').addEventListener('click', async () => {
      await postJson('/api/explore/start', {
        area: document.getElementById('area').value.trim(),
        session: document.getElementById('session').value.trim() || null
      });
      await refresh();
    });
    document.getElementById('start-map').addEventListener('click', async () => {
      await postJson('/api/create_map/start', {
        area: document.getElementById('area').value.trim(),
        session: document.getElementById('session').value.trim()
      });
      await refresh();
    });
    document.getElementById('pause-task').addEventListener('click', async () => {
      const taskId = currentState?.active_task?.task_id;
      if (!taskId) return;
      await postJson('/api/task/pause', {task_id: taskId});
      await refresh();
    });
    document.getElementById('resume-task').addEventListener('click', async () => {
      const taskId = currentState?.active_task?.task_id;
      if (!taskId) return;
      await postJson('/api/task/resume', {task_id: taskId});
      await refresh();
    });
    document.getElementById('cancel-task').addEventListener('click', async () => {
      const taskId = currentState?.active_task?.task_id || null;
      await postJson('/api/task/cancel', {task_id: taskId});
      await refresh();
    });
    document.getElementById('approve-map').addEventListener('click', async () => {
      await postJson('/api/approve');
      await refresh();
    });
    document.getElementById('save-region').addEventListener('click', async () => {
      if (!selectedRegionId) return;
      await postJson('/api/region/update', {
        region_id: selectedRegionId,
        label: document.getElementById('region-label').value.trim(),
        polygon_2d: JSON.parse(document.getElementById('region-polygon').value || '[]'),
        default_waypoints: JSON.parse(document.getElementById('region-waypoints').value || '[]')
      });
      await refresh();
    });
    document.getElementById('split-region').addEventListener('click', async () => {
      if (!selectedRegionId) return;
      await postJson('/api/region/split', {region_id: selectedRegionId});
      await refresh();
    });
    document.getElementById('merge-regions').addEventListener('click', async () => {
      await postJson('/api/regions/merge', {
        region_ids: JSON.parse(document.getElementById('merge-ids').value || '[]'),
        new_label: document.getElementById('merge-label').value.trim() || null
      });
      selectedRegionId = null;
      await refresh();
    });
    document.getElementById('save-place').addEventListener('click', async () => {
      await postJson('/api/named_place', {
        name: document.getElementById('place-name').value.trim(),
        pose: JSON.parse(document.getElementById('place-pose').value || '{}'),
        region_id: selectedRegionId
      });
      await refresh();
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class ExplorationReviewServer:
    def __init__(
        self,
        controller: ExplorationUIController,
        *,
        host: str = "127.0.0.1",
        port: int = 8770,
        allow_task_controls: bool = True,
    ) -> None:
        self.controller = controller
        self.host = host
        self.port = port
        self.allow_task_controls = allow_task_controls
        self._server: ThreadingHTTPServer | None = None

    def _html_page(self) -> str:
        subtitle = (
            "Manual exploration triggers, live map progress, semantic room editing, and final approval all happen here."
            if self.allow_task_controls
            else "Post-run map review, region correction, waypoint edits, and approval happen here."
        )
        control_attr = "" if self.allow_task_controls else 'style="display:none;"'
        return (
            HTML_PAGE.replace("__HERO_SUBTITLE__", subtitle)
            .replace("__CONTROL_PANEL_ATTR__", control_attr)
        )

    def serve_forever(self) -> None:
        controller = self.controller

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/" or self.path == "/index.html":
                    self._send_html(self.server.codex_html_page())
                    return
                if self.path == "/api/state":
                    self._send_json(controller.snapshot())
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def do_POST(self) -> None:
                payload = self._read_json_body()
                if self.path == "/api/explore/start":
                    self._send_json(
                        controller.start_explore(
                            area=str(payload.get("area", "workspace")),
                            session=payload.get("session"),
                            source="operator",
                        )
                    )
                    return
                if self.path == "/api/create_map/start":
                    session = str(payload.get("session") or f"map_{int(time.time())}")
                    self._send_json(
                        controller.start_create_map(
                            area=str(payload.get("area", "workspace")),
                            session=session,
                            source="operator",
                        )
                    )
                    return
                if self.path == "/api/task/pause":
                    response = controller.pause_task(str(payload.get("task_id")))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/task/resume":
                    response = controller.resume_task(str(payload.get("task_id")))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/task/cancel":
                    response = controller.cancel_task(payload.get("task_id"))
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/region/update":
                    response = controller.update_region(
                        str(payload.get("region_id")),
                        label=payload.get("label"),
                        polygon_2d=payload.get("polygon_2d"),
                        default_waypoints=payload.get("default_waypoints"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/region/split":
                    response = controller.split_region(
                        str(payload.get("region_id")),
                        polygons=payload.get("polygons"),
                    )
                    self._send_json({"regions": response})
                    return
                if self.path == "/api/regions/merge":
                    response = controller.merge_regions(
                        list(payload.get("region_ids", [])),
                        new_label=payload.get("new_label"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/named_place":
                    response = controller.set_named_place(
                        name=str(payload.get("name")),
                        pose=dict(payload.get("pose", {})),
                        region_id=payload.get("region_id"),
                    )
                    self._send_json(response or {"status": "missing"})
                    return
                if self.path == "/api/approve":
                    self._send_json(controller.approve_map() or {"status": "missing"})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                body = self.rfile.read(length).decode("utf-8")
                return json.loads(body)

            def _send_html(self, content: str) -> None:
                encoded = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_json(self, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        setattr(self._server, "codex_html_page", self._html_page)
        self._server.serve_forever()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
