/* =========================================================================
   AlphaMaster · 背景动效引擎
   三层画布：
     1. 神经网络层  — 神经元节点 + 突触连线 + 沿突触流动的数据脉冲
     2. 数学符号层  — 深空中缓慢漂移 / 旋转的数学符号与希腊字母
     3. 光晕层      — 随鼠标缓动的极光辉光
   纯 Canvas 实现，尊重 prefers-reduced-motion，页面隐藏时自动暂停。
   ========================================================================= */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const neuralCanvas = document.getElementById("neuralCanvas");
  const glyphCanvas = document.getElementById("glyphCanvas");
  if (!neuralCanvas || !glyphCanvas) return;

  const nctx = neuralCanvas.getContext("2d");
  const gctx = glyphCanvas.getContext("2d");

  let W = 0, H = 0, DPR = Math.min(window.devicePixelRatio || 1, 2);

  // 鼠标缓动（视差 + 辉光跟随）
  const mouse = { x: 0.5, y: 0.35, tx: 0.5, ty: 0.35 };

  /* ----------------------------- 数据结构 ----------------------------- */
  let nodes = [];
  let pulses = [];
  let glyphs = [];

  const GLYPH_CHARS = [
    "∑", "∫", "∂", "∇", "π", "λ", "θ", "μ", "σ", "φ", "Ψ", "Ω", "α", "β", "γ",
    "√", "∞", "≈", "≠", "∈", "⊗", "⊕", "∴", "Δ", "ρ", "τ", "ξ", "ζ",
    "01", "10", "f(x)", "dₜ", "∮", "≜", "⟨ψ⟩", "e^x", "∝",
  ];

  function rand(a, b) { return a + Math.random() * (b - a); }

  function computeCounts() {
    const area = W * H;
    const nodeCount = Math.max(28, Math.min(90, Math.round(area / 22000)));
    const glyphCount = Math.max(10, Math.min(30, Math.round(area / 70000)));
    return { nodeCount, glyphCount };
  }

  function buildScene() {
    const { nodeCount, glyphCount } = computeCounts();

    nodes = new Array(nodeCount).fill(0).map(() => {
      const depth = rand(0.35, 1); // 越大越靠前
      return {
        x: rand(0, W),
        y: rand(0, H),
        vx: rand(-0.12, 0.12) * depth,
        vy: rand(-0.12, 0.12) * depth,
        depth,
        r: rand(1.1, 2.6) * depth,
        // 呼吸脉动
        phase: rand(0, Math.PI * 2),
        pspeed: rand(0.6, 1.6),
      };
    });

    glyphs = new Array(glyphCount).fill(0).map(() => ({
      char: GLYPH_CHARS[(Math.random() * GLYPH_CHARS.length) | 0],
      x: rand(0, W),
      y: rand(0, H),
      size: rand(26, 84),
      vx: rand(-0.06, 0.06),
      vy: rand(-0.05, 0.05),
      rot: rand(0, Math.PI * 2),
      vrot: rand(-0.0025, 0.0025),
      alpha: rand(0.02, 0.07),
      pulse: rand(0, Math.PI * 2),
    }));

    pulses = [];
  }

  function resize() {
    W = window.innerWidth;
    H = window.innerHeight;
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    for (const c of [neuralCanvas, glyphCanvas]) {
      c.width = W * DPR;
      c.height = H * DPR;
      c.style.width = W + "px";
      c.style.height = H + "px";
    }
    nctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    gctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    buildScene();
  }

  /* ----------------------- 连线 & 脉冲发射 ----------------------- */
  const LINK_DIST = 168;      // 建立突触的最大距离
  const LINK_DIST_SQ = LINK_DIST * LINK_DIST;

  function spawnPulse(a, b) {
    pulses.push({
      ax: a.x, ay: a.y, bx: b.x, by: b.y,
      t: 0,
      speed: rand(0.006, 0.016),
      hue: Math.random() < 0.5 ? "teal" : "violet",
    });
  }

  function maybeSpawnPulses() {
    if (reduceMotion) return;
    // 控制脉冲总量
    if (pulses.length > 46) return;
    if (Math.random() > 0.28) return;
    const a = nodes[(Math.random() * nodes.length) | 0];
    // 找一个近邻
    let best = null, bestD = LINK_DIST_SQ;
    for (const b of nodes) {
      if (b === a) continue;
      const dx = a.x - b.x, dy = a.y - b.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = b; }
    }
    if (best) spawnPulse(a, best);
  }

  /* ------------------------------ 绘制 ------------------------------ */
  let last = performance.now();
  let running = true;

  function frame(now) {
    if (!running) return;
    const dt = Math.min(48, now - last);
    last = now;

    // 鼠标缓动
    mouse.x += (mouse.tx - mouse.x) * 0.05;
    mouse.y += (mouse.ty - mouse.y) * 0.05;
    const parX = (mouse.x - 0.5);
    const parY = (mouse.y - 0.5);

    drawGlyphs(dt, parX, parY);
    drawNeural(dt, now, parX, parY);

    requestAnimationFrame(frame);
  }

  function drawGlyphs(dt, parX, parY) {
    gctx.clearRect(0, 0, W, H);
    gctx.textAlign = "center";
    gctx.textBaseline = "middle";
    for (const g of glyphs) {
      if (!reduceMotion) {
        g.x += g.vx * (dt / 16);
        g.y += g.vy * (dt / 16);
        g.rot += g.vrot * (dt / 16);
        g.pulse += 0.01 * (dt / 16);
      }
      // 环绕
      if (g.x < -100) g.x = W + 100;
      if (g.x > W + 100) g.x = -100;
      if (g.y < -100) g.y = H + 100;
      if (g.y > H + 100) g.y = -100;

      const px = g.x + parX * 34 * (g.size / 84);
      const py = g.y + parY * 34 * (g.size / 84);
      const a = g.alpha + Math.sin(g.pulse) * 0.02;

      gctx.save();
      gctx.translate(px, py);
      gctx.rotate(g.rot);
      gctx.font = `${g.size}px "JetBrains Mono", monospace`;
      gctx.fillStyle = `rgba(120, 190, 235, ${Math.max(0.012, a)})`;
      gctx.shadowColor = "rgba(94, 234, 212, 0.5)";
      gctx.shadowBlur = 12;
      gctx.fillText(g.char, 0, 0);
      gctx.restore();
    }
  }

  function drawNeural(dt, now, parX, parY) {
    nctx.clearRect(0, 0, W, H);

    // 更新节点位置
    for (const n of nodes) {
      if (!reduceMotion) {
        n.x += n.vx * (dt / 16);
        n.y += n.vy * (dt / 16);
      }
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
      n.x = Math.max(0, Math.min(W, n.x));
      n.y = Math.max(0, Math.min(H, n.y));
    }

    // 突触连线
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      const ax = a.x + parX * 18 * a.depth;
      const ay = a.y + parY * 18 * a.depth;
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dsq = dx * dx + dy * dy;
        if (dsq > LINK_DIST_SQ) continue;
        const d = Math.sqrt(dsq);
        const strength = 1 - d / LINK_DIST;
        const bx = b.x + parX * 18 * b.depth;
        const by = b.y + parY * 18 * b.depth;
        nctx.beginPath();
        nctx.moveTo(ax, ay);
        nctx.lineTo(bx, by);
        nctx.strokeStyle = `rgba(94, 234, 212, ${strength * 0.14})`;
        nctx.lineWidth = strength * 1.1;
        nctx.stroke();
      }
    }

    // 节点（神经元）
    for (const n of nodes) {
      const px = n.x + parX * 18 * n.depth;
      const py = n.y + parY * 18 * n.depth;
      const breath = reduceMotion ? 0.7 : 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(now * 0.001 * n.pspeed + n.phase));
      const r = n.r * (0.8 + breath * 0.5);

      const grd = nctx.createRadialGradient(px, py, 0, px, py, r * 5);
      grd.addColorStop(0, `rgba(140, 245, 225, ${0.5 * breath})`);
      grd.addColorStop(0.4, `rgba(94, 234, 212, ${0.16 * breath})`);
      grd.addColorStop(1, "rgba(94, 234, 212, 0)");
      nctx.fillStyle = grd;
      nctx.beginPath();
      nctx.arc(px, py, r * 5, 0, Math.PI * 2);
      nctx.fill();

      nctx.fillStyle = `rgba(200, 255, 245, ${0.75 * breath})`;
      nctx.beginPath();
      nctx.arc(px, py, r, 0, Math.PI * 2);
      nctx.fill();
    }

    // 数据脉冲
    maybeSpawnPulses();
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.t += p.speed * (dt / 16);
      if (p.t >= 1) { pulses.splice(i, 1); continue; }
      const ease = p.t;
      const x = p.ax + (p.bx - p.ax) * ease + parX * 18;
      const y = p.ay + (p.by - p.ay) * ease + parY * 18;
      const fade = Math.sin(p.t * Math.PI); // 两端淡入淡出
      const col = p.hue === "teal" ? "94, 234, 212" : "129, 140, 248";

      const grd = nctx.createRadialGradient(x, y, 0, x, y, 7);
      grd.addColorStop(0, `rgba(${col}, ${0.9 * fade})`);
      grd.addColorStop(1, `rgba(${col}, 0)`);
      nctx.fillStyle = grd;
      nctx.beginPath();
      nctx.arc(x, y, 7, 0, Math.PI * 2);
      nctx.fill();

      nctx.fillStyle = `rgba(235, 255, 252, ${fade})`;
      nctx.beginPath();
      nctx.arc(x, y, 1.6, 0, Math.PI * 2);
      nctx.fill();
    }
  }

  /* ------------------------------ 事件 ------------------------------ */
  window.addEventListener("pointermove", (e) => {
    mouse.tx = e.clientX / W;
    mouse.ty = e.clientY / H;
  }, { passive: true });

  window.addEventListener("resize", () => {
    clearTimeout(window.__bgResize);
    window.__bgResize = setTimeout(resize, 150);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      running = false;
    } else if (!running) {
      running = true;
      last = performance.now();
      requestAnimationFrame(frame);
    }
  });

  /* ------------------------------ 启动 ------------------------------ */
  resize();
  if (reduceMotion) {
    // 静态渲染一帧
    drawGlyphs(16, 0, 0);
    drawNeural(16, performance.now(), 0, 0);
  } else {
    requestAnimationFrame(frame);
  }
})();
