/**
 * state_diagram.js — Interactive Animated State Transition Diagram
 * ================================================================
 * Standalone, reusable library for rendering state transition diagrams.
 *
 * Compact 3-tier layout:  π (Start) → Hidden States → Observations
 *
 * Features:
 *   • Configurable colours, fonts, and layout via config object
 *   • Compact vertical layout
 *   • Bold, visible arrows
 *   • Wide-separated bidirectional arcs (no overlap)
 *   • Self-loops clearly visible on outer edges
 *   • Emission arrows with clear labels
 *   • Particle flow animation, replay, inspector
 *
 * Usage:
 *   const diagram = new StateTransitionDiagram('#container', '#inspector', config);
 *   diagram.feedIteration({ A, B, pi, iteration, log_likelihood });
 *
 * @requires d3.js v7+
 * @license MIT
 */

/* ══════════════════════════════════════════════════════════════════
 *  DEFAULT CONFIGURATION
 * ══════════════════════════════════════════════════════════════════ */

const STD_DEFAULT_STATE_COLORS = [
    { base: '#F59E0B', light: '#FDE68A', dark: '#92400E', grad: ['#FBBF24', '#D97706'] },
    { base: '#3B82F6', light: '#93C5FD', dark: '#1E3A8A', grad: ['#60A5FA', '#2563EB'] },
    { base: '#10B981', light: '#6EE7B7', dark: '#064E3B', grad: ['#34D399', '#059669'] },
    { base: '#F43F5E', light: '#FDA4AF', dark: '#881337', grad: ['#FB7185', '#E11D48'] },
    { base: '#8B5CF6', light: '#C4B5FD', dark: '#4C1D95', grad: ['#A78BFA', '#7C3AED'] },
    { base: '#06B6D4', light: '#67E8F9', dark: '#155E75', grad: ['#22D3EE', '#0891B2'] },
    { base: '#EC4899', light: '#F9A8D4', dark: '#831843', grad: ['#F472B6', '#DB2777'] },
    { base: '#14B8A6', light: '#5EEAD4', dark: '#134E4A', grad: ['#2DD4BF', '#0D9488'] },
    { base: '#F97316', light: '#FDBA74', dark: '#7C2D12', grad: ['#FB923C', '#EA580C'] },
    { base: '#6366F1', light: '#A5B4FC', dark: '#3730A3', grad: ['#818CF8', '#4F46E5'] },
    { base: '#84CC16', light: '#BEF264', dark: '#3F6212', grad: ['#A3E635', '#65A30D'] },
    { base: '#E879F9', light: '#F0ABFC', dark: '#701A75', grad: ['#D946EF', '#C026D3'] },
];
const STD_DEFAULT_OBS_COLOR  = { fill: '#F1F5F9', stroke: '#94A3B8', dark: '#334155' };
const STD_DEFAULT_PI_COLOR   = { fill: '#F5F3FF', stroke: '#8B5CF6', dark: '#5B21B6' };

/**
 * StateTransitionDiagram — Interactive animated state transition diagram.
 *
 * @param {string}  containerId   CSS selector for the diagram container element.
 * @param {string}  inspectorId   CSS selector for the inspector panel (optional).
 * @param {Object}  [config]      Configuration overrides.
 * @param {Array}   [config.stateColors]      Array of {base, light, dark, grad} colour objects.
 * @param {Object}  [config.obsColor]         {fill, stroke, dark} for observation nodes.
 * @param {Object}  [config.piColor]          {fill, stroke, dark} for start node.
 * @param {boolean} [config.particlesEnabled] Enable particle animation (default: true).
 * @param {boolean} [config.decongestionEnabled] Enable edge de-congestion filtering
 *                                              (default: false).
 * @param {number}  [config.animationSpeed]   Playback speed multiplier (default: 1).
 * @param {string}  [config.fontFamily]       Primary font (default: Inter, system-ui, sans-serif).
 * @param {string}  [config.monoFontFamily]   Monospace font (default: JetBrains Mono, monospace).
 */
class StateTransitionDiagram {
    constructor(containerId, inspectorId, config) {
        // Merge user config with defaults
        const cfg = config || {};
        this.STATE_COLORS = cfg.stateColors || STD_DEFAULT_STATE_COLORS;
        this.OBS_COLOR    = cfg.obsColor    || STD_DEFAULT_OBS_COLOR;
        this.PI_COLOR     = cfg.piColor     || STD_DEFAULT_PI_COLOR;
        this.fontFamily   = cfg.fontFamily  || "'Inter', system-ui, sans-serif";
        this.monoFont     = cfg.monoFontFamily || "'JetBrains Mono', monospace";

        this.container = document.querySelector(containerId);
        this.inspectorEl = inspectorId ? document.querySelector(inspectorId) : null;
        this.history = []; this.currentIdx = -1;
        this.isPlaying = false; this.playTimer = null; this.playSpeed = cfg.animationSpeed || 1;
        this.particlesOn = cfg.particlesEnabled !== undefined ? cfg.particlesEnabled : true;
        this.decongestionOn = cfg.decongestionEnabled !== undefined ? cfg.decongestionEnabled : false;
        this.isScrubbing = false;
        this.followLatest = true;
        this.built = false;
        this.particles = []; this.animFrame = null;
        this.svg = null; this.N = 0; this.M = 0; this._ctrl = null;
    }

    /* ═══════ PUBLIC API ═══════ */
    feedIteration(data) {
        this.history.push({ A: data.A, B: data.B, pi: data.pi, iteration: data.iteration, log_likelihood: data.log_likelihood });
        if (!this.built) { this._build(data.A, data.B, data.pi); this.built = true; }
        if (!this.isScrubbing && this.followLatest) {
            this.currentIdx = this.history.length - 1; this._render(this.currentIdx);
        }
        this._updateControls();
    }
    onComplete() { this.pause(); this._updateControls(); }
    seekTo(i) {
        if (i < 0 || i >= this.history.length) return;
        this.currentIdx = i;
        this.followLatest = i >= this.history.length - 1;
        this._render(i);
        this._updateControls();
    }
    play() {
        if (!this.history.length) return;
        this.followLatest = true;
        this.isPlaying = true;
        this._updateControls();
        this._playStep();
    }
    pause() { this.isPlaying = false; if (this.playTimer) clearTimeout(this.playTimer); this.playTimer = null; this._updateControls(); }
    stepForward() { this.pause(); if (this.currentIdx < this.history.length - 1) { this.currentIdx++; this._render(this.currentIdx); } this._updateControls(); }
    stepBack() {
        this.pause();
        if (this.currentIdx > 0) {
            this.currentIdx--;
            this.followLatest = false;
            this._render(this.currentIdx);
        }
        this._updateControls();
    }
    goFirst() { this.pause(); this.seekTo(0); }
    goLast() { this.pause(); this.followLatest = true; this.seekTo(this.history.length - 1); }
    setSpeed(s) { this.playSpeed = s; }
    toggleParticles(on) { this.particlesOn = on; if (!on) this._clearParticles(); }
    toggleDecongestion(on) {
        this.decongestionOn = typeof on === 'boolean' ? on : !this.decongestionOn;
        if (this.currentIdx >= 0) this._render(this.currentIdx);
        this._updateControls();
    }
    toggle3D() { this.container.classList.toggle('view-3d'); }
    reset() {
        this.pause(); this.history = []; this.currentIdx = -1; this.built = false;
        this.isScrubbing = false;
        this.followLatest = true;
        this._clearParticles(); if (this.animFrame) cancelAnimationFrame(this.animFrame);
        if (this.container) { this.container.innerHTML = ''; this.container.style.height = ''; }
        this._updateControls();
    }

    /* ═══════ SAVE / DOWNLOAD ═══════ */

    /**
     * Save the current diagram as an SVG file.
     * @param {string} [filename='state_transition_diagram.svg']
     */
    saveSVG(filename) {
        if (!this.svg) return;
        const svgNode = this.svg.node();
        const serializer = new XMLSerializer();
        let svgString = serializer.serializeToString(svgNode);
        if (!svgString.includes('xmlns="http://www.w3.org/2000/svg"')) {
            svgString = svgString.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
        }
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || 'state_transition_diagram.svg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Save the current diagram as a PNG file.
     * @param {string} [filename='state_transition_diagram.png']
     * @param {number} [scale=2] - Resolution multiplier (2 = retina).
     */
    savePNG(filename, scale) {
        if (!this.svg) return;
        scale = scale || 2;
        const svgNode = this.svg.node();
        const w = +svgNode.getAttribute('width');
        const h = +svgNode.getAttribute('height');
        const serializer = new XMLSerializer();
        let svgString = serializer.serializeToString(svgNode);
        if (!svgString.includes('xmlns="http://www.w3.org/2000/svg"')) {
            svgString = svgString.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
        }
        const canvas = document.createElement('canvas');
        canvas.width = w * scale;
        canvas.height = h * scale;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        img.onload = function () {
            ctx.fillStyle = '#FAFBFC';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(url);
            canvas.toBlob(function (pngBlob) {
                const a = document.createElement('a');
                a.href = URL.createObjectURL(pngBlob);
                a.download = filename || 'state_transition_diagram.png';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
            }, 'image/png');
        };
        img.src = url;
    }

    /* ═══════ BUILD ═══════ */
    _build(A, B, pi) {
        this.container.innerHTML = '';
        this.N = A.length; this.M = B[0].length;
        const N = this.N, M = this.M;
        const STATE_COLORS = this.STATE_COLORS;
        const OBS_COLOR = this.OBS_COLOR;
        const PI_COLOR = this.PI_COLOR;

        // ── Dynamic sizing based on state/observation count ──
        const minNodeSpacing = 110;
        const baseRadius = 40;
        const R = N <= 4 ? baseRadius : Math.max(18, baseRadius - (N - 4) * 3.5);
        this._R = R;

        // Width scales with the larger of N states or M observations
        const stateGapIdeal = Math.max(R * 2.5, minNodeSpacing);
        const obsGapIdeal = Math.max(80, minNodeSpacing - 10);
        const neededW_states = (N - 1) * stateGapIdeal + 2 * R + 120;
        const neededW_obs = (M - 1) * obsGapIdeal + 160;
        const containerW = this.container.clientWidth || 860;
        const W = Math.max(containerW, neededW_states, neededW_obs);

        // Height: arc height uses sqrt scaling capped at 120px
        const maxDist = (N - 1) * stateGapIdeal;
        const maxArcH = Math.min(120, 30 + Math.sqrt(maxDist) * 4);
        const neededH_top = 50 + 18 + maxArcH + R + 60;
        const emissionGap = 140 + Math.max(0, N - 2) * 25;
        const neededH_bottom = R + 20 + emissionGap;
        const H = Math.max(390, neededH_top + R * 1.7 + neededH_bottom - 24);

        // Apply dynamic height to container
        this.container.style.height = H + 'px';

        this._svgW = W;
        this._svgH = H;
        this.svg = d3.select(this.container).append('svg')
            .attr('width', W).attr('height', H)
            .attr('viewBox', `0 0 ${W} ${H}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('font-family', this.fontFamily);
        const defs = this.svg.append('defs');

        // Gradients
        for (let i = 0; i < N; i++) {
            const c = STATE_COLORS[i % STATE_COLORS.length];
            const g = defs.append('radialGradient').attr('id', `sg${i}`)
                .attr('cx', '35%').attr('cy', '35%').attr('r', '65%');
            g.append('stop').attr('offset', '0%').attr('stop-color', c.light).attr('stop-opacity', 0.85);
            g.append('stop').attr('offset', '100%').attr('stop-color', c.grad[1]);
        }

        // Arrowheads — small, per-state colour
        for (let i = 0; i < N; i++) {
            const c = STATE_COLORS[i % STATE_COLORS.length];
            defs.append('marker').attr('id', `ah${i}`)
                .attr('viewBox', '0 -3 6 6').attr('refX', 5).attr('refY', 0)
                .attr('markerWidth', 5).attr('markerHeight', 5).attr('orient', 'auto')
                .append('path').attr('d', 'M0,-2.5L6,0L0,2.5Z').attr('fill', c.dark);
        }
        defs.append('marker').attr('id', 'ah-pi')
            .attr('viewBox', '0 -3 6 6').attr('refX', 5).attr('refY', 0)
            .attr('markerWidth', 4.5).attr('markerHeight', 4.5).attr('orient', 'auto')
            .append('path').attr('d', 'M0,-2L6,0L0,2Z').attr('fill', PI_COLOR.dark);
        defs.append('marker').attr('id', 'ah-em')
            .attr('viewBox', '0 -3 6 6').attr('refX', 5).attr('refY', 0)
            .attr('markerWidth', 4.5).attr('markerHeight', 4.5).attr('orient', 'auto')
            .append('path').attr('d', 'M0,-2L6,0L0,2Z').attr('fill', OBS_COLOR.dark);

        // Shadow
        const f = defs.append('filter').attr('id', 'shd')
            .attr('x', '-15%').attr('y', '-15%').attr('width', '130%').attr('height', '130%');
        f.append('feDropShadow').attr('dx', 0).attr('dy', 1).attr('stdDeviation', 2)
            .attr('flood-color', 'rgba(0,0,0,0.08)');

        /* ════════ GRID LAYOUT (Scalable 3-Layer) ════════ */
        const piY = 50;

        // Hidden states (Layer 2) — dynamic spacing & vertical position
        const stateGap = (W - 120) / Math.max(N, 1);
        const stateX0 = (W - (N - 1) * stateGap) / 2;
        const arcMaxDist = (N - 1) * stateGap;
        const arcClearance = Math.min(120, 30 + Math.sqrt(arcMaxDist) * 4);
        const stateY = piY + 18 + arcClearance + R + 30;
        this._sp = [];
        for (let i = 0; i < N; i++) this._sp.push({ x: stateX0 + i * stateGap, y: stateY });

        // Observations (Layer 3) — scaled node size
        const obsGap = Math.min(120, (W - 100) / Math.max(M, 1));
        const obsX0 = (W - (M - 1) * obsGap) / 2;
        const obsW = Math.max(50, 80 - Math.max(0, M - 4) * 5);
        const obsH = Math.max(30, 40 - Math.max(0, M - 6) * 2);
        const obsBottomPad = 18;
        const obsYTarget = Math.max(stateY + R + emissionGap, H * 0.9);
        const obsY = Math.min(obsYTarget, H - obsH / 2 - obsBottomPad);
        this._op = [];
        for (let k = 0; k < M; k++) this._op.push({ x: obsX0 + k * obsGap, y: obsY });

        /* ── Layers (Z-ordered groups for 3D) ── */
        this._gLayer4 = this.svg.append('g').attr('class', 'layer-4');
        this._gLayer3 = this.svg.append('g').attr('class', 'layer-3');
        this._gLayer2 = this.svg.append('g').attr('class', 'layer-2');
        this._gLayer1 = this.svg.append('g').attr('class', 'layer-1');

        this._emG = this._gLayer4;
        this._obsG = this._gLayer4;
        this._stateG = this._gLayer3;
        this._edG = this._gLayer2;
        this._ptG = this._gLayer2;
        this._piG = this._gLayer1;

        /* ── Labels ── */
        const lblSize = N > 8 ? '8px' : '10px';
        const lbl = (x, y, t) => this.svg.append('text').attr('x', x).attr('y', y)
            .attr('fill', '#94a3b8').attr('font-size', lblSize).attr('font-weight', '600')
            .attr('letter-spacing', '0.05em').text(t);
        lbl(20, piY + 4, 'START');
        lbl(20, stateY - R - 20, 'HIDDEN STATES');
        lbl(20, obsY - obsH / 2 - 20, 'OBSERVATIONS');

        /* ══════ START NODE (Layer 1) ══════ */
        this._piG.append('rect').attr('x', W / 2 - 50).attr('y', piY - 18).attr('width', 100).attr('height', 36)
            .attr('rx', 18).attr('fill', PI_COLOR.fill).attr('stroke', PI_COLOR.stroke)
            .attr('stroke-width', 2).attr('filter', 'url(#shd)');
        this._piG.append('text').attr('x', W / 2).attr('y', piY)
            .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
            .attr('font-size', '13px').attr('font-weight', '700').attr('fill', PI_COLOR.dark).text('START');

        // π arrows
        const piFontSize = N > 8 ? '8px' : N > 5 ? '9px' : '11px';
        this._piA = [];
        for (let i = 0; i < N; i++) {
            const s = this._sp[i];
            const sx = W / 2, sy = piY + 18;
            const ex = s.x, ey = s.y - R - 3;
            const bend = (ex - sx) * 0.06;
            const mx = (sx + ex) / 2 + bend, my = (sy + ey) / 2;
            const d = `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;
            const path = this._piG.append('path').attr('d', d)
                .attr('fill', 'none').attr('stroke', PI_COLOR.stroke)
                .attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3')
                .attr('marker-end', 'url(#ah-pi)').attr('opacity', 0.5);
            const hit = this._piG.append('path').attr('d', d)
                .attr('fill', 'none').attr('stroke', 'rgba(0,0,0,0)')
                .attr('stroke-width', 5).style('pointer-events', 'stroke')
                .attr('cursor', 'pointer');
            const lx = sx * 0.25 + mx * 0.5 + ex * 0.25, ly = sy * 0.25 + my * 0.5 + ey * 0.25 - 4;
            const label = this._piG.append('text').attr('x', lx).attr('y', ly)
                .attr('text-anchor', 'middle').attr('font-size', piFontSize)
                .attr('font-family', this.monoFont).attr('font-weight', '600')
                .attr('fill', PI_COLOR.dark).text(`π=${pi[i].toFixed(2)}`);
            this._piA.push({ path, hit, label });
            path.on('mouseover', (ev) => this._showTip(ev, `π[${i}]`))
                .on('mouseout', () => this._hideTip());
            hit.on('mouseover', (ev) => this._showTip(ev, `π[${i}]`))
                .on('mouseout', () => this._hideTip());
        }

        /* ══════ OBSERVATION NODES (Layer 4) ══════ */
        const obsFontSize = M > 8 ? '11px' : M > 5 ? '13px' : '16px';
        for (let k = 0; k < M; k++) {
            const o = this._op[k];
            this._obsG.append('rect').attr('x', o.x - obsW / 2).attr('y', o.y - obsH / 2)
                .attr('width', obsW).attr('height', obsH).attr('rx', 5)
                .attr('fill', OBS_COLOR.fill).attr('stroke', OBS_COLOR.stroke)
                .attr('stroke-width', 1.2).attr('filter', 'url(#shd)');
            this._obsG.append('text').attr('x', o.x).attr('y', o.y)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                .attr('font-size', obsFontSize).attr('font-weight', '700').attr('fill', OBS_COLOR.dark)
                .text(`O${k}`);
        }

        /* ══════ EMISSION ARROWS (Sigmoid, Layer 3) ══════ */
        const emFontSize = N > 8 ? '8px' : N > 5 ? '9px' : '11px';
        const emStroke = N > 6 ? 1.5 : 2;
        this._emR = {};
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < M; k++) {
                const sp = this._sp[i], op = this._op[k];
                const col = STATE_COLORS[i % STATE_COLORS.length];

                const sx = sp.x, sy = sp.y + R;
                const ex = op.x, ey = op.y - obsH / 2;

                const dist = ey - sy;
                const c1x = sx, c1y = sy + dist * 0.5;
                const c2x = ex, c2y = ey - dist * 0.5;

                const d = `M${sx},${sy} C${c1x},${c1y} ${c2x},${c2y} ${ex},${ey}`;

                const path = this._emG.append('path').attr('d', d)
                    .attr('fill', 'none').attr('stroke', col.base)
                    .attr('stroke-width', emStroke).attr('opacity', 0.4)
                    .attr('stroke-dasharray', '4 3').attr('marker-end', 'url(#ah-em)');
                const hit = this._emG.append('path').attr('d', d)
                    .attr('fill', 'none').attr('stroke', 'rgba(0,0,0,0)')
                    .attr('stroke-width', 5).style('pointer-events', 'stroke')
                    .attr('cursor', 'pointer');

                const lx = (sx + ex) / 2;
                const ly = (sy + ey) / 2;

                const label = this._emG.append('text').attr('x', lx).attr('y', ly)
                    .attr('text-anchor', 'middle').attr('font-size', emFontSize)
                    .attr('font-family', this.monoFont).attr('font-weight', '600')
                    .attr('fill', col.dark).attr('opacity', 0).text('');

                this._emR[`em-${i}-${k}`] = {
                    path, hit, label, sx, sy, c1x, c1y, c2x, c2y, ex, ey,
                    baseC1x: c1x, baseC1y: c1y, baseC2x: c2x, baseC2y: c2y
                };
                path.on('mouseover', (ev) => this._showTip(ev, `B[${i}][${k}]`))
                    .on('mouseout', () => this._hideTip());
                hit.on('mouseover', (ev) => this._showTip(ev, `B[${i}][${k}]`))
                    .on('mouseout', () => this._hideTip());
            }
        }

        /* ══════ TRANSITION ARROWS (Layer 2) ══════ */
        this._trR = {};
        this._slR = {};

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const key = `${i}-${j}`;
                const col = STATE_COLORS[i % STATE_COLORS.length];

                if (i === j) {
                    // Self-loop: scaled to radius
                    const sp = this._sp[i];
                    const loopW = R * 0.5;
                    const loopH = R * 1.25;

                    const sl_sx = sp.x + loopW, sl_sy = sp.y - R + 5;
                    const sl_ex = sp.x - loopW, sl_ey = sp.y - R + 5;
                    const sl_c1x = sp.x + loopW + R * 0.5, sl_c1y = sp.y - R - loopH;
                    const sl_c2x = sp.x - loopW - R * 0.5, sl_c2y = sp.y - R - loopH;

                    const dLoop = `M${sl_sx},${sl_sy} C${sl_c1x},${sl_c1y} ${sl_c2x},${sl_c2y} ${sl_ex},${sl_ey}`;

                    const path = this._edG.append('path').attr('d', dLoop)
                        .attr('fill', 'none').attr('stroke', col.base)
                        .attr('stroke-width', N > 6 ? 1.5 : 2).attr('opacity', 0.5)
                        .attr('marker-end', `url(#ah${i})`);
                    const hit = this._edG.append('path').attr('d', dLoop)
                        .attr('fill', 'none').attr('stroke', 'rgba(0,0,0,0)')
                        .attr('stroke-width', 5).style('pointer-events', 'stroke')
                        .attr('cursor', 'pointer');

                    const slLabelFontSize = N > 8 ? '8px' : N > 5 ? '9px' : '11px';
                    const label = this._edG.append('text').attr('x', sp.x).attr('y', sp.y - R - loopH)
                        .attr('text-anchor', 'middle').attr('font-size', slLabelFontSize)
                        .attr('font-weight', '600').attr('fill', col.dark).text('0.00');
                    this._slR[key] = { path, hit, label };
                    hit.on('mouseover', (ev) => this._showTip(ev, `A[${i}][${j}]`))
                        .on('mouseout', () => this._hideTip());

                } else {
                    // Inter-state arc — forward (above) & reverse (below)
                    const src = this._sp[i], tgt = this._sp[j];
                    const dist = Math.abs(tgt.x - src.x);
                    const arcH = Math.min(120, 30 + Math.sqrt(dist) * 4);
                    const midX = (src.x + tgt.x) / 2;

                    let sx, sy, ex, ey, cx, cy;
                    if (i < j) {
                        sx = src.x + R * 0.5; sy = src.y - R * 0.8;
                        ex = tgt.x - R * 0.5; ey = tgt.y - R * 0.8;
                        cx = midX;
                        cy = Math.min(src.y, tgt.y) - R - arcH;
                    } else {
                        sx = src.x - R * 0.5; sy = src.y + R * 0.8;
                        ex = tgt.x + R * 0.5; ey = tgt.y + R * 0.8;
                        cx = midX;
                        cy = Math.max(src.y, tgt.y) + R + arcH * 0.45;
                    }

                    const d = `M${sx},${sy} Q${cx},${cy} ${ex},${ey}`;

                    const trStroke = N > 6 ? 1.5 : 2;
                    const path = this._edG.append('path').attr('d', d)
                        .attr('fill', 'none').attr('stroke', col.base)
                        .attr('stroke-width', trStroke).attr('opacity', 0.35)
                        .attr('marker-end', `url(#ah${i})`);
                    const hit = this._edG.append('path').attr('d', d)
                        .attr('fill', 'none').attr('stroke', 'rgba(0,0,0,0)')
                        .attr('stroke-width', 5).style('pointer-events', 'stroke')
                        .attr('cursor', 'pointer');

                    const lx = 0.25 * sx + 0.5 * cx + 0.25 * ex;
                    const lyOff = i < j ? -5 : 5;
                    const ly = 0.25 * sy + 0.5 * cy + 0.25 * ey + lyOff;

                    const trFontSize = N > 8 ? '8px' : N > 5 ? '9px' : '11px';
                    const bgW = N > 8 ? 24 : 30;
                    const bgH = N > 8 ? 13 : 16;
                    const bg = this._edG.append('rect').attr('x', lx - bgW / 2).attr('y', ly - bgH / 2)
                        .attr('width', bgW).attr('height', bgH).attr('rx', 8)
                        .attr('fill', 'rgba(255,255,255,0.9)').attr('stroke', '#e2e8f0')
                        .attr('stroke-width', 0.5).attr('opacity', 0);

                    const label = this._edG.append('text').attr('x', lx).attr('y', ly)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-size', trFontSize).attr('font-weight', '700')
                        .attr('font-family', this.monoFont)
                        .attr('fill', col.dark).attr('opacity', 0).text('');

                    this._trR[key] = {
                        path, hit, label, bg,
                        sx, sy, cx, cy, ex, ey,
                        baseCy: cy,
                        yDir: i < j ? -1 : 1
                    };

                    path.on('mouseover', (ev) => this._showTip(ev, `A[${i}][${j}]`))
                        .on('mouseout', () => this._hideTip());
                    hit.on('mouseover', (ev) => this._showTip(ev, `A[${i}][${j}]`))
                        .on('mouseout', () => this._hideTip());
                }
            }
        }

        /* ══════ STATE CIRCLES (Layer 3) ══════ */
        const stateFontSize = N > 10 ? '11px' : N > 6 ? '14px' : '20px';
        this._sn = [];
        for (let i = 0; i < N; i++) {
            const s = this._sp[i], c = STATE_COLORS[i % STATE_COLORS.length];
            const circle = this._stateG.append('circle').attr('cx', s.x).attr('cy', s.y).attr('r', R)
                .attr('fill', `url(#sg${i})`).attr('stroke', c.dark).attr('stroke-width', N > 8 ? 1.5 : 2)
                .attr('filter', 'url(#shd)').attr('cursor', 'pointer');
            this._stateG.append('text').attr('x', s.x).attr('y', s.y)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                .attr('font-size', stateFontSize).attr('font-weight', '700').attr('fill', '#fff')
                .attr('pointer-events', 'none').style('text-shadow', '0 1px 2px rgba(0,0,0,0.3)')
                .text(`S${i}`);
            this._sn.push(circle);
            circle.on('click', () => this._inspect(i))
                .on('mouseover', function () { d3.select(this).attr('stroke-width', 3); })
                .on('mouseout', function () { d3.select(this).attr('stroke-width', 2); });
        }

        // Tooltip div
        this._tip = d3.select(this.container).append('div')
            .style('position', 'absolute').style('display', 'none')
            .style('background', 'rgba(15,23,42,0.9)').style('color', '#f1f5f9')
            .style('padding', '3px 8px').style('border-radius', '5px')
            .style('font-family', this.monoFont).style('font-size', '10px')
            .style('pointer-events', 'none').style('z-index', '50');

        this._startParticleLoop();
    }

    /* ═══════ RENDER ═══════ */
    _render(idx) {
        if (idx < 0 || idx >= this.history.length || !this.built) return;
        const { A, B, pi } = this.history[idx];
        const N = this.N, M = this.M;
        const isFiltered = this.decongestionOn;

        const prominentTransitions = Array.from({ length: N }, () => new Set());
        const prominentEmissions = Array.from({ length: N }, () => new Set());
        const transitionLanes = Array.from({ length: N }, () => ({}));
        const emissionLanes = Array.from({ length: N }, () => ({}));
        for (let i = 0; i < N; i++) {
            const rankedTransitions = [];
            for (let j = 0; j < N; j++) {
                if (i !== j) rankedTransitions.push({ j, v: A[i][j] });
            }
            rankedTransitions.sort((a, b) => b.v - a.v);

            if (isFiltered) {
                rankedTransitions.slice(0, 2).forEach(({ j }) => prominentTransitions[i].add(j));
                rankedTransitions.forEach(({ j, v }) => {
                    if (v >= 0.35) prominentTransitions[i].add(j);
                });
            } else {
                const upper = rankedTransitions.filter(({ j }) => i < j);
                const lower = rankedTransitions.filter(({ j }) => i > j);
                upper.forEach(({ j }, laneIdx) => { transitionLanes[i][j] = laneIdx; });
                lower.forEach(({ j }, laneIdx) => { transitionLanes[i][j] = laneIdx; });
            }

            const rankedEmissions = [];
            for (let k = 0; k < M; k++) rankedEmissions.push({ k, v: B[i][k] });
            rankedEmissions.sort((a, b) => b.v - a.v);
            if (isFiltered) {
                rankedEmissions.slice(0, 2).forEach(({ k }) => prominentEmissions[i].add(k));
                rankedEmissions.forEach(({ k, v }) => {
                    if (v >= 0.35) prominentEmissions[i].add(k);
                });
            } else {
                rankedEmissions.forEach(({ k }, laneIdx) => {
                    emissionLanes[i][k] = laneIdx;
                });
            }
        }

        // π
        for (let i = 0; i < N; i++) {
            const r = this._piA[i], v = pi[i];
            r.path.transition().duration(180).attr('stroke-width', 1.5 + v * 5).attr('opacity', Math.max(0.25, 0.2 + v * 0.7));
            r.label.text(`π=${v.toFixed(2)}`).attr('font-size', '12px').attr('font-weight', '800');
        }

        // Transitions — BOLD visibility
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
            const v = A[i][j], k = `${i}-${j}`;
            if (i === j) {
                const r = this._slR[k]; if (!r) continue;
                r.path.transition().duration(180)
                    .attr('stroke-width', 1 + v * 6)
                    .attr('opacity', Math.max(0.12, 0.10 + v * 0.8));
                r.label.text(v.toFixed(2));
            } else {
                const r = this._trR[k]; if (!r) continue;
                const laneIdx = transitionLanes[i][j] || 0;
                const laneOffset = isFiltered ? 0 : laneIdx * 10;
                const valueOffset = isFiltered ? 0 : (1 - v) * 18;
                const dynamicCy = r.baseCy + r.yDir * (laneOffset + valueOffset);
                const dynamicPath = `M${r.sx},${r.sy} Q${r.cx},${dynamicCy} ${r.ex},${r.ey}`;
                const lyOff = r.yDir < 0 ? -5 : 5;
                const ly = 0.25 * r.sy + 0.5 * dynamicCy + 0.25 * r.ey + lyOff;

                const isProminent = !isFiltered || prominentTransitions[i].has(j);
                const opacity = isFiltered
                    ? (isProminent ? Math.max(0.22, 0.2 + v * 0.7) : 0.04)
                    : Math.max(0.16, 0.14 + v * 0.72);
                const strokeWidth = isFiltered
                    ? (isProminent ? (1 + v * 6) : 0.8)
                    : (0.9 + v * 5);
                r.path.transition().duration(180)
                    .attr('d', dynamicPath)
                    .attr('stroke-width', strokeWidth)
                    .attr('opacity', opacity);
                if (r.hit) r.hit.attr('d', dynamicPath);
                r.cy = dynamicCy;

                const showLbl = isFiltered ? (isProminent && v > 0.02) : (v > 0.08);
                r.label.text(v.toFixed(2)).attr('x', r.cx).attr('y', ly).attr('opacity', showLbl ? 1 : 0)
                    .attr('font-size', '11px').attr('font-weight', '700');
                r.bg.attr('x', r.cx - 15).attr('y', ly - 8).attr('opacity', showLbl ? 1 : 0);
            }
        }

        // Emissions — visible
        for (let i = 0; i < N; i++) for (let k = 0; k < M; k++) {
            const v = B[i][k], r = this._emR[`em-${i}-${k}`]; if (!r) continue;
            const laneIdx = emissionLanes[i][k] || 0;
            const laneOffset = isFiltered ? 0 : laneIdx * 8;
            const strengthOffset = isFiltered ? 0 : (1 - v) * 14;
            const dir = r.ex >= r.sx ? 1 : -1;
            const spread = laneOffset + strengthOffset;
            const dynamicC1x = isFiltered ? r.baseC1x : (r.baseC1x + dir * spread * 0.4);
            const dynamicC2x = isFiltered ? r.baseC2x : (r.baseC2x + dir * spread * 0.9);
            const dynamicC1y = isFiltered ? r.baseC1y : (r.baseC1y + spread * 0.55);
            const dynamicC2y = isFiltered ? r.baseC2y : (r.baseC2y - spread * 0.15);
            const dynamicPath = `M${r.sx},${r.sy} C${dynamicC1x},${dynamicC1y} ${dynamicC2x},${dynamicC2y} ${r.ex},${r.ey}`;
            const lx = 0.125 * r.sx + 0.375 * dynamicC1x + 0.375 * dynamicC2x + 0.125 * r.ex;
            const ly = 0.125 * r.sy + 0.375 * dynamicC1y + 0.375 * dynamicC2y + 0.125 * r.ey;

            r.c1x = dynamicC1x; r.c1y = dynamicC1y;
            r.c2x = dynamicC2x; r.c2y = dynamicC2y;

            const isProminent = !isFiltered || prominentEmissions[i].has(k);
            const opacity = isFiltered
                ? (isProminent ? Math.max(0.35, 0.35 + v * 0.6) : 0.03)
                : Math.max(0.18, 0.2 + v * 0.55);
            const strokeWidth = isFiltered
                ? (isProminent ? (1 + v * 5) : 0.7)
                : (0.8 + v * 4);
            r.path.transition().duration(180)
                .attr('d', dynamicPath)
                .attr('stroke-width', strokeWidth)
                .attr('opacity', opacity);
            if (r.hit) r.hit.attr('d', dynamicPath);

            const showLbl = isFiltered ? (isProminent && v > 0.08) : (v > 0.12);
            r.label.text(showLbl ? v.toFixed(2) : '').attr('x', lx).attr('y', ly).attr('opacity', showLbl ? 1 : 0)
                .attr('font-size', '11px').attr('font-weight', '700');
        }

        this._rebuildParticles(A);
        if (this.inspectorEl?.classList.contains('visible')) this._renderInspector(idx);
    }

    /* ═══════ PARTICLES ═══════ */
    _rebuildParticles(A) {
        this._clearParticles(); if (!this.particlesOn) return;
        const STATE_COLORS = this.STATE_COLORS;
        const isFiltered = this.decongestionOn;
        const prominentTransitions = Array.from({ length: this.N }, () => new Set());
        if (isFiltered) {
            for (let i = 0; i < this.N; i++) {
                const ranked = [];
                for (let j = 0; j < this.N; j++) {
                    if (i !== j) ranked.push({ j, v: A[i][j] });
                }
                ranked.sort((a, b) => b.v - a.v);
                ranked.slice(0, 2).forEach(({ j }) => prominentTransitions[i].add(j));
                ranked.forEach(({ j, v }) => {
                    if (v >= 0.35) prominentTransitions[i].add(j);
                });
            }
        }

        for (let i = 0; i < this.N; i++) for (let j = 0; j < this.N; j++) {
            if (i === j) continue;
            const v = A[i][j];
            if (isFiltered) {
                if (!prominentTransitions[i].has(j) || v < 0.03) continue;
            } else if (v < 0.02) {
                continue;
            }
            const r = this._trR[`${i}-${j}`]; if (!r) continue;
            const c = STATE_COLORS[i % STATE_COLORS.length];
            const n = Math.max(1, Math.round(v * 3));
            for (let p = 0; p < n; p++) {
                this.particles.push({
                    t: p / n, speed: 0.002 + v * 0.004,
                    el: this._ptG.append('circle').attr('r', 2).attr('fill', c.base)
                        .attr('opacity', 0.75).style('filter', `drop-shadow(0 0 2px ${c.base})`)
                        .attr('pointer-events', 'none'), ref: r
                });
            }
        }
    }
    _startParticleLoop() {
        const tick = () => {
            this.animFrame = requestAnimationFrame(tick); if (!this.particlesOn) return;
            for (const p of this.particles) {
                p.t += p.speed; if (p.t > 1) p.t -= 1;
                const t = p.t, m = 1 - t, m2 = m * m, m3 = m2 * m, t2 = t * t, t3 = t2 * t;
                const r = p.ref;
                if (r.cx !== undefined) {
                    const x = m2 * r.sx + 2 * m * t * r.cx + t2 * r.ex;
                    const y = m2 * r.sy + 2 * m * t * r.cy + t2 * r.ey;
                    p.el.attr('cx', x).attr('cy', y);
                } else if (r.p2x !== undefined) {
                    let x, y;
                    if (t < 0.3) {
                        const st = t / 0.3;
                        x = r.p1x; y = r.p1y + (r.p2y - r.p1y) * st;
                    } else {
                        const st = (t - 0.3) / 0.7;
                        x = r.p2x + (r.p3x - r.p2x) * st;
                        y = r.p2y + (r.p3y - r.p2y) * st;
                    }
                    p.el.attr('cx', x).attr('cy', y);
                } else {
                    const x = m3 * r.sx + 3 * m2 * t * r.c1x + 3 * m * t2 * r.c2x + t3 * r.ex;
                    const y = m3 * r.sy + 3 * m2 * t * r.c1y + 3 * m * t2 * r.c2y + t3 * r.ey;
                    p.el.attr('cx', x).attr('cy', y);
                }
            }
        }; tick();
    }
    _clearParticles() { for (const p of this.particles) p.el.remove(); this.particles = []; }

    /* ═══════ REPLAY ═══════ */
    _playStep() {
        if (!this.isPlaying) return;
        if (this.currentIdx < this.history.length - 1) {
            this.currentIdx++; this._render(this.currentIdx); this._updateControls();
            this.playTimer = setTimeout(() => this._playStep(), Math.max(50, 500 / this.playSpeed));
        } else this.pause();
    }

    /* ═══════ INSPECTOR ═══════ */
    _inspect(si) { if (!this.inspectorEl || this.currentIdx < 0) return; this.inspectorEl.classList.add('visible'); this._renderInspector(this.currentIdx, si); }
    _renderInspector(ii, hl) {
        if (!this.inspectorEl) return; const d = this.history[ii]; if (!d) return;
        const N = d.A.length, M = d.B[0].length;
        const STATE_COLORS = this.STATE_COLORS;
        let h = `<h4>Iteration ${d.iteration} &nbsp;|&nbsp; LL = ${d.log_likelihood.toFixed(4)}</h4>`;
        h += `<div style="margin-bottom:6px"><strong>π:</strong> [${d.pi.map((v, i) =>
            `<span style="color:${STATE_COLORS[i % STATE_COLORS.length].dark}">${v.toFixed(4)}</span>`).join(', ')}]</div>`;
        h += `<div style="margin-bottom:6px"><strong>A:</strong><table><tr><th></th>`;
        for (let j = 0; j < N; j++) h += `<th>S${j}</th>`; h += `</tr>`;
        for (let i = 0; i < N; i++) {
            h += `<tr style="${i === hl ? 'background:#fffbeb;' : ''}"><th>S${i}</th>`;
            for (let j = 0; j < N; j++) { const v = d.A[i][j]; h += `<td style="${v > 0.3 ? 'font-weight:700;' : ''}color:${STATE_COLORS[i % STATE_COLORS.length].dark}">${v.toFixed(4)}</td>`; }
            h += `</tr>`;
        } h += `</table></div>`;
        h += `<strong>B:</strong><table><tr><th></th>`;
        for (let k = 0; k < M; k++) h += `<th>O${k}</th>`; h += `</tr>`;
        for (let i = 0; i < N; i++) {
            h += `<tr style="${i === hl ? 'background:#f0fdf4;' : ''}"><th>S${i}</th>`;
            for (let k = 0; k < M; k++) { const v = d.B[i][k]; h += `<td style="${v > 0.3 ? 'font-weight:700;' : ''}">${v.toFixed(4)}</td>`; }
            h += `</tr>`;
        } h += `</table>`;
        this.inspectorEl.innerHTML = h;
    }

    /* ═══════ TOOLTIP ═══════ */
    _showTip(ev, txt) {
        if (!this._tip) return; let f = txt;
        if (this.currentIdx >= 0) {
            const d = this.history[this.currentIdx];
            const mA = txt.match(/^A\[(\d+)\]\[(\d+)\]$/);
            const mB = txt.match(/^B\[(\d+)\]\[(\d+)\]$/);
            const mPi = txt.match(/^π\[(\d+)\]$/);
            if (mA) f = `${txt} = ${d.A[+mA[1]][+mA[2]].toFixed(6)}`;
            else if (mB) f = `${txt} = ${d.B[+mB[1]][+mB[2]].toFixed(6)}`;
            else if (mPi) f = `${txt} = ${d.pi[+mPi[1]].toFixed(6)}`;
        }
        const r = this.container.getBoundingClientRect();
        this._tip.style('display', 'block').text(f).style('left', (ev.clientX - r.left + 10) + 'px').style('top', (ev.clientY - r.top - 22) + 'px');
    }
    _hideTip() { if (this._tip) this._tip.style('display', 'none'); }

    /* ═══════ CONTROLS ═══════ */
    wireControls(c) {
        this._ctrl = c;
        c.btnFirst?.addEventListener('click', () => this.goFirst());
        c.btnBack?.addEventListener('click', () => this.stepBack());
        c.btnPlay?.addEventListener('click', () => {
            if (this.isPlaying) this.pause();
            else { if (this.currentIdx >= this.history.length - 1) this.currentIdx = 0; this.play(); }
        });
        c.btnForward?.addEventListener('click', () => this.stepForward());
        c.btnLast?.addEventListener('click', () => this.goLast());
        const syncPlaySpeedFromSelect = () => {
            if (!c.speedSelect) return;
            const selectedSpeed = parseFloat(c.speedSelect.value);
            if (Number.isFinite(selectedSpeed) && selectedSpeed > 0) this.setSpeed(selectedSpeed);
        };
        c.speedSelect?.addEventListener('change', syncPlaySpeedFromSelect);
        syncPlaySpeedFromSelect();
        const scrubToIndex = (idx) => {
            const safeIdx = Number(idx);
            if (!Number.isFinite(safeIdx)) return;
            this.seekTo(Math.round(safeIdx));
        };
        const scrubAtClientX = (clientX) => {
            if (!c.timeline || !this.history.length) return;
            const rect = c.timeline.getBoundingClientRect();
            const width = Math.max(1, rect.width);
            const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / width));
            const maxIdx = Math.max(0, this.history.length - 1);
            const idx = Math.round(ratio * maxIdx);
            c.timeline.value = String(idx);
            this.seekTo(idx);
        };
        const beginScrub = () => {
            this.isScrubbing = true;
            this.followLatest = false;
            this.pause();
        };
        const endScrub = () => {
            this.isScrubbing = false;
        };
        c.timeline?.addEventListener('pointerdown', (e) => {
            beginScrub();
            scrubAtClientX(e.clientX);
        });
        c.timeline?.addEventListener('pointermove', (e) => {
            if (!this.isScrubbing) return;
            if ((e.buttons & 1) !== 1) {
                endScrub();
                return;
            }
            scrubAtClientX(e.clientX);
        });
        c.timeline?.addEventListener('mousedown', beginScrub);
        c.timeline?.addEventListener('touchstart', (e) => {
            beginScrub();
            const t = e.touches?.[0];
            if (t) scrubAtClientX(t.clientX);
        }, { passive: true });
        c.timeline?.addEventListener('touchmove', (e) => {
            if (!this.isScrubbing) return;
            if (!e.touches || e.touches.length === 0) {
                endScrub();
                return;
            }
            const t = e.touches?.[0];
            if (t) scrubAtClientX(t.clientX);
        }, { passive: true });
        c.timeline?.addEventListener('input', e => { scrubToIndex(e.target.value); });
        c.timeline?.addEventListener('change', e => { endScrub(); scrubToIndex(e.target.value); });
        c.timeline?.addEventListener('pointerup', endScrub);
        window.addEventListener('pointerup', endScrub);
        window.addEventListener('pointercancel', endScrub);
        window.addEventListener('mouseup', endScrub);
        window.addEventListener('touchend', endScrub, { passive: true });
        c.btnParticles?.addEventListener('click', () => {
            this.particlesOn = !this.particlesOn;
            if (!this.particlesOn) this._clearParticles();
            else if (this.currentIdx >= 0) this._rebuildParticles(this.history[this.currentIdx].A);
            c.btnParticles.classList.toggle('active', this.particlesOn);
        });
        c.btnDecongestion?.addEventListener('click', () => this.toggleDecongestion());
        c.btn3D?.addEventListener('click', () => {
            this.toggle3D();
            c.btn3D.classList.toggle('active');
        });

        // Fullscreen resize handling
        this._fsHandler = () => this._handleFullscreenChange();
        document.addEventListener('fullscreenchange', this._fsHandler);
        document.addEventListener('webkitfullscreenchange', this._fsHandler);
    }

    _handleFullscreenChange() {
        if (!this.svg) return;
        const svgNode = this.svg.node();
        const card = this.container.closest('#diagram-card');
        const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);

        if (isFs && card) {
            // In fullscreen: remove fixed dimensions and let CSS flex handle sizing
            svgNode.removeAttribute('width');
            svgNode.removeAttribute('height');
            this.container.style.height = '';
        } else {
            // Exiting fullscreen: restore original fixed dimensions
            svgNode.setAttribute('width', this._svgW);
            svgNode.setAttribute('height', this._svgH);
            this.container.style.height = this._svgH + 'px';
        }
    }
    _updateControls() {
        const c = this._ctrl; if (!c) return;
        const l = this.history.length, i = this.currentIdx;
        if (c.btnFirst) c.btnFirst.disabled = i <= 0;
        if (c.btnBack) c.btnBack.disabled = i <= 0;
        if (c.btnForward) c.btnForward.disabled = i >= l - 1;
        if (c.btnLast) c.btnLast.disabled = i >= l - 1;
        if (c.btnPlay) { c.btnPlay.innerHTML = this.isPlaying ? '⏸' : '▶'; c.btnPlay.disabled = l === 0; }
        if (c.btnDecongestion) c.btnDecongestion.classList.toggle('active', this.decongestionOn);
        if (c.decongestionLabel) c.decongestionLabel.textContent = this.decongestionOn ? 'Filtered' : 'All';
        if (c.timeline) {
            c.timeline.max = Math.max(0, l - 1); c.timeline.value = Math.max(0, i);
            c.timeline.style.setProperty('--progress', l > 1 ? (i / (l - 1)) * 100 + '%' : '0%');
        }
        if (c.iterLabel) c.iterLabel.textContent = l > 0 ? `Step ${i + 1} / ${l}` : 'No data';
    }
}

/* ── Backward-compatibility alias ── */
const HMMDiagram = StateTransitionDiagram;

/* ── Export ── */
if (typeof window !== 'undefined') {
    window.StateTransitionDiagram = StateTransitionDiagram;
    window.HMMDiagram = HMMDiagram;
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { StateTransitionDiagram, HMMDiagram };
}
