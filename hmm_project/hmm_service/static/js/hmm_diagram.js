/**
 * hmm_diagram.js — Interactive Animated HMM State Transition Diagram
 * ===================================================================
 * Compact 3-tier layout:  π (Start) → Hidden States → Observations
 *
 * Fixes applied:
 *   • Compact vertical layout (no huge gaps)
 *   • Bold, visible arrows (higher opacity & stroke-width)
 *   • Wide-separated bidirectional arcs (no overlap)
 *   • Self-loops clearly visible on outer edges
 *   • Emission arrows are short with clear labels
 *   • Particle flow animation, replay, inspector
 */

const STATE_COLORS = [
    { base: '#F59E0B', light: '#FDE68A', dark: '#92400E', grad: ['#FBBF24', '#D97706'] },
    { base: '#3B82F6', light: '#93C5FD', dark: '#1E3A8A', grad: ['#60A5FA', '#2563EB'] },
    { base: '#10B981', light: '#6EE7B7', dark: '#064E3B', grad: ['#34D399', '#059669'] },
    { base: '#F43F5E', light: '#FDA4AF', dark: '#881337', grad: ['#FB7185', '#E11D48'] },
    { base: '#8B5CF6', light: '#C4B5FD', dark: '#4C1D95', grad: ['#A78BFA', '#7C3AED'] },
    { base: '#06B6D4', light: '#67E8F9', dark: '#155E75', grad: ['#22D3EE', '#0891B2'] },
];
const OBS_COLOR = { fill: '#F1F5F9', stroke: '#94A3B8', dark: '#334155' };
const PI_COLOR = { fill: '#F5F3FF', stroke: '#8B5CF6', dark: '#5B21B6' };

class HMMDiagram {
    constructor(containerId, inspectorId) {
        this.container = document.querySelector(containerId);
        this.inspectorEl = inspectorId ? document.querySelector(inspectorId) : null;
        this.history = []; this.currentIdx = -1;
        this.isPlaying = false; this.playTimer = null; this.playSpeed = 1;
        this.particlesOn = true; this.built = false;
        this.particles = []; this.animFrame = null;
        this.svg = null; this.N = 0; this.M = 0; this._ctrl = null;
    }

    /* ═══════ PUBLIC API ═══════ */
    feedIteration(data) {
        this.history.push({ A: data.A, B: data.B, pi: data.pi, iteration: data.iteration, log_likelihood: data.log_likelihood });
        if (!this.built) { this._build(data.A, data.B, data.pi); this.built = true; }
        if (!this.isPlaying || this.currentIdx === this.history.length - 2) {
            this.currentIdx = this.history.length - 1; this._render(this.currentIdx);
        }
        this._updateControls();
    }
    onComplete() { this.pause(); this._updateControls(); }
    seekTo(i) { if (i < 0 || i >= this.history.length) return; this.currentIdx = i; this._render(i); this._updateControls(); }
    play() { if (!this.history.length) return; this.isPlaying = true; this._updateControls(); this._playStep(); }
    pause() { this.isPlaying = false; if (this.playTimer) clearTimeout(this.playTimer); this.playTimer = null; this._updateControls(); }
    stepForward() { this.pause(); if (this.currentIdx < this.history.length - 1) { this.currentIdx++; this._render(this.currentIdx); } this._updateControls(); }
    stepBack() { this.pause(); if (this.currentIdx > 0) { this.currentIdx--; this._render(this.currentIdx); } this._updateControls(); }
    goFirst() { this.pause(); this.seekTo(0); }
    goLast() { this.pause(); this.seekTo(this.history.length - 1); }
    setSpeed(s) { this.playSpeed = s; }
    toggleParticles(on) { this.particlesOn = on; if (!on) this._clearParticles(); }
    reset() {
        this.pause(); this.history = []; this.currentIdx = -1; this.built = false;
        this._clearParticles(); if (this.animFrame) cancelAnimationFrame(this.animFrame);
        if (this.container) this.container.innerHTML = ''; this._updateControls();
    }

    /* ═══════ BUILD ═══════ */
    _build(A, B, pi) {
        this.container.innerHTML = '';
        const W = this.container.clientWidth || 860;
        const H = this.container.clientHeight || 520;
        this.N = A.length; this.M = B[0].length;
        const N = this.N, M = this.M;

        // Adaptive radius
        const R = 45; // Smaller states (was 60)
        this._R = R;

        this.svg = d3.select(this.container).append('svg')
            .attr('width', W).attr('height', H)
            .attr('viewBox', `0 0 ${W} ${H}`)
            .style('font-family', "'Inter', system-ui, sans-serif");
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
                .attr('markerWidth', 4).attr('markerHeight', 4).attr('orient', 'auto')
                .append('path').attr('d', 'M0,-2.5L6,0L0,2.5Z').attr('fill', c.dark);
        }
        defs.append('marker').attr('id', 'ah-pi')
            .attr('viewBox', '0 -3 6 6').attr('refX', 5).attr('refY', 0)
            .attr('markerWidth', 3.5).attr('markerHeight', 3.5).attr('orient', 'auto')
            .append('path').attr('d', 'M0,-2L6,0L0,2Z').attr('fill', PI_COLOR.dark);
        defs.append('marker').attr('id', 'ah-em')
            .attr('viewBox', '0 -3 6 6').attr('refX', 5).attr('refY', 0)
            .attr('markerWidth', 3.5).attr('markerHeight', 3.5).attr('orient', 'auto')
            .append('path').attr('d', 'M0,-2L6,0L0,2Z').attr('fill', OBS_COLOR.dark);

        // Shadow
        const f = defs.append('filter').attr('id', 'shd')
            .attr('x', '-15%').attr('y', '-15%').attr('width', '130%').attr('height', '130%');
        f.append('feDropShadow').attr('dx', 0).attr('dy', 1).attr('stdDeviation', 2)
            .attr('flood-color', 'rgba(0,0,0,0.08)');

        /* ════════ COMPACT LAYOUT ════════
         *   START       y = 40
         *   States      y = 150
         *   Observations y = 310
         *   (for H=520, leaves room at bottom for labels)
         */
        const piY = 38;
        const stateY = Math.min(H * 0.40, 280);
        const obsY = Math.min(H * 0.70, 500); // Higher up to shorten arrows

        // Hidden states — spread horizontally
        const stateGap = Math.min(200, (W - 120) / Math.max(N - 1, 1));
        const stateX0 = (W - (N - 1) * stateGap) / 2;
        this._sp = [];
        for (let i = 0; i < N; i++) this._sp.push({ x: stateX0 + i * stateGap, y: stateY });

        // Observations — spread horizontally
        const obsGap = Math.min(100, (W - 80) / Math.max(M - 1, 1));
        const obsX0 = (W - (M - 1) * obsGap) / 2;
        this._op = [];
        const obsW = 60, obsH = 30; // Larger observations
        for (let k = 0; k < M; k++) this._op.push({ x: obsX0 + k * obsGap, y: obsY });

        /* ── Layers ── */
        this._emG = this.svg.append('g');
        this._piG = this.svg.append('g');
        this._edG = this.svg.append('g');
        this._ptG = this.svg.append('g');
        this._ndG = this.svg.append('g');

        /* ── Labels ── */
        const lbl = (x, y, t) => this.svg.append('text').attr('x', x).attr('y', y)
            .attr('fill', '#94a3b8').attr('font-size', '8px').attr('font-weight', '600')
            .attr('letter-spacing', '0.08em').text(t);
        lbl(10, piY + 4, 'INITIAL (π)');
        lbl(10, stateY - R - 12, 'HIDDEN STATES');
        lbl(10, obsY - obsH / 2 - 10, 'OBSERVATIONS');

        /* ══════ START NODE ══════ */
        this._ndG.append('rect').attr('x', W / 2 - 40).attr('y', piY - 14).attr('width', 80).attr('height', 28)
            .attr('rx', 14).attr('fill', PI_COLOR.fill).attr('stroke', PI_COLOR.stroke)
            .attr('stroke-width', 1.5).attr('filter', 'url(#shd)');
        this._ndG.append('text').attr('x', W / 2).attr('y', piY)
            .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
            .attr('font-size', '11px').attr('font-weight', '700').attr('fill', PI_COLOR.dark).text('START');

        // π arrows
        this._piA = [];
        for (let i = 0; i < N; i++) {
            const s = this._sp[i];
            const sx = W / 2, sy = piY + 10;
            const ex = s.x, ey = s.y - R - 3;
            const bend = (ex - sx) * 0.06;
            const mx = (sx + ex) / 2 + bend, my = (sy + ey) / 2;
            const d = `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;
            const path = this._piG.append('path').attr('d', d)
                .attr('fill', 'none').attr('stroke', PI_COLOR.stroke)
                .attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3')
                .attr('marker-end', 'url(#ah-pi)').attr('opacity', 0.5);
            const lx = sx * 0.25 + mx * 0.5 + ex * 0.25, ly = sy * 0.25 + my * 0.5 + ey * 0.25 - 4;
            const label = this._piG.append('text').attr('x', lx).attr('y', ly)
                .attr('text-anchor', 'middle').attr('font-size', '9px')
                .attr('font-family', "'JetBrains Mono',monospace").attr('font-weight', '600')
                .attr('fill', PI_COLOR.dark).text(`π=${pi[i].toFixed(2)}`);
            this._piA.push({ path, label });
        }

        /* ══════ OBSERVATION NODES ══════ */
        for (let k = 0; k < M; k++) {
            const o = this._op[k];
            this._ndG.append('rect').attr('x', o.x - obsW / 2).attr('y', o.y - obsH / 2)
                .attr('width', obsW).attr('height', obsH).attr('rx', 5)
                .attr('fill', OBS_COLOR.fill).attr('stroke', OBS_COLOR.stroke)
                .attr('stroke-width', 1.2).attr('filter', 'url(#shd)');
            this._ndG.append('text').attr('x', o.x).attr('y', o.y)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                .attr('font-size', '14px').attr('font-weight', '700').attr('fill', OBS_COLOR.dark)
                .text(`O${k}`);
        }

        /* ══════ EMISSION ARROWS (B matrix) ══════ */
        this._emR = {};
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < M; k++) {
                const sp = this._sp[i], op = this._op[k];
                const col = STATE_COLORS[i % STATE_COLORS.length];
                // Fan start points so lines from same state don't stack
                const fan = M > 1 ? (k - (M - 1) / 2) * 12 : 0;
                const sx = sp.x + fan, sy = sp.y + R + 1;
                const ex = op.x, ey = op.y - obsH / 2 - 3;
                const mx = (sx + ex) / 2, my = (sy + ey) / 2;
                const d = `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;

                const path = this._emG.append('path').attr('d', d)
                    .attr('fill', 'none').attr('stroke', col.base)
                    .attr('stroke-width', 1).attr('opacity', 0.15)
                    .attr('stroke-dasharray', '3 2').attr('marker-end', 'url(#ah-em)');

                // Label at t=0.35 for state 0, t=0.55 for state 1, etc
                const lt = 0.30 + i * 0.20;
                const mt = 1 - lt;
                const lx = mt * mt * sx + 2 * mt * lt * mx + lt * lt * ex;
                const ly = mt * mt * sy + 2 * mt * lt * my + lt * lt * ey - 4;
                const label = this._emG.append('text').attr('x', lx).attr('y', ly)
                    .attr('text-anchor', 'middle').attr('font-size', '8px')
                    .attr('font-family', "'JetBrains Mono',monospace").attr('font-weight', '600')
                    .attr('fill', col.dark).attr('opacity', 0).text('');

                this._emR[`em-${i}-${k}`] = { path, label, sx, sy, mx, my, ex, ey };
            }
        }

        /* ══════ TRANSITION ARROWS (A matrix) ══════ */
        this._trR = {};
        this._slR = {};

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const key = `${i}-${j}`;
                const col = STATE_COLORS[i % STATE_COLORS.length];

                if (i === j) {
                    // Self-loop on outer side
                    const sp = this._sp[i];
                    let side = N <= 1 ? 'top' : i === 0 ? 'left' : i === N - 1 ? 'right' : 'top';
                    const L = 22; // loop size
                    let sx, sy, ex, ey, c1x, c1y, c2x, c2y, lx, ly;

                    if (side === 'left') {
                        sx = sp.x - R; sy = sp.y - 8; ex = sp.x - R; ey = sp.y + 8;
                        c1x = sp.x - R - L; c1y = sp.y - 16; c2x = sp.x - R - L; c2y = sp.y + 16;
                        lx = sp.x - R - L - 8; ly = sp.y;
                    } else if (side === 'right') {
                        sx = sp.x + R; sy = sp.y + 8; ex = sp.x + R; ey = sp.y - 8;
                        c1x = sp.x + R + L; c1y = sp.y + 16; c2x = sp.x + R + L; c2y = sp.y - 16;
                        lx = sp.x + R + L + 8; ly = sp.y;
                    } else {
                        sx = sp.x - 8; sy = sp.y - R; ex = sp.x + 8; ey = sp.y - R;
                        c1x = sp.x - 16; c1y = sp.y - R - L; c2x = sp.x + 16; c2y = sp.y - R - L;
                        lx = sp.x; ly = sp.y - R - L - 6;
                    }
                    const d = `M${sx},${sy} C${c1x},${c1y} ${c2x},${c2y} ${ex},${ey}`;
                    const path = this._edG.append('path').attr('d', d)
                        .attr('fill', 'none').attr('stroke', col.base)
                        .attr('stroke-width', 1.5).attr('opacity', 0.5)
                        .attr('marker-end', `url(#ah${i})`);
                    const label = this._edG.append('text').attr('x', lx).attr('y', ly)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-size', '9px').attr('font-weight', '600')
                        .attr('font-family', "'JetBrains Mono',monospace")
                        .attr('fill', col.dark).text('0.00');
                    this._slR[key] = { path, label };

                } else {
                    // Inter-state arc
                    const src = this._sp[i], tgt = this._sp[j];
                    const dx = tgt.x - src.x, dy = tgt.y - src.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    const ux = dx / dist, uy = dy / dist, px = -uy, py = ux;

                    // Endpoint at circle boundary
                    const sx = src.x + ux * (R + 2), sy = src.y + uy * (R + 2);
                    const ex = tgt.x - ux * (R + 5), ey = tgt.y - uy * (R + 5);

                    // CUBIC BEZIER for massive separation
                    // Curve LEFT (Outward) always.
                    // This naturally separates A->B (Out) and B->A (In/Opposite) relative to chord.
                    const sign = -1;
                    const off = Math.max(65, dist * 0.38) * sign; // Slightly less ballooned

                    // Control points closer to ends (1/5 and 4/5) for "squarer/less rounded" look
                    const c1x = src.x + ux * (dist / 5) + px * off;
                    const c1y = src.y + uy * (dist / 5) + py * off;
                    const c2x = src.x + ux * (4 * dist / 5) + px * off;
                    const c2y = src.y + uy * (4 * dist / 5) + py * off;

                    const d = `M${sx},${sy} C${c1x},${c1y} ${c2x},${c2y} ${ex},${ey}`;

                    const path = this._edG.append('path').attr('d', d)
                        .attr('fill', 'none').attr('stroke', col.base)
                        .attr('stroke-width', 1.8).attr('opacity', 0.35)
                        .attr('marker-end', `url(#ah${i})`);

                    // Label at t=0.5 of cubic bezier
                    // B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
                    const t = 0.5, mt = 0.5;
                    const mt2 = mt * mt, mt3 = mt2 * mt, t2 = t * t, t3 = t2 * t;
                    const lx = mt3 * sx + 3 * mt2 * t * c1x + 3 * mt * t2 * c2x + t3 * ex;
                    const ly = mt3 * sy + 3 * mt2 * t * c1y + 3 * mt * t2 * c2y + t3 * ey;

                    const bg = this._edG.append('rect').attr('x', lx - 17).attr('y', ly - 9)
                        .attr('width', 34).attr('height', 18).attr('rx', 9)
                        .attr('fill', 'rgba(255,255,255,0.92)').attr('stroke', '#e2e8f0')
                        .attr('stroke-width', 0.5).attr('opacity', 0);
                    const label = this._edG.append('text').attr('x', lx).attr('y', ly)
                        .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                        .attr('font-size', '9px').attr('font-weight', '700')
                        .attr('font-family', "'JetBrains Mono',monospace")
                        .attr('fill', col.dark).attr('opacity', 0).text('0.00');

                    this._trR[key] = { path, label, bg, sx, sy, c1x, c1y, c2x, c2y, ex, ey }; // store simplified ref

                    path.on('mouseover', (ev) => this._showTip(ev, `A[${i}][${j}]`))
                        .on('mouseout', () => this._hideTip());
                }
            }
        }

        /* ══════ STATE CIRCLES (on top) ══════ */
        this._sn = [];
        for (let i = 0; i < N; i++) {
            const s = this._sp[i], c = STATE_COLORS[i % STATE_COLORS.length];
            const circle = this._ndG.append('circle').attr('cx', s.x).attr('cy', s.y).attr('r', R)
                .attr('fill', `url(#sg${i})`).attr('stroke', c.dark).attr('stroke-width', 2)
                .attr('filter', 'url(#shd)').attr('cursor', 'pointer');
            this._ndG.append('text').attr('x', s.x).attr('y', s.y)
                .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
                .attr('font-size', '18px').attr('font-weight', '700').attr('fill', '#fff')
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
            .style('font-family', 'JetBrains Mono,monospace').style('font-size', '10px')
            .style('pointer-events', 'none').style('z-index', '50');

        this._startParticleLoop();
    }

    /* ═══════ RENDER ═══════ */
    _render(idx) {
        if (idx < 0 || idx >= this.history.length || !this.built) return;
        const { A, B, pi } = this.history[idx];
        const N = this.N, M = this.M;

        // π
        for (let i = 0; i < N; i++) {
            const r = this._piA[i], v = pi[i];
            r.path.transition().duration(180).attr('stroke-width', 1.5 + v * 4).attr('opacity', Math.max(0.25, 0.2 + v * 0.7));
            r.label.text(`π=${v.toFixed(2)}`).attr('font-size', '11px').attr('font-weight', '800');
        }

        // Transitions — BOLD visibility
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
            const v = A[i][j], k = `${i}-${j}`;
            if (i === j) {
                const r = this._slR[k]; if (!r) continue;
                r.path.transition().duration(180)
                    .attr('stroke-width', 1 + v * 3)
                    .attr('opacity', Math.max(0.12, 0.10 + v * 0.8));
                r.label.text(v.toFixed(2));
            } else {
                const r = this._trR[k]; if (!r) continue;
                // Always show, just vary intensity
                const opacity = Math.max(0.2, 0.2 + v * 0.8);
                r.path.transition().duration(180)
                    .attr('stroke-width', 1.5 + v * 5.5)
                    .attr('opacity', opacity);

                // Only show label if v > 0.005
                const showLbl = v > 0.005;
                r.label.text(v.toFixed(2)).attr('opacity', showLbl ? 1 : 0)
                    .attr('font-size', '11px').attr('font-weight', '800');
                r.bg.attr('opacity', showLbl ? 1 : 0);
            }
        }

        // Emissions — visible
        for (let i = 0; i < N; i++) for (let k = 0; k < M; k++) {
            const v = B[i][k], r = this._emR[`em-${i}-${k}`]; if (!r) continue;
            // Always show faint
            const opacity = Math.max(0.15, 0.15 + v * 0.7);
            r.path.transition().duration(180)
                .attr('stroke-width', 1.5 + v * 4) // Much thicker
                .attr('opacity', opacity);

            const showLbl = v > 0.03;
            r.label.text(showLbl ? v.toFixed(2) : '').attr('opacity', showLbl ? 1 : 0)
                .attr('font-size', '11px').attr('font-weight', '800'); // Larger emission labels
        }

        this._rebuildParticles(A);
        if (this.inspectorEl?.classList.contains('visible')) this._renderInspector(idx);
    }

    /* ═══════ PARTICLES ═══════ */
    _rebuildParticles(A) {
        this._clearParticles(); if (!this.particlesOn) return;
        for (let i = 0; i < this.N; i++) for (let j = 0; j < this.N; j++) {
            if (i === j) continue; const v = A[i][j]; if (v < 0.02) continue;
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
                // Cubic Bezier: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
                const r = p.ref;
                const x = m3 * r.sx + 3 * m2 * t * r.c1x + 3 * m * t2 * r.c2x + t3 * r.ex;
                const y = m3 * r.sy + 3 * m2 * t * r.c1y + 3 * m * t2 * r.c2y + t3 * r.ey;
                p.el.attr('cx', x).attr('cy', y);
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
        if (this.currentIdx >= 0) { const m = txt.match(/A\[(\d+)\]\[(\d+)\]/); if (m) f = `${txt} = ${this.history[this.currentIdx].A[+m[1]][+m[2]].toFixed(6)}`; }
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
        c.speedSelect?.addEventListener('change', e => this.setSpeed(parseFloat(e.target.value)));
        c.timeline?.addEventListener('input', e => { this.pause(); this.seekTo(parseInt(e.target.value, 10)); });
        c.btnParticles?.addEventListener('click', () => {
            this.particlesOn = !this.particlesOn;
            if (!this.particlesOn) this._clearParticles();
            else if (this.currentIdx >= 0) this._rebuildParticles(this.history[this.currentIdx].A);
            c.btnParticles.classList.toggle('active', this.particlesOn);
        });
    }
    _updateControls() {
        const c = this._ctrl; if (!c) return;
        const l = this.history.length, i = this.currentIdx;
        if (c.btnFirst) c.btnFirst.disabled = i <= 0;
        if (c.btnBack) c.btnBack.disabled = i <= 0;
        if (c.btnForward) c.btnForward.disabled = i >= l - 1;
        if (c.btnLast) c.btnLast.disabled = i >= l - 1;
        if (c.btnPlay) { c.btnPlay.innerHTML = this.isPlaying ? '⏸' : '▶'; c.btnPlay.disabled = l === 0; }
        if (c.timeline) {
            c.timeline.max = Math.max(0, l - 1); c.timeline.value = Math.max(0, i);
            c.timeline.style.setProperty('--progress', l > 1 ? (i / (l - 1)) * 100 + '%' : '0%');
        }
        if (c.iterLabel) c.iterLabel.textContent = l > 0 ? `Step ${i + 1} / ${l}` : 'No data';
    }
}
window.HMMDiagram = HMMDiagram;
