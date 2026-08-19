/* ═══════════════════════════════════════════════════════════
   MNIST Digit Classifier — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

(() => {
    "use strict";

    // ── DOM Elements ────────────────────────────────────────
    const canvas = document.getElementById("draw-canvas");
    const ctx = canvas.getContext("2d");
    const wrapper = document.getElementById("canvas-wrapper");
    const hint = document.getElementById("canvas-hint");
    const btnClear = document.getElementById("btn-clear");
    const btnPredict = document.getElementById("btn-predict");
    const brushInput = document.getElementById("brush-size");
    const brushValue = document.getElementById("brush-value");
    const probGrid = document.getElementById("prob-grid");
    const predDigit = document.getElementById("prediction-digit");
    const confValue = document.getElementById("confidence-value");
    const confBar = document.getElementById("confidence-bar");
    const predHero = document.getElementById("prediction-hero");
    const statusBadge = document.getElementById("status-badge");
    const previewCanvas = document.getElementById("preview-canvas");
    const previewCtx = previewCanvas.getContext("2d");
    const featureMapGrid = document.getElementById("feature-map-grid");
    const filterGrid = document.getElementById("filter-grid");

    // ── State ───────────────────────────────────────────────
    let drawing = false;
    let hasDrawn = false;
    let brushSize = 18;
    let lastX = 0, lastY = 0;

    // ── Initialize ──────────────────────────────────────────
    function init() {
        clearCanvas();
        buildProbGrid();
        setupCanvasEvents();
        setupButtons();
        setupBrush();
    }

    // ── Canvas Drawing ──────────────────────────────────────
    function clearCanvas() {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        hint.classList.remove("hidden");
        wrapper.classList.remove("drawing");

        // Reset results
        predDigit.textContent = "?";
        confValue.textContent = "—";
        confBar.style.width = "0%";
        predHero.classList.remove("active");
        statusBadge.textContent = "Waiting";
        statusBadge.classList.remove("active");
        resetProbBars();
        brushSize = 18;
        brushInput.value = 18;
        brushValue.textContent = "18px";

        filterGrid.innerHTML = "";

        document.querySelectorAll(".filter-item.selected, .feature-map.selected")
            .forEach(element => element.classList.remove("selected"));

        featureMapGrid.innerHTML = "";

        previewCtx.fillStyle = "#000";
        previewCtx.fillRect(0, 0, 28, 28);
    }

    function getCanvasPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let clientX, clientY;
        if (e.touches) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    }

    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        hasDrawn = true;
        hint.classList.add("hidden");
        wrapper.classList.add("drawing");

        const pos = getCanvasPos(e);
        lastX = pos.x;
        lastY = pos.y;

        // Draw a dot at the start point
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fillStyle = "#fff";
        ctx.fill();
    }

    function draw(e) {
        if (!drawing) return;
        e.preventDefault();

        const pos = getCanvasPos(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = brushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();

        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDraw(e) {
        if (e) e.preventDefault();
        drawing = false;
    }

    function setupCanvasEvents() {
        // Mouse
        canvas.addEventListener("mousedown", startDraw);
        canvas.addEventListener("mousemove", draw);
        canvas.addEventListener("mouseup", stopDraw);
        canvas.addEventListener("mouseleave", stopDraw);
        // Touch
        canvas.addEventListener("touchstart", startDraw, { passive: false });
        canvas.addEventListener("touchmove", draw, { passive: false });
        canvas.addEventListener("touchend", stopDraw);
        canvas.addEventListener("touchcancel", stopDraw);
    }

    // ── Brush ───────────────────────────────────────────────
    function setupBrush() {
        brushInput.addEventListener("input", () => {
            brushSize = parseInt(brushInput.value);
            brushValue.textContent = brushSize + "px";
        });
    }

    // ── Buttons ─────────────────────────────────────────────
    function setupButtons() {
        btnClear.addEventListener("click", clearCanvas);
        btnPredict.addEventListener("click", predict);

        // Keyboard shortcut
        document.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                if (document.activeElement === document.body) {
                    e.preventDefault();
                    predict();
                }
            }
            if (e.key === "Escape" || e.key === "Delete") {
                clearCanvas();
            }
        });
    }

    // ── Probability Grid ────────────────────────────────────
    function buildProbGrid() {
        probGrid.innerHTML = "";
        for (let i = 0; i < 10; i++) {
            const row = document.createElement("div");
            row.className = "prob-row";
            row.id = `prob-row-${i}`;
            row.innerHTML = `
                <span class="prob-label">${i}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="prob-bar-${i}"></div>
                </div>
                <span class="prob-percent" id="prob-pct-${i}">0%</span>
            `;
            probGrid.appendChild(row);
        }
    }

    function resetProbBars() {
        for (let i = 0; i < 10; i++) {
            const bar = document.getElementById(`prob-bar-${i}`);
            const pct = document.getElementById(`prob-pct-${i}`);
            const row = document.getElementById(`prob-row-${i}`);
            if (bar) bar.style.width = "0%";
            if (pct) pct.textContent = "0%";
            if (row) row.classList.remove("highlight");
        }
    }

    function updateProbBars(probabilities, predicted) {
        for (let i = 0; i < 10; i++) {
            const bar = document.getElementById(`prob-bar-${i}`);
            const pct = document.getElementById(`prob-pct-${i}`);
            const row = document.getElementById(`prob-row-${i}`);
            const val = probabilities[i] * 100;

            // Stagger animation
            setTimeout(() => {
                bar.style.width = val + "%";
                pct.textContent = val.toFixed(1) + "%";
                row.classList.toggle("highlight", i === predicted);
            }, i * 40);
        }
    }
    function displayFeatureMaps(featureMaps) {
        featureMapGrid.innerHTML = "";

        featureMaps.forEach((map, index) => {
            const wrapper = document.createElement("div");
            wrapper.className = "feature-map";
            wrapper.style.cursor = "pointer";

wrapper.onclick = function () {
    const filters = document.querySelectorAll(".filter-item");

    filters.forEach(filter => {
        filter.classList.remove("selected");
    });

    const maps = document.querySelectorAll(".feature-map");

    maps.forEach(map => {
        map.classList.remove("selected");
    });

    wrapper.classList.add("selected");

    const selectedFilter = filters[index];

    if (selectedFilter) {
        selectedFilter.classList.add("selected");

        selectedFilter.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }
};
            const canvas = document.createElement("canvas");
            canvas.width = 26;
            canvas.height = 26;

            const ctx = canvas.getContext("2d");
            const imageData = ctx.createImageData(26, 26);

            for (let y = 0; y < 26; y++) {
                for (let x = 0; x < 26; x++) {
                    const value = map[y][x];
                    const pixel = (y * 26 + x) * 4;

                    imageData.data[pixel] = value;
                    imageData.data[pixel + 1] = value;
                    imageData.data[pixel + 2] = value;
                    imageData.data[pixel + 3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);

            const label = document.createElement("span");
            label.textContent = `Map ${index + 1}`;

            wrapper.appendChild(canvas);
            wrapper.appendChild(label);
            featureMapGrid.appendChild(wrapper);
        });
    }
    function displayFilters(filters) {
        filterGrid.innerHTML = "";

        filters.forEach((filter, index) => {
            const wrapper = document.createElement("div");
            wrapper.className = "filter-item";

            wrapper.onclick = function () {
                document.querySelectorAll(".filter-item").forEach(filter => {
                    filter.classList.remove("selected");
                });

                wrapper.classList.add("selected");

                const maps = document.querySelectorAll(".feature-map");

                maps.forEach(map => {
                    map.classList.remove("selected");
                });

                const selectedMap = maps[index];

                if (selectedMap) {
                    selectedMap.classList.add("selected");

                    selectedMap.scrollIntoView({
                        behavior: "smooth",
                        block: "center"
                    });
                }
            };

            const canvas = document.createElement("canvas");
            canvas.width = 3;
            canvas.height = 3;

            const ctx = canvas.getContext("2d");
            const imageData = ctx.createImageData(3, 3);

            for (let y = 0; y < 3; y++) {
                for (let x = 0; x < 3; x++) {
                    const value = filter[y][x];
                    const pixel = (y * 3 + x) * 4;

                    imageData.data[pixel] = value;
                    imageData.data[pixel + 1] = value;
                    imageData.data[pixel + 2] = value;
                    imageData.data[pixel + 3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);

            const label = document.createElement("span");
            label.textContent = `Filter ${index + 1}`;

            wrapper.appendChild(canvas);
            wrapper.appendChild(label);
            filterGrid.appendChild(wrapper);
        });
    }
    // ── Prediction ──────────────────────────────────────────
    async function predict() {
        if (btnPredict.classList.contains("loading")) return;
        if (!hasDrawn) {
            shakeElement(wrapper);
            return;
        }

        btnPredict.classList.add("loading");
        statusBadge.textContent = "Analyzing...";
        statusBadge.classList.remove("active");

        // Get canvas as base64 PNG
        const imageData = canvas.toDataURL("image/png");

        // Update preview canvas
        updatePreview();

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageData }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();

            if (data.prediction === -1) {
                predDigit.textContent = "?";
                confValue.textContent = "Empty";
                confBar.style.width = "0%";
                statusBadge.textContent = "Empty";
                shakeElement(wrapper);
                return;
            }

            // Update hero
            predDigit.textContent = data.prediction;
            confValue.textContent = (data.confidence * 100).toFixed(1) + "%";
            confBar.style.width = (data.confidence * 100) + "%";
            predHero.classList.add("active");

            // Update status
            statusBadge.textContent = "Done";
            statusBadge.classList.add("active");

            // Update bars
            updateProbBars(data.probabilities, data.prediction);
            if (data.conv1_filters) {
                displayFilters(data.conv1_filters);
            }

            if (data.conv1_maps) {
                displayFeatureMaps(data.conv1_maps);
            }

        } catch (err) {
            console.error("Prediction error:", err);
            predDigit.textContent = "!";
            confValue.textContent = "Error";
            statusBadge.textContent = "Error";
            statusBadge.classList.remove("active");
            predHero.classList.remove("active");

        } finally {
            btnPredict.classList.remove("loading");
        }
    }

    // ── Preview (28×28 downscale) ───────────────────────────
    function updatePreview() {
        previewCtx.clearRect(0, 0, 28, 28);
        previewCtx.drawImage(canvas, 0, 0, 28, 28);
    }

    // ── Shake Animation ─────────────────────────────────────
    function shakeElement(el) {
        el.style.animation = "none";
        el.offsetHeight; // reflow
        el.style.animation = "shake 0.4s ease";
        setTimeout(() => { el.style.animation = "none"; }, 400);
    }

    // Add shake keyframes dynamically
    const style = document.createElement("style");
    style.textContent = `
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            20% { transform: translateX(-6px); }
            40% { transform: translateX(6px); }
            60% { transform: translateX(-4px); }
            80% { transform: translateX(4px); }
        }
    `;
    document.head.appendChild(style);

    // ── Boot ────────────────────────────────────────────────
    init();
})();
