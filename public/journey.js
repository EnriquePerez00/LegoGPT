// LegoGPT User Journey Controller

let currentStep = 1;
const totalSteps = 5;

// Three.js global variables
let scene, camera, renderer, controls;
let container = document.getElementById("journey-viewer3d");
let partsList = []; // The active parts list of the current model
let colorCatalog = {}; // Maps color IDs to hex values
let currentSequenceIndex = 0; // Current step in playback
let playbackInterval = null; // Auto-play timer
let uploadedImageFile = null; // Stored file reference
let activeLdrContent = ""; // Saved LDR code for export
let loadedModels = []; // Available checkpoints

// Initial Setup on DOM load
window.addEventListener("DOMContentLoaded", () => {
    initThreeJS();
    loadColorCatalog();
    loadAvailableModels();
    setupEventListeners();
    updateStepUI();
    
    // Connect websocket for live training logs if any
    connectWebSocket();
});

// Initialize Three.js Scene
function initThreeJS() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);
    
    // Camera
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 5000);
    camera.position.set(100, -150, 200); // LDraw Z and Y coordinate adjustments
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight1.position.set(200, -400, 300);
    scene.add(dirLight1);
    
    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-200, 400, -300);
    scene.add(dirLight2);
    
    // Grid Helper
    const gridHelper = new THREE.GridHelper(400, 40, 0x334155, 0x1e293b);
    gridHelper.position.y = 0.5;
    scene.add(gridHelper);
    
    // Render loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
    
    window.addEventListener("resize", onWindowResize);
}

function onWindowResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Load LDraw Color Catalog
async function loadColorCatalog() {
    try {
        const res = await fetch("color_catalog.json");
        const data = await res.json();
        colorCatalog = data;
        logToServerConsole("Catálogo de colores cargado correctamente.");
    } catch (e) {
        logToServerConsole("Error cargando catálogo de colores, usando paleta fallback.");
    }
}

// Fetch available models from backend
async function loadAvailableModels() {
    try {
        const res = await fetch("/models");
        const data = await res.json();
        const select = document.getElementById("journey-model-select");
        select.innerHTML = '<option value="">-- Usar Modelo --</option>';
        data.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m.name;
            opt.textContent = `${m.name} (${m.theme})`;
            select.appendChild(opt);
        });
    } catch (e) {
        logToServerConsole("Error cargando lista de modelos.");
    }
}

// Event Listeners
function setupEventListeners() {
    // Next/Prev buttons
    document.getElementById("btn-wizard-next").addEventListener("click", () => {
        if (currentStep < totalSteps) {
            currentStep++;
            updateStepUI();
        }
    });
    
    document.getElementById("btn-wizard-prev").addEventListener("click", () => {
        if (currentStep > 1) {
            currentStep--;
            updateStepUI();
        }
    });
    
    // Generation buttons
    document.getElementById("btn-generate-mosaic").addEventListener("click", generateMosaic);
    document.getElementById("btn-generate-3d").addEventListener("click", generate3D);
    document.getElementById("btn-download-ldr").addEventListener("click", downloadLdrFile);
    
    // Playback buttons
    document.getElementById("btn-play-prev").addEventListener("click", () => {
        stopPlayback();
        showSequenceStep(currentSequenceIndex - 1);
    });
    
    document.getElementById("btn-play-next").addEventListener("click", () => {
        stopPlayback();
        showSequenceStep(currentSequenceIndex + 1);
    });
    
    document.getElementById("btn-play-toggle").addEventListener("click", togglePlayback);
}

// Update UI on step transition
function updateStepUI() {
    // Progress circles
    for (let i = 1; i <= totalSteps; i++) {
        const ind = document.getElementById(`step-ind-${i}`);
        ind.className = "step-indicator";
        if (i === currentStep) {
            ind.classList.add("active");
        } else if (i < currentStep) {
            ind.classList.add("completed");
        }
    }
    
    // Sidebar panels
    for (let i = 1; i <= totalSteps; i++) {
        const sec = document.getElementById(`sec-step-${i}`);
        sec.classList.remove("active");
        if (i === currentStep) {
            sec.classList.add("active");
        }
    }
    
    // Enable/Disable wizard nav buttons
    document.getElementById("btn-wizard-prev").disabled = (currentStep === 1);
    document.getElementById("btn-wizard-next").disabled = (currentStep === totalSteps);
    
    // Show/hide split panel overlays depending on the step
    const origBox = document.getElementById("original-preview-img");
    const origPlaceholder = document.getElementById("original-preview-placeholder");
    const eeveeBox = document.getElementById("eevee-preview-img");
    const eeveePlaceholder = document.getElementById("eevee-preview-placeholder");
    
    if (currentStep === 1 && uploadedImageFile) {
        origBox.style.display = "block";
        origPlaceholder.style.display = "none";
    } else {
        // keep it or hide it, let's keep it visible when there's an image
    }
    
    if (currentStep === 3) {
        runRealGraphValidation();
        renderModelParts(partsList);
    } else if (currentStep === 4) {
        runRealAssemblySequencing();
    } else if (currentStep === 5) {
        runRealEeveeRendering();
        renderModelParts(partsList);
    } else {
        renderModelParts(partsList);
    }
}

// Jump directly to step
function jumpToStep(step) {
    currentStep = step;
    updateStepUI();
}

// Live logs logging
function logToServerConsole(text) {
    const consoleBox = document.getElementById("journey-terminal-output");
    const div = document.createElement("div");
    div.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    consoleBox.appendChild(div);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

// Websocket for live logs
function connectWebSocket() {
    const loc = window.location;
    const wsProto = loc.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${loc.host}/ws`;
    const socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        document.getElementById("ws-status-dot").className = "dot connected";
        document.getElementById("ws-status-text").textContent = "Conectado";
        logToServerConsole("WebSocket conectado con el servidor LegoGPT.");
    };
    
    socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "log") {
            logToServerConsole(msg.message);
        }
    };
    
    socket.onclose = () => {
        document.getElementById("ws-status-dot").className = "dot disconnected";
        document.getElementById("ws-status-text").textContent = "Desconectado";
        logToServerConsole("WebSocket desconectado. Reintentando en 5s...");
        setTimeout(connectWebSocket, 5000);
    };
}

// Handle Image Selection
function handleImageSelection(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    uploadedImageFile = file;
    document.getElementById("image-name-display").textContent = file.name;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById("original-preview-img");
        img.src = e.target.result;
        img.style.display = "block";
        document.getElementById("original-preview-placeholder").style.display = "none";
    };
    reader.readAsDataURL(file);
    logToServerConsole(`Cargada imagen: ${file.name}`);
}

// FASE 1: Generate 2D Mosaic
async function generateMosaic() {
    if (!uploadedImageFile) {
        alert("Por favor, selecciona una imagen primero.");
        return;
    }
    
    const size = document.getElementById("mosaic-size").value;
    const dither = document.getElementById("mosaic-dither").checked;
    
    logToServerConsole(`Generando mosaico 2D de ${size}x${size}...`);
    
    const formData = new FormData();
    formData.append("file", uploadedImageFile);
    
    try {
        const res = await fetch(`/generate-mosaic?size=${size}&use_dithering=${dither}`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        
        if (data.status === "success") {
            partsList = data.parts;
            document.getElementById("journey-piece-count").textContent = `${partsList.length} piezas`;
            logToServerConsole(`Mosaico generado con éxito. ${partsList.length} placas instanciadas.`);
            
            // Build LDraw string for mosaic
            buildLdrContent("Mosaico 2D");
            
            // Render parts in Three.js
            renderModelParts(partsList);
            
            // Fill step metrics with mock 2D stats (mosaics are always stable)
            fillGraphMetrics({
                wheel_count: 0,
                wheels_at_bottom: true,
                symmetry_score: 1.0,
                cabin_height_valid: true,
                cabin_centered: true,
                has_cabin_parts: false,
                blocked_aesthetic_count: 0,
                is_stable: true
            });
            
            // Advance to Phase 3/4 to check it
            setTimeout(() => {
                currentStep = 3;
                updateStepUI();
            }, 1000);
        } else {
            logToServerConsole(`Error: ${data.message}`);
        }
    } catch (e) {
        logToServerConsole(`Error en la llamada al servidor: ${e}`);
    }
}

// FASE 2: Generate 3D Structure
async function generate3D() {
    const prompt = document.getElementById("journey-prompt").value;
    const pieces = document.getElementById("journey-pieces").value;
    const model = document.getElementById("journey-model-select").value;
    
    logToServerConsole(`Solicitando generación 3D para el prompt: "${prompt}"...`);
    
    try {
        let res, data;
        
        if (model) {
            // Using fast inference endpoint
            res = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_name: model,
                    num_pieces: parseInt(pieces)
                })
            });
            data = await res.json();
        } else {
            // Standard generate-prompt endpoint (CLI + Blender)
            res = await fetch("/generate-prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prompt: prompt,
                    num_pieces: parseInt(pieces)
                })
            });
            data = await res.json();
        }
        
        if (data.status === "success") {
            partsList = data.parts;
            document.getElementById("journey-piece-count").textContent = `${partsList.length} piezas`;
            logToServerConsole(`Modelo 3D generado. ${partsList.length} piezas instanciadas.`);
            
            // Set LDraw content
            buildLdrContent(prompt);
            
            // Render
            renderModelParts(partsList);
            
            // Set render preview if Blender ran
            if (data.render_url) {
                const renderImg = document.getElementById("eevee-preview-img");
                renderImg.src = data.render_url;
                renderImg.style.display = "block";
                document.getElementById("eevee-preview-placeholder").style.display = "none";
                logToServerConsole("Render de Blender Eevee cargado.");
            }
            
            // Perform metric logic for Phase 3 (LegoGraph)
            runRealGraphValidation();
            
            setTimeout(() => {
                currentStep = 3;
                updateStepUI();
            }, 1000);
        } else {
            logToServerConsole(`Error: ${data.message}`);
        }
    } catch (e) {
        logToServerConsole(`Error: ${e}`);
    }
}

// FASE 3: LegoGraph check (real backend validation)
async function runRealGraphValidation() {
    if (!partsList || partsList.length === 0) return;
    logToServerConsole("Solicitando validación topológica LegoGraph...");
    
    try {
        const res = await fetch("/validate-graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ parts: partsList })
        });
        const data = await res.json();
        
        if (data.status === "success") {
            logToServerConsole(`LegoGraph completado. Estabilidad: ${data.is_stable ? 'Válida' : 'Fallo'}, Colisiones: ${data.collisions_count}`);
            
            const stableCard = document.getElementById("metric-stability");
            const connCard = document.getElementById("metric-connected");
            const baseCard = document.getElementById("metric-base-parts");
            const symCard = document.getElementById("metric-symmetry");
            
            stableCard.className = "metric-card " + (data.is_stable ? "success" : "danger");
            stableCard.querySelector(".val").textContent = data.is_stable ? "ESTABLE" : "INESTABLE";
            
            const is_connected = data.is_stable && (data.collisions_count === 0);
            connCard.className = "metric-card " + (is_connected ? "success" : "danger");
            connCard.querySelector(".val").textContent = is_connected ? "CONECTADO" : "CON ERRORES";
            
            const wc = data.topology.wheel_count;
            baseCard.className = "metric-card " + (wc <= 4 ? "success" : "warning");
            baseCard.querySelector(".val").textContent = `${wc} piezas`;
            
            const sym = Math.round(data.topology.symmetry_score * 100);
            symCard.className = "metric-card success";
            symCard.querySelector(".val").textContent = `${sym}%`;
            
            if (data.topology.blocked_aesthetic_count > 0) {
                logToServerConsole(`Advertencia: ${data.topology.blocked_aesthetic_count} azulejos estéticos bloqueados verticalmente.`);
            }
        }
    } catch (e) {
        logToServerConsole(`Error de validación: ${e}`);
    }
}

// FASE 4: Assembly sequencing (real bottom-up plan)
async function runRealAssemblySequencing() {
    if (!partsList || partsList.length === 0) return;
    logToServerConsole("Calculating physical bottom-up assembly sequence...");
    
    try {
        const res = await fetch("/sequence-assembly", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ parts: partsList })
        });
        const data = await res.json();
        
        if (data.status === "success") {
            partsList = data.parts;
            logToServerConsole("Physical build sequence loaded.");
            showSequenceStep(partsList.length);
        }
    } catch (e) {
        logToServerConsole(`Error sequencing build: ${e}`);
        showSequenceStep(partsList.length);
    }
}

// FASE 5: Blender Eevee Render
async function runRealEeveeRendering() {
    if (!partsList || partsList.length === 0) return;
    logToServerConsole("Iniciando renderizado fotorrealista con Blender Eevee...");
    
    const eeveePlaceholder = document.getElementById("eevee-preview-placeholder");
    const eeveeBox = document.getElementById("eevee-preview-img");
    
    eeveePlaceholder.textContent = "Renderizando escena...";
    eeveePlaceholder.style.display = "block";
    eeveeBox.style.display = "none";
    
    try {
        const res = await fetch("/render-eevee", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ parts: partsList, name: "journey_render" })
        });
        const data = await res.json();
        
        if (data.status === "success") {
            eeveeBox.src = `${data.render_url}?t=${Date.now()}`;
            eeveeBox.style.display = "block";
            eeveePlaceholder.style.display = "none";
            logToServerConsole("Render de Blender Eevee guardado y cargado en el panel lateral.");
        } else {
            eeveePlaceholder.textContent = data.message || "Fallo en el render.";
            logToServerConsole(`Advertencia de Render: ${data.message}`);
        }
    } catch (e) {
        logToServerConsole(`Error al renderizar: ${e}`);
        eeveePlaceholder.textContent = "Error en el renderizado.";
    }
}

function fillGraphMetrics(metrics) {
    const stableCard = document.getElementById("metric-stability");
    const connCard = document.getElementById("metric-connected");
    const baseCard = document.getElementById("metric-base-parts");
    const symCard = document.getElementById("metric-symmetry");
    
    // Stability
    stableCard.className = "metric-card " + (metrics.is_stable ? "success" : "danger");
    stableCard.querySelector(".val").textContent = metrics.is_stable ? "ESTABLE" : "INESTABLE";
    
    // Connected
    connCard.className = "metric-card success";
    connCard.querySelector(".val").textContent = "CONECTADO";
    
    // Base Parts
    const count = metrics.wheel_count || 4; // Mock/fallback
    baseCard.className = "metric-card " + (count <= 4 ? "success" : "warning");
    baseCard.querySelector(".val").textContent = `${count} piezas`;
    
    // Symmetry
    const sym = metrics.symmetry_score * 100;
    symCard.className = "metric-card success";
    symCard.querySelector(".val").textContent = `${sym}%`;
}

// Visualizer: Render model parts in Three.js Scene
function renderModelParts(parts) {
    // Clear old elements (keep lights and grids)
    const toRemove = [];
    scene.traverse(child => {
        if (child.isMesh && child.name !== "grid") {
            toRemove.push(child);
        }
    });
    toRemove.forEach(child => scene.remove(child));
    
    if (!parts || parts.length === 0) return;
    
    // Determine colors
    parts.forEach(p => {
        const colorId = String(p.color);
        let colorHex = "#ffffff";
        if (colorCatalog[colorId]) {
            colorHex = colorCatalog[colorId].hex || "#ffffff";
        }
        
        // Build mesh (BoxGeometry fallback for visualization)
        // Approximate dimensions based on standard 1x1 size (20 LDU = 1 ThreeJS unit approximately)
        let w = 1.0, h = 1.2, d = 1.0; // standard 1x1 brick in visualizer scale
        
        if (p.part_id === "3024.dat") {
            w = 1.0; h = 0.4; d = 1.0; // 1x1 plate
        } else if (p.part_id === "3022.dat") {
            w = 2.0; h = 0.4; d = 2.0; // 2x2 plate
        } else if (p.part_id === "3020.dat") {
            w = 2.0; h = 0.4; d = 4.0; // 2x4 plate
        } else if (p.part_id === "3003.dat") {
            w = 2.0; h = 1.2; d = 2.0; // 2x2 brick
        } else if (p.part_id === "3001.dat") {
            w = 2.0; h = 1.2; d = 4.0; // 2x4 brick
        } else if (p.part_id === "3010.dat") {
            w = 1.0; h = 1.2; d = 4.0; // 1x4 brick
        } else if (p.part_id.includes("wheel") || p.part_id.includes("tire") || p.part_id === "42610.dat" || p.part_id === "3139.dat") {
            w = 1.2; h = 1.2; d = 0.8; // wheel approximation
        }
        
        const geometry = new THREE.BoxGeometry(w * 10, h * 10, d * 10);
        const material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(colorHex),
            roughness: 0.15,
            metalness: 0.05
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        
        // Set transform (LDraw Y increases downwards, so we flip Y position)
        const m = p.transform;
        mesh.position.set(m[3] / 2.0, -m[7] / 2.0, m[11] / 2.0);
        
        const rotMatrix = new THREE.Matrix4().set(
             m[0], -m[1],  m[2], 0,
            -m[4],  m[5], -m[6], 0,
             m[8], -m[9],  m[10], 0,
             0,     0,     0,    1
        );
        mesh.rotation.setFromRotationMatrix(rotMatrix);
        scene.add(mesh);
    });
    
    // Adjust camera target to center of model
    if (parts.length > 0) {
        let avgX = 0, avgY = 0, avgZ = 0;
        parts.forEach(p => {
            avgX += p.transform[12];
            avgY += p.transform[13];
            avgZ += p.transform[14];
        });
        controls.target.set(avgX / (2.0 * parts.length), -avgY / (2.0 * parts.length), avgZ / (2.0 * parts.length));
    }
}

// FASE 4: Step-by-Step Playback
function showSequenceStep(step) {
    if (partsList.length === 0) return;
    
    // Clamp step index
    currentSequenceIndex = Math.max(0, Math.min(step, partsList.length));
    
    // Update label
    document.getElementById("play-step-info").textContent = `Paso: ${currentSequenceIndex} / ${partsList.length}`;
    
    // Render only the first N parts
    const subset = partsList.slice(0, currentSequenceIndex);
    renderModelParts(subset);
}

function togglePlayback() {
    const btn = document.getElementById("btn-play-toggle");
    if (playbackInterval) {
        stopPlayback();
    } else {
        btn.textContent = "Pausar";
        btn.style.backgroundColor = "#ef4444";
        
        if (currentSequenceIndex >= partsList.length) {
            currentSequenceIndex = 0;
        }
        
        playbackInterval = setInterval(() => {
            if (currentSequenceIndex < partsList.length) {
                showSequenceStep(currentSequenceIndex + 1);
            } else {
                stopPlayback();
            }
        }, 800);
    }
}

function stopPlayback() {
    const btn = document.getElementById("btn-play-toggle");
    btn.textContent = "Reproducir";
    btn.style.backgroundColor = "#10b981";
    if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
    }
}

// Build LDraw file string for export
function buildLdrContent(title) {
    let lines = ["0 LegoGPT Generated Build", `0 Prompt/Title: ${title}`, "0 STEP"];
    partsList.forEach(p => {
        // Flatten matrix rotation
        const r = p.transform;
        const rotStr = `${r[0].toFixed(5)} ${r[1].toFixed(5)} ${r[2].toFixed(5)} ${r[4].toFixed(5)} ${r[5].toFixed(5)} ${r[6].toFixed(5)} ${r[8].toFixed(5)} ${r[9].toFixed(5)} ${r[10].toFixed(5)}`;
        lines.push(`1 ${p.color} ${p.transform[12].toFixed(3)} ${p.transform[13].toFixed(3)} ${p.transform[14].toFixed(3)} ${rotStr} ${p.part_id}`);
    });
    activeLdrContent = lines.join("\n") + "\n";
}

// Download LDraw file
function downloadLdrFile() {
    if (!activeLdrContent) {
        alert("No hay ningún modelo generado para exportar.");
        return;
    }
    const blob = new Blob([activeLdrContent], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "legogpt_model.ldr";
    a.click();
    logToServerConsole("Archivo LDraw exportado para descarga.");
}
