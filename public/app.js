const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const terminalOutput = document.getElementById('terminal-output');
const wsStatusDot = document.getElementById('ws-status-dot');
const wsStatusText = document.getElementById('ws-status-text');
const inputModelName = document.getElementById('input-model-name');
const inputTheme = document.getElementById('input-theme');
const inputMaxPieces = document.getElementById('input-max-pieces');
const inputEpochs = document.getElementById('input-epochs');
const inputPatience = document.getElementById('input-patience');
const btnGenerate = document.getElementById('btn-generate');
const inferenceModelSelect = document.getElementById('inference-model-select');

// Initialize Chart
const ctx = document.getElementById('lossChart').getContext('2d');
const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Training Loss',
            data: [],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                ticks: { color: '#94a3b8' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#94a3b8' }
            }
        },
        plugins: {
            legend: {
                labels: { color: '#f8fafc' }
            }
        }
    }
});

let ws;

function connectWebSocket() {
    ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
        wsStatusDot.className = 'dot connected';
        wsStatusText.textContent = 'Conectado';
        logToTerminal('Sistema conectado al backend.');
    };

    ws.onclose = () => {
        wsStatusDot.className = 'dot disconnected';
        wsStatusText.textContent = 'Desconectado';
        logToTerminal('Conexión perdida. Reintentando en 3s...', 'warning');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
            // Update chart
            lossChart.data.labels.push(`Ep ${data.epoch}`);
            lossChart.data.datasets[0].data.push(data.loss);
            
            // Keep only last 50 points to avoid clutter
            if (lossChart.data.labels.length > 50) {
                lossChart.data.labels.shift();
                lossChart.data.datasets[0].data.shift();
            }
            lossChart.update();
            
            logToTerminal(`Época ${data.epoch} | Loss: ${data.loss.toFixed(4)}`);
        } else if (data.type === 'info') {
            logToTerminal(data.message);
            if (data.message.includes('Training finished') || data.message.includes('detenido')) {
                btnStart.disabled = false;
                btnStop.disabled = true;
                loadModels();
            }
        }
    };
}

function logToTerminal(message, type = 'info') {
    const line = document.createElement('div');
    line.className = 'log-line';
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${message}`;
    if (type === 'warning') line.style.color = '#fbbf24';
    
    terminalOutput.appendChild(line);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

btnStart.addEventListener('click', async () => {
    try {
        const config = {
            model_name: inputModelName.value.trim() || 'modelo_lego_1',
            theme: inputTheme.value,
            max_pieces: parseInt(inputMaxPieces.value, 10) || 100,
            max_epochs: parseInt(inputEpochs.value, 10),
            early_stopping_patience: parseInt(inputPatience.value, 10)
        };

        const res = await fetch('http://localhost:8000/train/start', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.status === 'started') {
            btnStart.disabled = true;
            btnStop.disabled = false;
            // Clear chart
            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.update();
            logToTerminal('Entrenamiento iniciado...');
        } else {
            logToTerminal(`Estado: ${data.status}`);
        }
    } catch (error) {
        logToTerminal(`Error: ${error.message}`, 'warning');
    }
});

btnStop.addEventListener('click', async () => {
    try {
        const res = await fetch('http://localhost:8000/train/stop', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'stopped') {
            btnStart.disabled = false;
            btnStop.disabled = true;
            logToTerminal('Entrenamiento detenido por el usuario.');
            loadModels();
        }
    } catch (error) {
        logToTerminal(`Error: ${error.message}`, 'warning');
    }
});

// Init
connectWebSocket();

// ==========================================
// Three.js 3D Viewer Setup
// ==========================================
const viewerContainer = document.getElementById('viewer3d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020617);
const camera = new THREE.PerspectiveCamera(75, viewerContainer.clientWidth / viewerContainer.clientHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewerContainer.clientWidth, viewerContainer.clientHeight);
viewerContainer.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
camera.position.set(100, 100, 100);
controls.target.set(0, 0, 0);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(100, 200, 50);
scene.add(dirLight);

const gridHelper = new THREE.GridHelper(400, 20, 0x444444, 0x222222);
scene.add(gridHelper);

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    if(viewerContainer) {
        camera.aspect = viewerContainer.clientWidth / viewerContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(viewerContainer.clientWidth, viewerContainer.clientHeight);
    }
});

let LEGO_COLORS = {
    1: 0x0055BF, 4: 0xC91A09, 14: 0xF2CD37, 0: 0x05131D, 15: 0xFFFFFF
};

// Cargar catálogo de colores oficial de LegoVision
async function loadColorCatalog() {
    try {
        const res = await fetch('color_catalog.json');
        const catalog = await res.json();
        for (const [code, info] of Object.entries(catalog)) {
            if (info.hex) {
                // Convert #HEX to 0xHEX number for Three.js
                const hexNum = parseInt(info.hex.replace('#', '0x'), 16);
                LEGO_COLORS[code] = hexNum;
            }
        }
        console.log("Catálogo cromático oficial cargado.");
    } catch (e) {
        console.warn("Fallo al cargar catálogo de colores, usando defaults.", e);
    }
}
loadColorCatalog();

function getPartDimensions(partId) {
    const PART_DIMENSIONS = {
        "3024.dat": { width: 20.0, height: 8.0, depth: 20.0 },    // Plate 1x1
        "3023.dat": { width: 20.0, height: 8.0, depth: 40.0 },    // Plate 1x2
        "3022.dat": { width: 40.0, height: 8.0, depth: 40.0 },    // Plate 2x2
        "3020.dat": { width: 40.0, height: 8.0, depth: 80.0 },    // Plate 2x4
        "3710.dat": { width: 20.0, height: 8.0, depth: 80.0 },    // Plate 1x4
        "3005.dat": { width: 20.0, height: 24.0, depth: 20.0 },   // Brick 1x1
        "3004.dat": { width: 20.0, height: 24.0, depth: 40.0 },   // Brick 1x2
        "3003.dat": { width: 40.0, height: 24.0, depth: 40.0 },   // Brick 2x2
        "3001.dat": { width: 40.0, height: 24.0, depth: 80.0 },   // Brick 2x4
        "3010.dat": { width: 20.0, height: 24.0, depth: 80.0 },   // Brick 1x4
    };
    const key = partId.toLowerCase().replace(/\\/g, '/');
    const basename = key.split('/').pop();
    const dim = PART_DIMENSIONS[basename];
    if (dim) return dim;
    if (basename.includes("plate") || basename.startsWith("302")) {
        return { width: 20.0, height: 8.0, depth: 20.0 };
    }
    return { width: 20.0, height: 24.0, depth: 20.0 };
}


btnGenerate.addEventListener('click', async () => {
    const modelName = inferenceModelSelect.value;
    if (!modelName) {
        logToTerminal('Por favor, selecciona un modelo de la lista para generar la estructura.', 'warning');
        return;
    }

    btnGenerate.disabled = true;
    logToTerminal(`Generando estructura 3D mediante Inferencia con modelo '${modelName}'...`);
    viewerTitle.textContent = `Inferencia: ${modelName}`;
    viewerStepControls.style.display = 'none'; // Ocultar controles en inferencia
    stopPlayback();
    
    try {
        const res = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            logToTerminal(`Se generaron ${data.parts.length} piezas exitosamente.`);
            
            // Clean scene
            scene.children = scene.children.filter(c => c.type === 'AmbientLight' || c.type === 'DirectionalLight' || c.type === 'GridHelper');
            
            // Update piece count badge
            const badge = document.getElementById('viewer-piece-count');
            if (badge) {
                badge.textContent = `${data.parts.length} piezas`;
                badge.style.display = 'inline-block';
            }
            
            data.parts.forEach(part => {
                const { width, height, depth } = getPartDimensions(part.part_id);
                
                const geometry = new THREE.BoxGeometry(width, height, depth);
                geometry.translate(0, height / 2, 0); // Align origin to bottom face (LDraw style)
                const colorHex = LEGO_COLORS[part.color] || 0x888888;
                const material = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.2 });
                const cube = new THREE.Mesh(geometry, material);
                
                const m = part.transform;
                cube.position.set(m[3], -m[7], m[11]); // Invert Y
                
                // Apply Y-inversion mirrored rotation matrix
                const rotMatrix = new THREE.Matrix4().set(
                     m[0], -m[1],  m[2], 0,
                    -m[4],  m[5], -m[6], 0,
                     m[8], -m[9],  m[10], 0,
                     0,     0,     0,    1
                );
                cube.rotation.setFromRotationMatrix(rotMatrix);
                
                const edges = new THREE.EdgesGeometry(geometry);
                const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000 }));
                cube.add(line);
                
                scene.add(cube);
            });
            
            // Auto-center camera on generated model
            if (data.parts.length > 0) {
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                let minZ = Infinity, maxZ = -Infinity;
                
                data.parts.forEach(p => {
                    const m = p.transform;
                    const x = m[3];
                    const y = -m[7];
                    const z = m[11];
                    if (x < minX) minX = x; if (x > maxX) maxX = x;
                    if (y < minY) minY = y; if (y > maxY) maxY = y;
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
                });
                
                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const centerZ = (minZ + maxZ) / 2;
                
                controls.target.set(centerX, centerY, centerZ);
                
                const sizeX = maxX - minX;
                const sizeY = maxY - minY;
                const sizeZ = maxZ - minZ;
                const maxDim = Math.max(sizeX, sizeY, sizeZ, 100);
                
                camera.position.set(centerX + maxDim * 1.5, centerY + maxDim * 1.5, centerZ + maxDim * 1.5);
            }
        } else {
            logToTerminal(`Error en inferencia: ${data.message}`, 'warning');
        }
    } catch (e) {
        logToTerminal(`Fallo de conexión: ${e.message}`, 'warning');
    }
    btnGenerate.disabled = false;
});

// ==========================================
// OMR File Upload & Step-by-Step Viewer Controls
// ==========================================
const fileUpload = document.getElementById('file-upload');
const btnUpload = document.getElementById('btn-upload');
const viewerTitle = document.getElementById('viewer-title');
const viewerStepControls = document.getElementById('viewer-step-controls');
const btnPrevStep = document.getElementById('btn-prev-step');
const btnNextStep = document.getElementById('btn-next-step');
const btnPlayStep = document.getElementById('btn-play-step');
const stepInfo = document.getElementById('step-info');

let activeModelParts = [];
let availableSteps = [];
let currentStepIndex = 0;
let isPlaying = false;
let playInterval = null;

btnUpload.addEventListener('click', () => {
    fileUpload.click();
});

fileUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    stopPlayback();
    logToTerminal(`Subiendo y analizando el archivo: ${file.name}...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch('http://localhost:8000/upload-mpd', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            activeModelParts = data.parts;
            logToTerminal(`Archivo procesado con éxito. Encontradas ${activeModelParts.length} piezas físicas.`);
            
            // Update piece count badge
            const badge = document.getElementById('viewer-piece-count');
            if (badge) {
                badge.textContent = `${activeModelParts.length} piezas`;
                badge.style.display = 'inline-block';
            }
            
            // Extract unique steps
            const steps = [...new Set(activeModelParts.map(p => p.step_id))];
            availableSteps = steps.sort((a, b) => a - b);
            currentStepIndex = 0;
            
            // Show step controls
            viewerTitle.textContent = `Modelo: ${file.name}`;
            viewerStepControls.style.display = 'flex';
            
            // Auto-center camera on model
            if (activeModelParts.length > 0) {
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                let minZ = Infinity, maxZ = -Infinity;
                
                activeModelParts.forEach(p => {
                    const m = p.transform;
                    const x = m[3];
                    const y = -m[7];
                    const z = m[11];
                    if (x < minX) minX = x; if (x > maxX) maxX = x;
                    if (y < minY) minY = y; if (y > maxY) maxY = y;
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
                });
                
                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const centerZ = (minZ + maxZ) / 2;
                
                controls.target.set(centerX, centerY, centerZ);
                
                // Adjust camera distance based on bounding box size
                const sizeX = maxX - minX;
                const sizeY = maxY - minY;
                const sizeZ = maxZ - minZ;
                const maxDim = Math.max(sizeX, sizeY, sizeZ, 100);
                
                camera.position.set(centerX + maxDim * 1.5, centerY + maxDim * 1.5, centerZ + maxDim * 1.5);
            }
            
            renderActiveModelForStep(0);
        } else {
            logToTerminal(`Error al parsear el archivo: ${data.message}`, 'warning');
        }
    } catch (e) {
        logToTerminal(`Fallo al subir archivo: ${e.message}`, 'warning');
    }
    
    // Reset file input value so same file can be uploaded again
    fileUpload.value = '';
});

function renderActiveModelForStep(stepIndex) {
    if (availableSteps.length === 0) return;
    const targetStep = availableSteps[stepIndex];
    stepInfo.textContent = `Paso: ${stepIndex + 1} / ${availableSteps.length}`;
    
    // Clean scene (except lights & helper)
    scene.children = scene.children.filter(c => c.type === 'AmbientLight' || c.type === 'DirectionalLight' || c.type === 'GridHelper');
    
    const partsToRender = activeModelParts.filter(p => p.step_id <= targetStep);
    
    partsToRender.forEach(part => {
        // Scale brick dynamically based on standard LDraw parts
        const { width, height, depth } = getPartDimensions(part.part_id);
        
        const geometry = new THREE.BoxGeometry(width, height, depth);
        geometry.translate(0, height / 2, 0); // Align origin to bottom face (LDraw style)
        const colorHex = LEGO_COLORS[part.color] || 0x888888;
        
        // Highlight current step parts & dim previous ones
        const isNewPart = (part.step_id === targetStep);
        const material = new THREE.MeshStandardMaterial({ 
            color: colorHex, 
            roughness: 0.2,
            metalness: 0.1,
            transparent: !isNewPart,
            opacity: isNewPart ? 1.0 : 0.45,
            emissive: isNewPart ? 0x3b82f6 : 0x000000, // Light blue emissive glow for new parts
            emissiveIntensity: isNewPart ? 0.4 : 0.0
        });
        
        const cube = new THREE.Mesh(geometry, material);
        
        // Apply global translation
        const m = part.transform;
        cube.position.set(m[3], -m[7], m[11]);
        
        // Apply Y-inversion mirrored rotation matrix
        const rotMatrix = new THREE.Matrix4().set(
             m[0], -m[1],  m[2], 0,
            -m[4],  m[5], -m[6], 0,
             m[8], -m[9],  m[10], 0,
             0,     0,     0,    1
        );
        cube.rotation.setFromRotationMatrix(rotMatrix);
        
        // Black outline, or blue outline if newly placed
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ 
            color: isNewPart ? 0x3b82f6 : 0x334155, 
            linewidth: isNewPart ? 2.5 : 1
        }));

        cube.add(line);
        
        scene.add(cube);
    });
}

btnPrevStep.addEventListener('click', () => {
    if (currentStepIndex > 0) {
        currentStepIndex--;
        renderActiveModelForStep(currentStepIndex);
    }
});

btnNextStep.addEventListener('click', () => {
    if (currentStepIndex < availableSteps.length - 1) {
        currentStepIndex++;
        renderActiveModelForStep(currentStepIndex);
    }
});

btnPlayStep.addEventListener('click', () => {
    if (isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
});

function startPlayback() {
    isPlaying = true;
    btnPlayStep.textContent = 'Detener';
    btnPlayStep.style.backgroundColor = '#ef4444';
    
    if (currentStepIndex >= availableSteps.length - 1) {
        currentStepIndex = 0;
        renderActiveModelForStep(currentStepIndex);
    }
    
    playInterval = setInterval(() => {
        if (currentStepIndex < availableSteps.length - 1) {
            currentStepIndex++;
            renderActiveModelForStep(currentStepIndex);
        } else {
            stopPlayback();
        }
    }, 1000);
}

function stopPlayback() {
    isPlaying = false;
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
    btnPlayStep.textContent = 'Reproducir Montaje';
    btnPlayStep.style.backgroundColor = '#10b981';
}

async function loadThemes() {
    try {
        const res = await fetch('http://localhost:8000/themes');
        const data = await res.json();
        const select = document.getElementById('input-theme');
        // Clear options keeping first one
        select.innerHTML = '<option value="All">Todos los temas</option>';
        if (data.themes) {
            data.themes.forEach(theme => {
                const opt = document.createElement('option');
                opt.value = theme;
                opt.textContent = theme;
                select.appendChild(opt);
            });
        }
    } catch(e) {
        console.warn("Error loading themes", e);
    }
}

let allModels = [];

async function loadModels() {
    try {
        const res = await fetch('http://localhost:8000/models');
        const data = await res.json();
        const select = document.getElementById('inference-model-select');
        const selectPartsModel = document.getElementById('parts-model-select');
        if (select) select.innerHTML = '<option value="">-- Seleccionar Modelo --</option>';
        if (selectPartsModel) selectPartsModel.innerHTML = '<option value="">-- Seleccionar Modelo --</option>';

        if (data.models) {
            allModels = data.models;
            data.models.forEach(model => {
                const text = `${model.name} (${model.theme}, <=${model.max_pieces} pcs)`;
                
                if (select) {
                    const opt = document.createElement('option');
                    opt.value = model.name;
                    opt.textContent = text;
                    select.appendChild(opt);
                }
                
                if (selectPartsModel) {
                    const optParts = document.createElement('option');
                    optParts.value = model.name;
                    optParts.textContent = text;
                    selectPartsModel.appendChild(optParts);
                }
            });
            renderModelsList();
        }
    } catch(e) {
        console.warn("Error loading models", e);
    }
}

function renderModelsList() {
    const listContainer = document.getElementById('models-list');
    if (!listContainer) return;
    
    const searchTerm = (document.getElementById('search-model')?.value || '').toLowerCase();
    
    listContainer.innerHTML = '';
    
    const filtered = allModels.filter(m => 
        m.name.toLowerCase().includes(searchTerm) || 
        m.theme.toLowerCase().includes(searchTerm)
    );
    
    if (filtered.length === 0) {
        listContainer.innerHTML = '<p style="color: #94a3b8;">No se encontraron modelos.</p>';
        return;
    }
    
    filtered.forEach(model => {
        const card = document.createElement('div');
        card.style.background = '#0f172a';
        card.style.padding = '1rem';
        card.style.borderRadius = '8px';
        card.style.border = '1px solid #334155';
        
        const piecesText = (model.allowed_parts && model.allowed_parts.length > 0) 
            ? model.allowed_parts.join(', ') 
            : 'Todas las piezas';
            
        card.innerHTML = `
            <h3 style="margin-top: 0; color: #38bdf8;">${model.name}</h3>
            <p style="margin: 0.25rem 0; font-size: 0.9rem;"><strong>Tema:</strong> ${model.theme}</p>
            <p style="margin: 0.25rem 0; font-size: 0.9rem;"><strong>Límite:</strong> ${model.max_pieces} piezas por set</p>
            <p style="margin: 0.25rem 0; font-size: 0.9rem;"><strong>Universo de piezas permitidas:</strong> <span style="font-size: 0.8rem; color: #cbd5e1; display: block; max-height: 60px; overflow-y: auto; background: #1e293b; padding: 4px; border-radius: 4px; margin-top: 4px; word-break: break-all;">${piecesText}</span></p>
            <button class="btn danger" style="margin-top: 0.5rem; width: 100%; background-color: #ef4444;" onclick="deleteModel('${model.name}')">Eliminar Modelo</button>
        `;
        listContainer.appendChild(card);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-model');
    if (searchInput) {
        searchInput.addEventListener('input', renderModelsList);
    }
});

window.deleteModel = async function(modelName) {
    if (!confirm(`¿Estás seguro de que quieres eliminar el modelo '${modelName}'?`)) return;
    
    try {
        const res = await fetch(`http://localhost:8000/models/${modelName}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (data.status === 'success') {
            logToTerminal(data.message);
            loadModels();
        } else {
            logToTerminal(`Error al eliminar: ${data.message}`, 'warning');
        }
    } catch (e) {
        logToTerminal(`Error de red al eliminar: ${e.message}`, 'warning');
    }
}

// Init logic
async function initData() {
    try {
        await loadThemes();
        await loadModels();
        
        const setsRes = await fetch('http://localhost:8000/local-sets');
        const setsData = await setsRes.json();
        const select = document.getElementById('local-sets-select');
        if (setsData.sets) {
            setsData.sets.forEach(set => {
                const opt = document.createElement('option');
                opt.value = set;
                opt.textContent = set;
                select.appendChild(opt);
            });
        }
    } catch(e) {
        console.warn("Could not load initial data", e);
    }
}
initData();

const btnLoadLocal = document.getElementById('btn-load-local');
const localSetsSelect = document.getElementById('local-sets-select');

btnLoadLocal.addEventListener('click', async () => {
    const filename = localSetsSelect.value;
    if (!filename) return;
    
    stopPlayback();
    logToTerminal(`Cargando archivo local OMR: ${filename}...`);
    
    try {
        const res = await fetch('http://localhost:8000/upload-local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            activeModelParts = data.parts;
            logToTerminal(`Archivo procesado con éxito. Encontradas ${activeModelParts.length} piezas físicas.`);
            
            // Update piece count badge
            const badge = document.getElementById('viewer-piece-count');
            if (badge) {
                badge.textContent = `${activeModelParts.length} piezas`;
                badge.style.display = 'inline-block';
            }
            
            // Extract unique steps
            const steps = [...new Set(activeModelParts.map(p => p.step_id))];
            availableSteps = steps.sort((a, b) => a - b);
            currentStepIndex = 0;
            
            // Show step controls
            viewerTitle.textContent = `Modelo: ${filename}`;
            viewerStepControls.style.display = 'flex';
            
            // Auto-center camera on model
            if (activeModelParts.length > 0) {
                let minX = Infinity, maxX = -Infinity;
                let minY = Infinity, maxY = -Infinity;
                let minZ = Infinity, maxZ = -Infinity;
                
                activeModelParts.forEach(p => {
                    const m = p.transform;
                    const x = m[3];
                    const y = -m[7];
                    const z = m[11];
                    if (x < minX) minX = x; if (x > maxX) maxX = x;
                    if (y < minY) minY = y; if (y > maxY) maxY = y;
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
                });
                
                const centerX = (minX + maxX) / 2;
                const centerY = (minY + maxY) / 2;
                const centerZ = (minZ + maxZ) / 2;
                
                controls.target.set(centerX, centerY, centerZ);
                
                // Adjust camera distance based on bounding box size
                const sizeX = maxX - minX;
                const sizeY = maxY - minY;
                const sizeZ = maxZ - minZ;
                const maxDim = Math.max(sizeX, sizeY, sizeZ, 100);
                
                camera.position.set(centerX + maxDim * 1.5, centerY + maxDim * 1.5, centerZ + maxDim * 1.5);
            }
            
            renderActiveModelForStep(0);
        } else {
            logToTerminal(`Error al parsear el archivo: ${data.message}`, 'warning');
        }
    } catch (e) {
        logToTerminal(`Fallo al cargar set local: ${e.message}`, 'warning');
    }
});

// ==========================================
// Tabs Navigation and Custom User Journey Flows
// ==========================================

window.switchTab = function(tabId) {
    // Switch active tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    // Find the button by checking its onclick attribute
    const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
    if (btn) btn.classList.add('active');
    
    // Switch active panels
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const panel = document.getElementById(`tab-${tabId}`);
    if (panel) panel.classList.add('active');
    
    logToTerminal(`Navegación: Pestaña '${tabId}' seleccionada.`);
};

// 1. Prompt Generation Event Handler
const btnGeneratePrompt = document.getElementById('btn-generate-prompt');
const inputPrompt = document.getElementById('input-prompt');
const inputPromptPieces = document.getElementById('input-prompt-pieces');
const renderPreviewContainer = document.getElementById('render-preview-container');
const renderPreviewPlaceholder = document.getElementById('render-preview-placeholder');
const renderPreviewImg = document.getElementById('render-preview-img');

btnGeneratePrompt.addEventListener('click', async () => {
    const promptText = inputPrompt.value.trim();
    if (!promptText) {
        logToTerminal('Por favor, introduce un prompt válido (ej: Silla roja).', 'warning');
        return;
    }
    
    const numPieces = parseInt(inputPromptPieces.value, 10) || 12;
    btnGeneratePrompt.disabled = true;
    logToTerminal(`Iniciando generación con LegoGPT para el prompt: "${promptText}" (${numPieces} piezas)...`);
    
    // Reset Eevee Render Preview
    renderPreviewImg.style.display = 'none';
    renderPreviewPlaceholder.style.display = 'block';
    renderPreviewPlaceholder.textContent = 'Renderizando en Blender...';
    
    try {
        const res = await fetch('http://localhost:8000/generate-prompt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: promptText,
                num_pieces: numPieces
            })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            logToTerminal(`LegoGPT generó con éxito ${data.parts.length} piezas.`);
            
            // Render parts in Three.js
            renderPartsInScene(data.parts);
            
            // Update Eevee preview if available
            if (data.render_url) {
                // Add timestamp to break browser cache
                renderPreviewImg.src = `http://localhost:8000${data.render_url}?t=${Date.now()}`;
                renderPreviewImg.style.display = 'block';
                renderPreviewPlaceholder.style.display = 'none';
                logToTerminal('Render de Blender Eevee cargado con éxito.');
            } else {
                renderPreviewPlaceholder.textContent = 'Render omitido (Blender no disponible)';
            }
        } else {
            logToTerminal(`Error en la generación: ${data.message}`, 'warning');
            renderPreviewPlaceholder.textContent = 'Error de generación';
        }
    } catch (err) {
        logToTerminal(`Fallo de red: ${err.message}`, 'warning');
        renderPreviewPlaceholder.textContent = 'Fallo de red';
    } finally {
        btnGeneratePrompt.disabled = false;
    }
});

// Helper to draw parts list in Three.js
function renderPartsInScene(parts) {
    // Clean scene
    scene.children = scene.children.filter(c => c.type === 'AmbientLight' || c.type === 'DirectionalLight' || c.type === 'GridHelper');
    
    // Update badge
    const badge = document.getElementById('viewer-piece-count');
    if (badge) {
        badge.textContent = `${parts.length} piezas`;
        badge.style.display = 'inline-block';
    }
    
    parts.forEach(part => {
        const { width, height, depth } = getPartDimensions(part.part_id);
        
        const geometry = new THREE.BoxGeometry(width, height, depth);
        geometry.translate(0, height / 2, 0);
        const colorHex = LEGO_COLORS[part.color] || 0x888888;
        const material = new THREE.MeshStandardMaterial({ color: colorHex, roughness: 0.2 });
        const cube = new THREE.Mesh(geometry, material);
        
        const m = part.transform;
        cube.position.set(m[3], -m[7], m[11]);
        
        const rotMatrix = new THREE.Matrix4().set(
             m[0], -m[1],  m[2], 0,
            -m[4],  m[5], -m[6], 0,
             m[8], -m[9],  m[10], 0,
             0,     0,     0,    1
        );
        cube.rotation.setFromRotationMatrix(rotMatrix);
        
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x000000 }));
        cube.add(line);
        
        scene.add(cube);
    });
    
    // Recenter camera
    if (parts.length > 0) {
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        
        parts.forEach(p => {
            const m = p.transform;
            const x = m[3];
            const y = -m[7];
            const z = m[11];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        });
        
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        
        controls.target.set(centerX, centerY, centerZ);
        
        const sizeX = maxX - minX;
        const sizeY = maxY - minY;
        const sizeZ = maxZ - minZ;
        const maxDim = Math.max(sizeX, sizeY, sizeZ, 100);
        
        camera.position.set(centerX + maxDim * 1.5, centerY + maxDim * 1.5, centerZ + maxDim * 1.5);
    }
}

// 2. Voxelizer File Upload / Drag-and-drop
const voxelFileUpload = document.getElementById('voxel-file-upload');
const voxelDropzone = document.getElementById('voxel-dropzone');

async function handleVoxelMeshUpload(file) {
    logToTerminal(`Voxelizando malla 3D: "${file.name}"...`);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch('http://localhost:8000/voxelize', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            logToTerminal(`Voxelización exitosa. Malla convertida en ${data.parts.length} piezas LEGO.`);
            renderPartsInScene(data.parts);
            
            // Hide render thumbnail when voxelizing (since we only show renders for prompt generations)
            renderPreviewImg.style.display = 'none';
            renderPreviewPlaceholder.style.display = 'block';
            renderPreviewPlaceholder.textContent = 'Voxelizador (Sin render)';
        } else {
            logToTerminal(`Error de voxelización: ${data.message}`, 'warning');
        }
    } catch (e) {
        logToTerminal(`Fallo al conectar con voxelizador: ${e.message}`, 'warning');
    }
}

voxelFileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleVoxelMeshUpload(file);
});

// Drag and drop events
voxelDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    voxelDropzone.style.borderColor = 'var(--primary-color)';
});

voxelDropzone.addEventListener('dragleave', () => {
    voxelDropzone.style.borderColor = 'rgba(255, 255, 255, 0.2)';
});

voxelDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    voxelDropzone.style.borderColor = 'rgba(255, 255, 255, 0.2)';
    const file = e.dataTransfer.files[0];
    if (file) handleVoxelMeshUpload(file);
});


