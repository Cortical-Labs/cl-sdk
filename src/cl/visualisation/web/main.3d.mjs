import * as THREE from '/visualiser/three.module.min.js';
import { connect, getAnalysisMsPerResult, getAnalysisToRender, getBufferSize, stimulate, isBuffering, setVisibleAbsUv } from '/visualiser/analysis.mjs';
import { rowColChannelLayout } from '/visualiser/layout.mjs';

connect(); // Start analysis module connection

// Channels 0, 7, 56, 63 are physically disconnected (corner electrodes)
// Skip rendering for these to save ~6% of waveform processing
const disconnectedChannels = new Set([0, 7, 56, 63]);

// Get URL parameters, supporting iframe embedding
let searchParams;
try {
    searchParams = window.parent?.location.search ?? window.location.search;

    // Merge with own search params to allow overrides
    const parentParams = new URLSearchParams(searchParams);
    const ownParams    = new URLSearchParams(window.location.search);
    for (const [key, value] of ownParams.entries()) {
        parentParams.set(key, value);
    }
    searchParams = parentParams.toString() ? `?${parentParams.toString()}` : '';
}
catch (e) {
    // Cross-origin iframe, fall back to own location
    searchParams = window.location.search;
}
const urlParams = new URLSearchParams(searchParams);

// The MEA endpoint is for showing the 3D visualiser only, and needs different camera defaults (as well as a full screen toggle)
const isMeaPage = window.location.pathname == '/mea/';

// Similar to the /vis endpoint, the ?jupyterMode=1 parameter indicates we're embedded in a Jupyter notebook.
// In Jupyter mode, click to stim is disabled, 2D mode is forced, keyboard shortcuts are disabled, and the visible range is fixed to 100uV.
// None of these updates the local storage, so the main web console UI is unaffected.
const isJupyterMode = urlParams.get('jupyterMode') === '1';

// In Jupyter mode, set fixed visible range to 100uV
if (isJupyterMode) {
    setVisibleAbsUv(100.0);
}

const focusOnChannels = urlParams.get('focusOnChannels')?.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

const cameraPos = isMeaPage ? { x: 0, y: 130, z: 69.5 } : { x: 0, y: 145, z: 59.5 };
const lookAt    = isMeaPage ? { x: 0, y:   0, z: 32   } : { x: 0, y:   0, z: 15   };

const cameraBaseFovDeg = 90;

const zoomLevels = {
    min    : { pos: { ...cameraPos }, lookAt: { ...lookAt } },
    max    : { pos: { x: 0, y: 15, z: 9 }, lookAt: { x: 0, y: 0, z: 4 } },
    current: { pos: { ...cameraPos }, lookAt: { ...lookAt } }
};
let currentZoom = 0.0;

function interpolate(min, max, fraction) {
    return min + (max - min) * fraction;
}

function interpolateZoom(t) {
    // t ranges from 0 (min zoom) to 1 (max zoom)
    return {
        pos: {
            x: interpolate(zoomLevels.min.pos.x, zoomLevels.max.pos.x, t),
            y: interpolate(zoomLevels.min.pos.y, zoomLevels.max.pos.y, t),
            z: interpolate(zoomLevels.min.pos.z, zoomLevels.max.pos.z, t)
        },
        lookAt: {
            x: interpolate(zoomLevels.min.lookAt.x, zoomLevels.max.lookAt.x, t),
            y: interpolate(zoomLevels.min.lookAt.y, zoomLevels.max.lookAt.y, t),
            z: interpolate(zoomLevels.min.lookAt.z, zoomLevels.max.lookAt.z, t)
        }
    };
}

const rightClickZoomLevelPercent = 85; // Percentage for right-click zoom level
const rightClickZoomLevel = interpolateZoom(rightClickZoomLevelPercent / 100);

function updateZoomFromSlider(value) {
    currentZoom = value / 100; // Convert 0-100 to 0-1
    const newZoom = interpolateZoom(currentZoom);

    // Update current zoom levels
    zoomLevels.current.pos.x    = newZoom.pos.x;
    zoomLevels.current.pos.y    = newZoom.pos.y;
    zoomLevels.current.pos.z    = newZoom.pos.z;
    zoomLevels.current.lookAt.x = newZoom.lookAt.x;
    zoomLevels.current.lookAt.y = newZoom.lookAt.y;
    zoomLevels.current.lookAt.z = newZoom.lookAt.z;

    // If currently zoomed in, update camera position immediately
    if (zoomedElectrode !== null || isFreeCameraMode) {
        navigateToElectrode(0, 0, 0); // Refresh current position with new zoom
    }
}

function showZoomControl(value = null) {
    const zoomControl = document.getElementById('zoom-control');
    if (zoomControl) {
        zoomControl.classList.remove('hide');
        zoomControl.classList.add('visible');
    }

    if (value === null)
        return;

    const zoomSlider = document.getElementById('zoom-slider');
    if (zoomSlider) {
        zoomSlider.value = value;
    }
}

function hideZoomControl() {
    const zoomControl = document.getElementById('zoom-control');
    if (zoomControl) {
        zoomControl.classList.remove('visible');
        zoomControl.classList.add('hide');
    }
}

let zoomedElectrode           = null;
let zoomedElectrodeGridPos    = null;
let currentAnimationFrameId   = null;
const zoomAnimationDurationMs = 300;

// Trackpad swipe gesture state
let   wheelDeltaX          = 0;
let   wheelDeltaY          = 0;
let   lastNavigationTime   = 0;
const navigationDebounceMs = 8;  // ~120fps for smooth movement

// Middle-click drag state (mirrors trackpad scrolling)
let isMiddleClickDragging  = false;
let middleClickStartX      = 0;
let middleClickStartY      = 0;
let middleClickLastX       = 0;
let middleClickLastY       = 0;

// Free camera mode (trackpad) vs snapped mode (keyboard)
let isFreeCameraMode   = false;
let freeCameraPosition = null;   // Stores current free camera target position

// Keep track of mouse position for tooltip
let mousePos = null;

// Raycasting throttle for performance (quality-dependent)
let lastRaycastTime = 0;
let raycastThrottleMs = 33;  // Default ~30fps, updated from quality settings
let pendingRaycastUpdate = null;

// Camera zoom configuration
const defaultCameraState = {
    pos   : { ...cameraPos },
    lookAt: { ...lookAt }
};

const windowStyle = getComputedStyle(document.body);

const gridSize           = 8;
const gapSizeX           = 8;
const gapSizeZ           = 4;
const electrodeWidth     = 40;
const electrodeDepth     = 28;
const measurementPlanes  = paramToNumber('linesPerElectrode', 1, 64, 8);
const frontPlaneIndex    = measurementPlanes - 1;
const measurementHeight  = 24;
const textureWidth       = paramToNumber('measurementsPerLine', 8, 1024, 192);
const spikeColour        = windowStyle.getPropertyValue('--electrode-spike-fill-colour');
const spikeOutlineColour = windowStyle.getPropertyValue('--electrode-spike-indicator-colour');
const stimColour         = windowStyle.getPropertyValue('--electrode-stim-fill-colour');
const stimOutlineColour  = windowStyle.getPropertyValue('--electrode-stim-colour');
const noSpikeColour      = windowStyle.getPropertyValue('--electrode-3d-min-max-colour');
const cellOutlineColour  = windowStyle.getPropertyValue('--electrode-3d-cell-outline-colour');

// Pre-create colour objects for reuse
const spikeColourObject        = new THREE.Color(spikeColour);
const spikeOutlineColourObject = new THREE.Color(spikeOutlineColour);
const stimColourObject         = new THREE.Color(stimColour);
const stimOutlineColourObject  = new THREE.Color(stimOutlineColour);
const noSpikeColourObject      = new THREE.Color(noSpikeColour);
const cellOutlineColourObject  = new THREE.Color(cellOutlineColour);

const DEBUG_MODE = paramToNumber('debugMode', 0, 1, 0) === 1;

// Quality presets for different hardware capabilities
// 'auto' will detect based on device, 'low', 'medium', 'high' are manual overrides
const qualityPreset = urlParams.get('quality') || 'auto';

// Auto-detect low-end devices
function detectLowEndDevice() {
    // Check for mobile/tablet devices
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    // Check for low core count (likely integrated GPU)
    const lowCoreCount = navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4;

    // Check device memory (if available, Chrome only)
    const lowMemory = navigator.deviceMemory && navigator.deviceMemory <= 4;

    // Check for high DPI on potentially weak device (tablets, low-end laptops)
    const highDpiLowEnd = window.devicePixelRatio > 1.5 && (isMobile || lowCoreCount);

    return isMobile || (lowCoreCount && lowMemory) || highDpiLowEnd;
}

// Quality settings based on preset
const qualitySettings = {
    potato: {
        tiltShiftEnabled : false,  // Disable all post-processing
        bloomEnabled     : false,
        blurTargetDivisor: 8,      // Minimal resolution (unused since bloom disabled)
        markerTextureSize: 192,    // Absolute minimum texture size
        maxAnisotropy    : 1,      // Disable anisotropic filtering
        tiltShiftTaps    : 3,      // Minimum taps (unused since tilt-shift disabled)
        raycastThrottleMs: 50,     // ~20fps for raycasting
        msaaSamples      : 0       // No anti-aliasing
    },
    low: {
        tiltShiftEnabled : false,
        bloomEnabled     : false,   // Disable bloom entirely for low-end devices
        blurTargetDivisor: 4,       // 1/4 resolution for blur passes
        markerTextureSize: 384,     // Smaller textures for low-end devices
        maxAnisotropy    : 1,       // Disable anisotropic filtering
        tiltShiftTaps    : 5,
        raycastThrottleMs: 33,      // ~30fps for raycasting
        msaaSamples      : 4        // Moderate MSAA
    },
    medium: {
        tiltShiftEnabled : true,
        bloomEnabled     : true,
        blurTargetDivisor: 2,      // 1/2 resolution for blur passes
        markerTextureSize: 768,
        maxAnisotropy    : 4,      // Moderate anisotropic filtering
        tiltShiftTaps    : 9,
        raycastThrottleMs: 16.67,  // ~60fps for raycasting
        msaaSamples      : 4       // Moderate MSAA
    },
    high: {
        tiltShiftEnabled : true,
        bloomEnabled     : true,
        blurTargetDivisor: 2,      // 1/2 resolution for blur passes
        markerTextureSize: 1536,
        maxAnisotropy    : 16,     // Max anisotropic filtering (capped by GPU)
        tiltShiftTaps    : 19,
        raycastThrottleMs: 8.33,   // ~120fps for raycasting (capped by screen refresh)
        msaaSamples      : 8       // Max MSAA
    }
};

// Resolve quality preset
let activeQuality;
if (qualityPreset === 'auto') {
    activeQuality = detectLowEndDevice() ? 'low' : 'high';
} else if (qualitySettings[qualityPreset]) {
    activeQuality = qualityPreset;
} else {
    activeQuality = 'high';
}

let currentQualitySettings = qualitySettings[activeQuality];
raycastThrottleMs = currentQualitySettings.raycastThrottleMs;
console.log(`MEA Visualizer: Using '${activeQuality}' quality preset`);

// Adaptive quality system - monitors FPS and auto-degrades if needed
const adaptiveQualityEnabled = paramToNumber('adaptive', 0, 1, 1) === 1; // Enabled by default
let adaptiveQualityState = {
    frameTimeHistory: [],           // Last N frame times in ms
    historySize: 60,                // ~1 second of frames at 60fps
    targetFrameTime: 16.67,         // 60fps target
    degradeThreshold: 20,           // Degrade if avg frame time > 20ms (~50fps)
    upgradeThreshold: 14,           // Upgrade if avg frame time < 14ms (~71fps)
    stableFramesNeeded: 120,        // Need 2 seconds of stable FPS before changing quality
    stableFrameCount: 0,            // Counter for stable frames
    lastQualityChange: 0,           // Timestamp of last quality change
    qualityCooldownMs: 5000,        // Wait 5 seconds between quality changes
    currentLevel: activeQuality     // Track current quality level
};

// Buffer history for plotting
const bufferHistory            = [];
const entriesHistory           = [];
const isBufferingHistory       = [];
const fpsHistory               = [];
const fpsAverageHistory        = [];
const markerCountHistory       = [];
const shiftTimeHistory         = [];
const addTimeHistory           = [];
const remainingTimeHistory     = [];
const pipelineSceneHistory     = [];   // Total GPU rendering time
const maxHistoryLength         = 512;  // Number of data points to display

// Setup colour change on prefers-color-scheme change
let isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
mediaQuery.addEventListener('change', (e) => {
    stopAnimation();
    isDarkMode = e.matches;
    spikeColourObject.set(windowStyle.getPropertyValue('--electrode-spike-fill-colour'));
    spikeOutlineColourObject.set(windowStyle.getPropertyValue('--electrode-spike-indicator-colour'));
    stimColourObject.set(windowStyle.getPropertyValue('--electrode-stim-fill-colour'));
    stimOutlineColourObject.set(windowStyle.getPropertyValue('--electrode-stim-colour'));
    noSpikeColourObject.set(windowStyle.getPropertyValue('--electrode-3d-min-max-colour'));
    cellOutlineColourObject.set(windowStyle.getPropertyValue('--electrode-3d-cell-outline-colour'));

    renderer.domElement.style.filter = windowStyle.getPropertyValue('--canvas-3d-filter');

    // Update geometry vertex colors directly
    refreshAllMarkerColors();
    setupPlaneMaterialCache(); // Ensure materials are refreshed

    continueAnimation();
});

const halfElectrodeWidth = electrodeWidth / 2;
const halfElectrodeDepth = electrodeDepth / 2;

const totalWidth = (gridSize * electrodeWidth) + ((gridSize - 1) * gapSizeX);
const totalDepth = (gridSize * electrodeDepth) + ((gridSize - 1) * gapSizeZ);
const left       = -totalWidth / 2;
const back       = -totalDepth / 2;

let firstTimestamp = undefined;
let lastTimestamp  = undefined;

const visualiserContainer = document.querySelector('#visualiser-container');
const rendererElement     = document.querySelector('#renderer');

const grid              = [];
const rowMarkers        = []; // Per-row, per-plane marker instanced meshes
const rowWaveforms      = []; // Per-row, per-plane merged waveform meshes
const planePositions    = [];
const channelBrightness = new Float32Array(64).fill(1.0);

const flashBrightness = paramToNumber('flashBrightness', 1, 10, 2.3, true);
const flashDecayRate  = paramToNumber('flashDecayRate', 0, 0.999, 0.99, true);

const minBloomStrength = paramToNumber('minBloomStrength', 0, 3, 0, true);
const maxBloomStrength = paramToNumber('maxBloomStrength', 0, 6, 6, true);
const bloomThreshold   = paramToNumber('bloomThreshold', 0, 1, 0.85, true);

let drawPlane = frontPlaneIndex;
let drawPos   = 0;

let animationRequestId;
let tooltip = null;

// Renderer stats for debug overlay
let lastDrawCalls = 0;
let lastTriangles = 0;

const dummy = new THREE.Object3D();

// Marker texture cache
let spikeMarkerTexture = null;
let stimMarkerTexture = null;

// Helper function to draw marker at specific size with style parameters
// mipmapLevel: 0 = highest resolution, higher = lower resolution
function drawMarker(ctx, size, fillColor, outlineColor, beamColor, markerType, mipmapLevel = 0) {
    ctx.clearRect(0, 0, size, size);

    const centerX = size / 2;

    // At lower mipmap levels, we need to:
    // 1. Slightly increase shape size to remain visible at low resolutions
    // 2. Ensure lines are at least 1-2 pixels wide to avoid subpixel rendering issues
    // 3. Use fully opaque colors without relying on antialiasing

    // Gentle scale compensation - enough to keep shapes visible but not oversized
    const scaleCompensation = 1.0 + (mipmapLevel * 0.05);

    // Minimum line width to avoid subpixel antialiasing artifacts
    const minLineWidth = Math.max(1.0, size * 0.015);

    // Outline width scales with size but has a minimum - don't apply scale compensation
    const lightModeOutlineBias = isDarkMode ? 1.0 : 0.9;
    const outlineWidth = Math.max(minLineWidth * (isDarkMode ? 1.5 : 1.0), size * 0.015 * lightModeOutlineBias);

    // Beam width - ensure it's always at least 1.5 pixels wide for crispness
    const beamWidth = Math.max(minLineWidth * 1.5, size * 0.02 * lightModeOutlineBias);
    const beamHeight = size * 0.80;

    // Disable antialiasing at lower mipmap levels for crisper edges
    if (mipmapLevel >= 3) {
        ctx.imageSmoothingEnabled = false;
    }

    if (markerType === 'spike') {
        // Draw circle marker
        // Scale radius up slightly at lower resolutions to maintain visual prominence
        const radius = size * 0.15 * scaleCompensation;

        // Position circle at top quarter of texture
        const circleY = size * 0.25;

        // Draw beam (vertical line from bottom of circle to bottom of texture)
        // Use rounded coordinates to avoid subpixel positioning
        const beamX = Math.round(centerX - beamWidth / 2);
        const beamY = Math.round(circleY + radius - (outlineWidth / 2));
        ctx.fillStyle = beamColor;
        ctx.globalAlpha = 1.0;
        ctx.fillRect(beamX, beamY, Math.round(beamWidth), Math.round(beamHeight + (outlineWidth / 2)));

        if (!isDarkMode) {
            // Draw fill circle - use full opacity
            ctx.fillStyle = fillColor;
            ctx.globalAlpha = 1.0;
            ctx.beginPath();
            ctx.arc(Math.round(centerX), Math.round(circleY), radius, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw outline ring with thicker stroke at lower resolutions
        ctx.strokeStyle = outlineColor;
        ctx.lineWidth = outlineWidth;
        ctx.globalAlpha = 1.0;
        ctx.beginPath();
        ctx.arc(Math.round(centerX), Math.round(circleY), radius - outlineWidth / 2, 0, Math.PI * 2);
        ctx.stroke();

    } else if (markerType === 'stim') {
        // Draw diamond marker (rotated square)
        // Scale diamond up slightly at lower resolutions
        const diamondSize = size * 0.36 * scaleCompensation;

        // Pre-round key values to ensure symmetry across all mipmap levels
        // Round half-dimensions to ensure left/right and top/bottom are equidistant from center
        const halfDiamond = Math.round(diamondSize / 2);
        const roundedCenterX = Math.round(centerX);
        const roundedCenterY = Math.round(size * 0.24);

        // Derive all points from pre-rounded values for perfect symmetry
        const topY = roundedCenterY - halfDiamond;
        const bottomY = roundedCenterY + halfDiamond;
        const leftX = roundedCenterX - halfDiamond;
        const rightX = roundedCenterX + halfDiamond;

        // Draw beam (vertical line from bottom of diamond to bottom of texture)
        const beamX = Math.round(roundedCenterX - beamWidth / 2);
        const beamY = Math.floor(bottomY - (outlineWidth / 2));
        ctx.fillStyle = beamColor;
        ctx.globalAlpha = 1.0;
        ctx.fillRect(beamX, beamY, Math.round(beamWidth), Math.round(beamHeight + (outlineWidth / 2)));

        // Draw fill diamond - use full opacity
        if (!isDarkMode) {
            ctx.fillStyle = fillColor;
            ctx.globalAlpha = 1.0;
            ctx.beginPath();
            ctx.moveTo(roundedCenterX, topY);      // Top point
            ctx.lineTo(rightX, roundedCenterY);    // Right point
            ctx.lineTo(roundedCenterX, bottomY);   // Bottom point
            ctx.lineTo(leftX, roundedCenterY);     // Left point
            ctx.closePath();
            ctx.fill();
        }

        // Draw outline diamond with thicker stroke at lower resolutions
        // Inset the outline by half the stroke width to keep it inside the fill
        const inset = Math.round(outlineWidth / 2);
        const diagInset = Math.round(outlineWidth * 0.707); // 1/√2 for 45° edges

        ctx.strokeStyle = outlineColor;
        ctx.globalAlpha = 1.0;
        ctx.lineWidth = outlineWidth;
        ctx.lineJoin = 'miter';
        ctx.beginPath();
        ctx.moveTo(roundedCenterX, topY + inset);              // Top point
        ctx.lineTo(rightX - diagInset, roundedCenterY);        // Right point
        ctx.lineTo(roundedCenterX, bottomY - inset);           // Bottom point
        ctx.lineTo(leftX + diagInset, roundedCenterY);         // Left point
        ctx.closePath();
        ctx.stroke();
    }
}

// Function to create a canvas-based texture for markers with custom mipmaps
function createMarkerTexture(fillColor, outlineColor, beamColor, markerType) {
    const baseSize = currentQualitySettings.markerTextureSize;  // Quality-dependent texture size
    const canvas   = document.createElement('canvas');
    canvas.width   = baseSize;
    canvas.height  = baseSize;
    const ctx      = canvas.getContext('2d', { alpha: true, willReadFrequently: false });

    // Enable antialiasing for smoother edges at high resolution
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw high-resolution version (base mipmap level 0)
    drawMarker(ctx, baseSize, fillColor, outlineColor, beamColor, markerType, 0);

    const texture = new THREE.CanvasTexture(canvas);

    // Generate custom mipmaps with adjusted rendering for each level
    const mipmaps     = [];
    let   currentSize = baseSize;
    let   mipmapLevel = 0;

    while (currentSize >= 16) {
        const mipmapCanvas  = document.createElement('canvas');
        mipmapCanvas.width  = currentSize;
        mipmapCanvas.height = currentSize;
        const mipmapCtx     = mipmapCanvas.getContext('2d', { alpha: true, willReadFrequently: false });

        // Enable high-quality antialiasing for higher resolution mipmaps
        // Disable for lower resolutions to get crisper pixels
        if (mipmapLevel < 3) {
            mipmapCtx.imageSmoothingEnabled = true;
            mipmapCtx.imageSmoothingQuality = 'high';
        } else {
            mipmapCtx.imageSmoothingEnabled = false;
        }

        // Pass mipmap level so drawMarker can adjust proportions
        drawMarker(mipmapCtx, currentSize, fillColor, outlineColor, beamColor, markerType, mipmapLevel);

        mipmaps.push(mipmapCanvas);

        currentSize = Math.floor(currentSize / 2);
        mipmapLevel++;
    }

    // Set the mipmaps on the texture
    texture.generateMipmaps = false; // Use our custom mipmaps
    texture.mipmaps = mipmaps;

    // Use LINEAR_MIPMAP_LINEAR for smooth transitions between mipmap levels
    // but the individual mipmaps themselves are pre-rendered with adjusted proportions
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;

    // Quality-dependent anisotropic filtering (capped by GPU capability)
    const maxGpuAnisotropy = renderer.capabilities.getMaxAnisotropy();
    texture.anisotropy = Math.min(currentQualitySettings.maxAnisotropy, maxGpuAnisotropy);

    // Disable premultiplied alpha since we're rendering with full opacity
    texture.premultiplyAlpha = false;

    // Slight negative bias to favor higher-res mipmaps (crisper at distance)
    // Negative = use higher resolution mipmaps, positive = use lower resolution sooner
    // texture.mipmapBias = -1.0;

    texture.needsUpdate = true;

    return texture;
}

function refreshAllMarkerColors() {
    const spikeBeamCol = isDarkMode ? spikeOutlineColourObject : spikeColourObject;
    const stimBeamCol  = isDarkMode ? stimOutlineColourObject : stimColourObject;

    // Regenerate marker textures
    spikeMarkerTexture = createMarkerTexture(
        spikeColourObject.getStyle(),
        spikeOutlineColourObject.getStyle(),
        spikeBeamCol.getStyle(),
        'spike'
    );

    stimMarkerTexture = createMarkerTexture(
        stimColourObject.getStyle(),
        stimOutlineColourObject.getStyle(),
        stimBeamCol.getStyle(),
        'stim'
    );
}

const planeMaterialCache = new Array(measurementPlanes);
planeMaterialCache.fill(null);

// Shared waveform materials - one per plane depth (reduces draw call state changes)
const sharedWaveformMaterials = new Array(measurementPlanes);
sharedWaveformMaterials.fill(null);

const maxFlashingSpikes = 32;

const minTiltShiftStrength = paramToNumber('minTiltShiftStrength', 0, 10, 0, true);
const maxTiltShiftStrength = paramToNumber('maxTiltShiftStrength', 0, 10, 3.0, true);
const tiltShiftFocusHeight = paramToNumber('tiltShiftFocusHeight', 0, 1, 0.5, true);
const tiltShiftFocusWidth  = paramToNumber('tiltShiftFocusWidth', 0, 1, 0.3, true);

const gridHalfWidth = totalWidth / 2 - halfElectrodeWidth;
const gridHalfDepth = totalDepth / 2 - halfElectrodeDepth;

const mouse = new THREE.Vector2();

let devicePixelRatio = window.devicePixelRatio || 1;

// Reusable vectors for render loop (avoid allocations per frame)
const _cameraPosition = new THREE.Vector3();
const _lookAtVector = new THREE.Vector3();
const _cameraVector = new THREE.Vector3();
const _cameraFloorVector = new THREE.Vector3(0, 0, 1);
const _electrodeWorldPos = new THREE.Vector3();

// Generate quality-dependent tilt-shift fragment shader
function generateTiltShiftFragmentShader(tapCount) {
    // Generate Gaussian weights for the specified tap count
    const halfTaps = Math.floor(tapCount / 2);
    const sigma = halfTaps / 2.5; // Adjust sigma based on tap count
    let weights = [];
    let totalWeight = 0;

    for (let i = 0; i <= halfTaps; i++) {
        const weight = Math.exp(-(i * i) / (2 * sigma * sigma));
        weights.push(weight);
        totalWeight += (i === 0) ? weight : weight * 2;
    }

    // Normalize weights
    weights = weights.map(w => (w / totalWeight).toFixed(6));

    // Generate weight array declaration
    const weightArraySize = halfTaps + 1;
    let weightDecl = `float weights[${weightArraySize}];\n`;
    for (let i = 0; i < weightArraySize; i++) {
        weightDecl += `            weights[${i}] = ${weights[i]};\n`;
    }

    return `
        uniform sampler2D tDiffuse;
        uniform vec2 resolution;
        uniform float focusHeight;
        uniform float focusWidth;
        uniform float blurStrength;
        uniform vec2 direction;
        varying vec2 vUv;

        void main() {
            // Do not apply tilt-shift if blur strength is zero
            if (blurStrength <= 0.0) {
                gl_FragColor = texture2D(tDiffuse, vUv);
                return;
            }

            // Calculate distance from focus center (vertical only)
            float distanceFromCenter = abs(vUv.y - focusHeight);

            // Calculate blur amount based on distance from focus area
            float blurAmount = smoothstep(focusWidth * 0.5, 1.0, distanceFromCenter / (1.0 - focusWidth * 0.5));
            blurAmount *= blurStrength;

            // Calculate chromatic aberration strength (correlates with blur)
            float chromaticStrength = blurAmount * 0.001;

            // Direction from center for radial chromatic aberration
            vec2 centerOffset = vUv - vec2(0.5, focusHeight);
            vec2 chromaticDirection = normalize(centerOffset);
            chromaticDirection.x *= 1.5;
            chromaticDirection.y *= 0.5;

            // Early exit for areas with minimal blur
            if (blurAmount < 0.05) {
                if (chromaticStrength > 0.00001) {
                    vec4 rSample = texture2D(tDiffuse, vUv + chromaticDirection * chromaticStrength * 0.5);
                    vec4 gSample = texture2D(tDiffuse, vUv);
                    vec4 bSample = texture2D(tDiffuse, vUv - chromaticDirection * chromaticStrength * 0.5);
                    float finalAlpha = max(max(rSample.a, gSample.a), bSample.a);
                    gl_FragColor = vec4(rSample.r, gSample.g, bSample.b, finalAlpha);
                } else {
                    gl_FragColor = texture2D(tDiffuse, vUv);
                }
                return;
            }

            // Use ${tapCount}-tap Gaussian kernel
            vec3  color       = vec3(0.0);
            float totalAlpha  = 0.0;
            float totalWeight = 0.0;

            ${weightDecl}

            float stepScale = 0.8 + blurAmount * 0.2;

            for (int i = -${halfTaps}; i <= ${halfTaps}; i++) {
                float weight = weights[abs(i)];
                vec2 offset = direction * float(i) * resolution * blurAmount * stepScale;

                vec4 rSample = texture2D(tDiffuse, vUv + offset + chromaticDirection * chromaticStrength);
                vec4 gSample = texture2D(tDiffuse, vUv + offset);
                vec4 bSample = texture2D(tDiffuse, vUv + offset - chromaticDirection * chromaticStrength);

                color += vec3(rSample.r, gSample.g, bSample.b) * weight;

                float sampleAlpha = max(max(rSample.a, gSample.a), bSample.a);
                totalAlpha += sampleAlpha * weight;
                totalWeight += weight;
            }

            gl_FragColor = vec4(color / totalWeight, totalAlpha / totalWeight);
        }
    `;
}

// Custom shader materials for tilt-shift effect (using quality-dependent tap count)
const tiltShiftShader = {
    uniforms: {
        tDiffuse    : { value: null },
        resolution  : { value: new THREE.Vector2(1.0 / (window.innerWidth * devicePixelRatio), 1.0 / (window.innerHeight * devicePixelRatio)) },
        focusHeight : { value: tiltShiftFocusHeight },
        focusWidth  : { value: tiltShiftFocusWidth },
        blurStrength: { value: minTiltShiftStrength },
        direction   : { value: new THREE.Vector2(1, 0) }
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: generateTiltShiftFragmentShader(currentQualitySettings.tiltShiftTaps)
};

// Custom shader materials for bloom effect
// Row-batched vertex shader - uses per-vertex brightness attribute and electrode index
const customVertexShader = `
    uniform float opacity;
    uniform float time;
    uniform float depthFactor;
    varying vec3 vColor;
    varying vec2 vUv;
    varying float vBrightness;
    varying float vElectrodeU;  // Normalized x position within electrode (0-1)
    varying float vElectrodeIdx; // Which electrode in the row (0-7)
    varying float vChannelOpacity; // Per-channel opacity for focus dimming

    attribute vec3 instanceColor;
    attribute float vertexBrightness;
    attribute float electrodeIndex;  // Which electrode this vertex belongs to
    attribute float electrodeU;      // Normalized U within the electrode
    attribute float channelOpacity;  // Per-electrode opacity (for focus dimming)

    void main() {
        vColor = instanceColor;
        vUv = uv;
        vBrightness = vertexBrightness;
        vElectrodeU = electrodeU;
        vElectrodeIdx = electrodeIndex;
        vChannelOpacity = channelOpacity;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

const customFragmentShader = `
    uniform float opacity;
    uniform float time;
    uniform float depthFactor;
    // Per-electrode spike flash data: xy = uv position, z = time, w = electrode index
    uniform vec4 spikeData[${maxFlashingSpikes}];
    uniform int activeSpikeCount;

    varying vec3 vColor;
    varying vec2 vUv;
    varying float vBrightness;
    varying float vElectrodeU;
    varying float vElectrodeIdx;
    varying float vChannelOpacity;

    void main() {
        vec3 baseColor = vColor;

        // Apply subtle cool tint for depth (subtle green shift)
        vec3 tintBase = vec3(0.0, 0.3, 0.0);
        vec3 depthTint = mix(tintBase, vec3(1.0, 1.0, 1.0), depthFactor);

        baseColor = baseColor * depthTint + (0.1 * tintBase * depthFactor);

        float flashIntensity = max(vBrightness - 1.0, 0.0);
        vec3 flashColor = vec3(1.0, 1.0, 1.0) * flashIntensity * 0.2 * depthFactor;

        // Accumulate effects from all active spikes (localized Gaussian flash)
        for (int i = 0; i < ${maxFlashingSpikes}; i++) {
            if (i >= activeSpikeCount) break;

            vec4 spike = spikeData[i];
            float spikeElectrodeIdx = spike.w;

            // Only apply flash if this fragment is on the same electrode as the spike
            if (abs(vElectrodeIdx - spikeElectrodeIdx) > 0.5) continue;

            float timeSinceSpike = time - spike.z;
            if (timeSinceSpike < 0.0 || timeSinceSpike > 2.0) continue;

            // Gaussian centered on spike U position within the electrode
            float uvDiff = vElectrodeU - spike.x;
            float sigma = 0.0075;
            float gaussian = exp(-uvDiff * uvDiff / (2.0 * sigma * sigma));
            float timeDecay = exp(-timeSinceSpike * 16.0);
            float intensity = gaussian * timeDecay * flashIntensity * depthFactor;

            flashColor += vec3(1.0, 0.94, 0.7) * intensity * 8.0;
        }

        vec3 finalColor = baseColor + flashColor;
        gl_FragColor = vec4(finalColor, opacity * vChannelOpacity);
    }
`;

// Track if WebGL is available
let webglAvailable = true;
let gl, renderer, scene, camera, raycaster, bloomRenderer;

try {
    const rendererResult = createRenderer();
    if (rendererResult) {
        ({ gl, renderer, scene, camera, raycaster, bloomRenderer } = rendererResult);
        refreshAllMarkerColors();
        start();
    } else {
        webglAvailable = false;
        showWebGLError();
    }
} catch (error) {
    console.error('Failed to initialize WebGL renderer:', error);
    webglAvailable = false;
    showWebGLError(error.message);
}

function showWebGLError(errorDetails = null) {
    // Hide the renderer element
    if (rendererElement) {
        rendererElement.style.display = 'none';
    }

    // Create error message container
    const errorContainer = document.createElement('div');
    errorContainer.id = 'webgl-error';
    errorContainer.innerHTML = `
        <div class="webgl-error-icon">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 9V13M12 17H12.01M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <h2>3D View Unavailable</h2>
        <p>WebGL could not be initialised. This may be because:</p>
        <ul>
            <li>Your browser doesn't support WebGL</li>
            <li>Hardware acceleration is disabled</li>
            <li>Your graphics drivers need updating</li>
            <li>WebGL is blocked by browser settings or extensions</li>
        </ul>
        ${errorDetails ? `<p class="webgl-error-details">Error: ${errorDetails}</p>` : ''}
        <p class="webgl-error-help">
            Try enabling hardware acceleration in your browser settings,
            or use a different browser like Chrome or Firefox.
        </p>
    `;

    // Insert into visualiser container
    if (visualiserContainer) {
        visualiserContainer.appendChild(errorContainer);
    }

    // Hide zoom control
    const zoomControl = document.getElementById('zoom-control');
    if (zoomControl) {
        zoomControl.style.display = 'none';
    }
}

// Helper function to find row and col for a given channel
function getRowColForChannel(channel) {
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            if (rowColChannelLayout[row][col] === channel) {
                return { row, col };
            }
        }
    }
    return null;
}

// Apply focus on channels from URL parameter
function applyFocusOnChannels() {
    if (!focusOnChannels || focusOnChannels.length === 0) {
        return;
    }

    // Filter out disconnected channels and channels that don't exist
    const validChannels = focusOnChannels.filter(ch =>ch >= 0 && ch < 64);

    if (validChannels.length === 0) {
        console.warn('No valid channels to focus on:', focusOnChannels);
        return;
    }

    if (validChannels.length === 1) {
        // Single channel: zoom to 80% (same as right-click)
        const channel = validChannels[0];
        const { row, col } = getRowColForChannel(channel);

        if (row === null || col === null) {
            console.warn(`Could not find row/col for channel ${channel}`);
            return;
        }

        const electrode = grid[col][row].electrode;

        // Get electrode world position
        const electrodeWorldPos = new THREE.Vector3();
        electrode.getWorldPosition(electrodeWorldPos);

        // Calculate zoom camera position at 80%
        const zoomedCameraPos = {
            x: electrodeWorldPos.x + rightClickZoomLevel.pos.x,
            y: rightClickZoomLevel.pos.y,
            z: electrodeWorldPos.z + rightClickZoomLevel.pos.z
        };

        const zoomedLookAt = {
            x: electrodeWorldPos.x + rightClickZoomLevel.lookAt.x,
            y: rightClickZoomLevel.lookAt.y,
            z: electrodeWorldPos.z + rightClickZoomLevel.lookAt.z
        };

        // Set up zoom state
        zoomedElectrode        = electrode;
        zoomedElectrodeGridPos = { row, col };
        isFreeCameraMode       = false;
        freeCameraPosition     = null;
        zoomLevels.current     = {
            pos   : { ...rightClickZoomLevel.pos },
            lookAt: { ...rightClickZoomLevel.lookAt }
        };
        currentZoom            = 0.8;

        // Set camera position directly (no animation on initial load)
        camera.position.set(zoomedCameraPos.x, zoomedCameraPos.y, zoomedCameraPos.z);
        camera.lookAt(zoomedLookAt.x, zoomedLookAt.y, zoomedLookAt.z);

        // Set bloom and tilt-shift
        setBloomStrength(interpolate(minBloomStrength, maxBloomStrength, currentZoom));
        setTiltShiftStrength(interpolate(minTiltShiftStrength, maxTiltShiftStrength, currentZoom));

        showZoomControl(80);

        console.log(`Focused on channel ${channel} at position (${row}, ${col})`);
    } else {
        // Multiple channels: find bounding box and calculate appropriate zoom
        const positions = [];

        for (const channel of validChannels) {
            const { row, col } = getRowColForChannel(channel);

            if (row === null || col === null) {
                console.warn(`Could not find row/col for channel ${channel}`);
                continue;
            }

            const electrode = grid[col][row].electrode;
            const worldPos = new THREE.Vector3();
            electrode.getWorldPosition(worldPos);
            positions.push({ x: worldPos.x, z: worldPos.z });
        }

        if (positions.length === 0) {
            console.warn('No valid electrode positions found');
            return;
        }

        // Calculate bounding box
        let minX = Infinity, maxX = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;

        for (const pos of positions) {
            minX = Math.min(minX, pos.x);
            maxX = Math.max(maxX, pos.x);
            minZ = Math.min(minZ, pos.z);
            maxZ = Math.max(maxZ, pos.z);
        }

        // Calculate center point
        const centerX = (minX + maxX) / 2;
        const centerZ = (minZ + maxZ) / 2;

        // Calculate required viewing area with some padding
        const requiredWidth = (maxX - minX) * 3;
        const requiredDepth = (maxZ - minZ) * 2.5;

        // Calculate the zoom level needed to fit the bounding box
        // We need to work backwards from the camera field of view
        // At different zoom levels, the camera sees different areas

        // The camera's field of view determines how much area is visible at a given distance
        // We need to find the zoom level where the visible area >= required area

        // Using camera FOV and position to calculate visible dimensions at y=0 plane
        const fovRadians = (cameraBaseFovDeg * Math.PI) / 180;

        // Binary search for the appropriate zoom level
        let lowZoom = 0.0;
        let highZoom = 1.0;
        let bestZoom = 0.0;
        const iterations = 20; // Binary search iterations for precision

        for (let i = 0; i < iterations; i++) {
            const testZoom = (lowZoom + highZoom) / 2;
            const testZoomState = interpolateZoom(testZoom);

            // Calculate visible area at this zoom level
            const cameraHeight = testZoomState.pos.y;
            const lookAtHeight = testZoomState.lookAt.y;
            const effectiveHeight = cameraHeight - lookAtHeight;

            // Visible height at y=0 plane
            const visibleHeight = 2 * effectiveHeight * Math.tan(fovRadians / 2);

            // Aspect ratio
            const aspect = window.innerWidth / window.innerHeight;
            const visibleWidth = visibleHeight * aspect;

            // Check if this zoom level fits our required area
            if (visibleWidth >= requiredWidth && visibleHeight >= requiredDepth) {
                // This zoom level works, try zooming in more
                bestZoom = testZoom;
                lowZoom = testZoom;
            } else {
                // Need to zoom out more
                highZoom = testZoom;
            }
        }

        if (bestZoom > 0.01) {
            // Apply the calculated zoom level
            const finalZoomState = interpolateZoom(bestZoom);

            // Set up free camera mode at the center point
            zoomedElectrode = null;
            zoomedElectrodeGridPos = null;
            isFreeCameraMode = true;
            freeCameraPosition = { x: centerX, z: centerZ };
            zoomLevels.current = {
                pos: { ...finalZoomState.pos },
                lookAt: { ...finalZoomState.lookAt }
            };
            currentZoom = bestZoom;

            // Calculate final camera position
            const finalCameraPos = {
                x: centerX + finalZoomState.pos.x,
                y: finalZoomState.pos.y,
                z: centerZ + finalZoomState.pos.z
            };

            const finalLookAt = {
                x: centerX + finalZoomState.lookAt.x,
                y: finalZoomState.lookAt.y,
                z: centerZ + finalZoomState.lookAt.z
            };

            // Set camera position directly (no animation on initial load)
            camera.position.set(finalCameraPos.x, finalCameraPos.y, finalCameraPos.z);
            camera.lookAt(finalLookAt.x, finalLookAt.y, finalLookAt.z);

            // Set bloom and tilt-shift
            setBloomStrength(interpolate(minBloomStrength, maxBloomStrength, currentZoom));
            setTiltShiftStrength(interpolate(minTiltShiftStrength, maxTiltShiftStrength, currentZoom));

            showZoomControl(bestZoom * 100);

            console.log(`Focused on ${validChannels.length} channels at center (${centerX.toFixed(1)}, ${centerZ.toFixed(1)}) with zoom ${(bestZoom * 100).toFixed(1)}%`);
        }
        else {
            console.log('Calculated zoom level is too close to default view; no zoom applied.');
        }
    }

    // Apply opacity reduction to non-focused channels
    applyChannelOpacity(validChannels);
}

// Apply opacity to channels based on focus
function applyChannelOpacity(focusedChannels) {
    const focusedSet = new Set(focusedChannels);
    const dimmedOpacity = 0.35;
    const fullOpacity = 1.0;

    // Update waveform opacity
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        for (let colIndex = 0; colIndex < gridSize; colIndex++) {
            const channel = rowColChannelLayout[rowIndex][colIndex];

            // Skip disconnected channels
            if (disconnectedChannels.has(channel)) continue;

            const isFocused = focusedSet.has(channel);
            const targetOpacity = isFocused ? fullOpacity : dimmedOpacity;

            // Set opacity for all planes in this row
            for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
                const waveformPlane = rowWaveforms[rowIndex][planeIndex];

                // Update channelOpacity attribute for all vertices of this electrode
                const channelOpacityAttr = waveformPlane.geometry.attributes.channelOpacity.array;
                const electrodeSegmentOffset = colIndex * textureWidth;

                for (let seg = 0; seg < textureWidth; seg++) {
                    const segmentIndex = electrodeSegmentOffset + seg;
                    for (let v = 0; v < 4; v++) {
                        const idx = segmentIndex * 4 + v;
                        channelOpacityAttr[idx] = targetOpacity;
                    }
                }
                waveformPlane.geometry.attributes.channelOpacity.needsUpdate = true;
            }
        }
    }
}

function start() {
    doRendererLayout();
    window.addEventListener('resize', doRendererLayout, false);
    window.addEventListener('keydown', onKeyDown, false);
    window.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('mousedown', onMouseDown, false);
    window.addEventListener('mousemove', onMouseMove, false);
    window.addEventListener('mouseup', onMouseUp, false);
    createTooltip();
    createMea();
    applyFocusOnChannels();
    continueAnimation();

    // Listen for changes in the DPI so we can do a renderer layout
    function listenOnDevicePixelRatio() {
        function onChange() {
            doRendererLayout();
            listenOnDevicePixelRatio();
        }
        matchMedia(
            `(resolution: ${devicePixelRatio}dppx)`
        ).addEventListener("change", onChange, { once: true });
    }
    listenOnDevicePixelRatio();

    rendererElement.addEventListener(
        'visibilitychange',
        function () {
            const zoomControl = document.getElementById('zoom-control');
            if (rendererElement.style.display === 'none' || rendererElement.hidden)
            {
                stopAnimation();
                if (zoomControl)
                    zoomControl.style.display = 'none'; // Use display property to prevent animation
            }
            else
            {
                continueAnimation();
                if (zoomControl)
                    zoomControl.style.display = 'flex';
            }
        });

    // Initialize zoom slider
    const zoomSlider = document.getElementById('zoom-slider');

    if (zoomSlider) {
        zoomSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (value <= 0)
                zoomOutToDefaultView();
            else
                updateZoomFromSlider(value);
        });
    }

    const dcOffsetCorrectionCheckbox = document.getElementById('dc-offset-correction');
    if (dcOffsetCorrectionCheckbox && urlParams.get('enableDcToggle') != null) {
        dcOffsetCorrectionCheckbox.classList.remove('hidden');
        const dcOffsetCorrectionCheckboxLabel = document.getElementById('dc-offset-correction-label');
        dcOffsetCorrectionCheckboxLabel?.classList.remove('hidden');
    }
}

function continueAnimation() {
    stopAnimation();
    resetMeaPlanePositions();
    firstTimestamp     = undefined;
    lastTimestamp      = undefined;
    animationRequestId = requestAnimationFrame(render);
}

// Adaptive quality monitoring - call this each frame with frame time
function updateAdaptiveQuality(frameTimeMs) {
    if (!adaptiveQualityEnabled) return;

    const state = adaptiveQualityState;
    const now = performance.now();

    // Add frame time to history
    state.frameTimeHistory.push(frameTimeMs);
    if (state.frameTimeHistory.length > state.historySize) {
        state.frameTimeHistory.shift();
    }

    // Need enough samples before making decisions
    if (state.frameTimeHistory.length < state.historySize) return;

    // Check cooldown
    if (now - state.lastQualityChange < state.qualityCooldownMs) return;

    // Calculate average frame time
    const avgFrameTime = state.frameTimeHistory.reduce((a, b) => a + b, 0) / state.frameTimeHistory.length;

    // Determine if we should change quality
    const qualityOrder = ['low', 'medium', 'high'];
    const currentIndex = qualityOrder.indexOf(state.currentLevel);

    let targetLevel = state.currentLevel;

    if (avgFrameTime > state.degradeThreshold && currentIndex > 0) {
        // Performance is poor, try to degrade quality
        state.stableFrameCount++;
        if (state.stableFrameCount >= state.stableFramesNeeded) {
            targetLevel = qualityOrder[currentIndex - 1];
        }
    } else if (avgFrameTime < state.upgradeThreshold && currentIndex < qualityOrder.length - 1) {
        // Performance is good, try to upgrade quality
        state.stableFrameCount++;
        if (state.stableFrameCount >= state.stableFramesNeeded) {
            targetLevel = qualityOrder[currentIndex + 1];
        }
    } else {
        // Performance is acceptable, reset stable counter
        state.stableFrameCount = 0;
    }

    // Apply quality change if needed
    if (targetLevel !== state.currentLevel) {
        console.log(`Adaptive Quality: Changing from '${state.currentLevel}' to '${targetLevel}' (avg frame time: ${avgFrameTime.toFixed(2)}ms)`);

        const previousSettings = currentQualitySettings;
        state.currentLevel = targetLevel;
        state.lastQualityChange = now;
        state.stableFrameCount = 0;
        state.frameTimeHistory = []; // Reset history after change

        // Apply new quality settings
        currentQualitySettings = qualitySettings[targetLevel];

        // Update raycast throttle for tooltip responsiveness
        raycastThrottleMs = currentQualitySettings.raycastThrottleMs;

        // Update tilt-shift enabled state (the shader tap count can't change without recreating materials)
        // But we can disable/enable tilt-shift entirely
        if (!currentQualitySettings.tiltShiftEnabled) {
            setTiltShiftStrength(0);
        }

        // Regenerate marker textures if texture size or anisotropy changed
        if (previousSettings.markerTextureSize !== currentQualitySettings.markerTextureSize ||
            previousSettings.maxAnisotropy !== currentQualitySettings.maxAnisotropy) {
            console.log(`Adaptive Quality: Regenerating marker textures (${previousSettings.markerTextureSize} -> ${currentQualitySettings.markerTextureSize})`);
            refreshAllMarkerColors();
            setupPlaneMaterialCache();
        }
    }
}

function setupPlaneMaterialCache() {
    if (planeMaterialCache[0] !== null) {
        // Reset existing materials and update textures
        for (let i = 0; i < measurementPlanes; i++) {
            planeMaterialCache[i].stim.map = stimMarkerTexture;
            planeMaterialCache[i].stim.needsUpdate = true;
            planeMaterialCache[i].stim.opacity  = 1;
            planeMaterialCache[i].stim.visible  = true;
            planeMaterialCache[i].spike.map = spikeMarkerTexture;
            planeMaterialCache[i].spike.needsUpdate = true;
            planeMaterialCache[i].spike.opacity = 1;
            planeMaterialCache[i].spike.visible = true;

            // Reset shared waveform material uniforms
            if (sharedWaveformMaterials[i] !== null) {
                sharedWaveformMaterials[i].uniforms.opacity.value = 1.0;
                sharedWaveformMaterials[i].uniforms.depthFactor.value = 1.0;
            }
        }
        return;
    }

    console.log('Initializing plane material cache');

    for (let i = 0; i < measurementPlanes; i++) {
        const stimMaterial = new THREE.MeshBasicMaterial({
            map        : stimMarkerTexture,
            transparent: true,
            opacity    : 0.85,
            depthWrite : false,
            side       : THREE.FrontSide,
            blending   : THREE.CustomBlending,
            blendSrc   : THREE.SrcAlphaFactor,
            blendDst   : THREE.OneMinusSrcAlphaFactor,
            blendSrcAlpha: THREE.OneFactor,
            blendDstAlpha: THREE.OneFactor,
        });

        const spikeMaterial = new THREE.MeshBasicMaterial({
            map        : spikeMarkerTexture,
            transparent: true,
            opacity    : 0.85,
            depthWrite : false,
            side       : THREE.FrontSide,
            blending   : THREE.CustomBlending,
            blendSrc   : THREE.SrcAlphaFactor,
            blendDst   : THREE.OneMinusSrcAlphaFactor,
            blendSrcAlpha: THREE.OneFactor,
            blendDstAlpha: THREE.OneFactor,
        });

        planeMaterialCache[i] = {
            stim : stimMaterial,
            spike: spikeMaterial
        };

        // Create shared waveform material for this plane depth
        // Initialize spike data array for uniforms
        const spikeDataArray = new Array(maxFlashingSpikes).fill(null).map(() => new THREE.Vector4(0, 0, 0, -1));

        sharedWaveformMaterials[i] = new THREE.ShaderMaterial({
            vertexShader: customVertexShader,
            fragmentShader: customFragmentShader,
            uniforms: {
                opacity         : { value: 1.0 },
                time            : { value: 0.0 },
                depthFactor     : { value: 1.0 },
                spikeData       : { value: spikeDataArray },
                activeSpikeCount: { value: 0 },
            },
            transparent: true,
            blending: THREE.NormalBlending,
            depthWrite: false,
        });
    }
}

function clearGeometries() {
    // Clear row-level waveform geometries
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        if (!rowWaveforms[rowIndex]) continue;

        for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
            const rowPlane = rowWaveforms[rowIndex][planeIndex];
            if (!rowPlane) continue;

            // Reset all positions to zero
            const positions = rowPlane.geometry.attributes.position.array;
            positions.fill(0);
            rowPlane.geometry.attributes.position.needsUpdate = true;

            // Reset brightness attribute
            const brightness = rowPlane.geometry.attributes.vertexBrightness.array;
            brightness.fill(1.0);
            rowPlane.geometry.attributes.vertexBrightness.needsUpdate = true;

            // Reset material uniforms
            if (rowPlane.material) {
                rowPlane.material.uniforms.opacity.value = 1.0;
                rowPlane.material.uniforms.depthFactor.value = 1.0;
                rowPlane.material.uniforms.activeSpikeCount.value = 0;
            }

            // Clear spike data
            if (rowPlane.spikes) {
                rowPlane.spikes = [];
            }
        }
    }

    // Reset channel brightness
    channelBrightness.fill(1.0);

    // Clear row-level marker instance counts
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        if (!rowMarkers[rowIndex]) continue;

        for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
            const rowPlane = rowMarkers[rowIndex][planeIndex];
            rowPlane.instancedMeshes.spike.count = 0;
            rowPlane.instancedMeshes.spike.instanceMatrix.needsUpdate = true;
            rowPlane.instancedMeshes.stim.count = 0;
            rowPlane.instancedMeshes.stim.instanceMatrix.needsUpdate = true;
        }
    }

    setupPlaneMaterialCache();
}

function setBloomStrength(newStrength) {
    // Clamp to valid range
    newStrength = Math.max(0, Math.min(6, newStrength));

    // Update the shader uniform
    bloomRenderer.compositeMaterial.uniforms.bloomStrength.value = newStrength * devicePixelRatio;
}

function setTiltShiftStrength(newStrength) {
    // Clamp to valid range
    newStrength = Math.max(0, Math.min(10, newStrength));

    // Update the shader uniform
    bloomRenderer.tiltShiftMaterialH.uniforms.blurStrength.value = newStrength * devicePixelRatio;
    bloomRenderer.tiltShiftMaterialV.uniforms.blurStrength.value = newStrength * devicePixelRatio;
}

function stopAnimation() {
    if (!animationRequestId)
        return;

    cancelAnimationFrame(animationRequestId);
    animationRequestId = undefined;
    clearGeometries();
    bloomRenderer.render();
}

function onRendererMouseClick(event) {
    event.preventDefault();

    const rect = renderer.domElement.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = - ((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    // Disable click-to-stimulate in Jupyter mode
    if (isJupyterMode)
        return;

    let isStimulationEnabled = false;
    const clickToStimulateChannel = document.getElementById('click-to-stimulate');
    if (clickToStimulateChannel)
        isStimulationEnabled = clickToStimulateChannel.checked;

    if (isStimulationEnabled && intersects.length > 0) {
        stimulate(intersects[0].object.userData.channel);
    }
}

function handlePinchZoom(event) {
    event.preventDefault();

    // If currently not zoomed in
    if (zoomedElectrode === null && !isFreeCameraMode) {
        // If deltaY is < 0, we zoom in by doing the same thing as a right click
        if (event.deltaY < 0) {

            zoomedElectrode        = null;
            zoomedElectrodeGridPos = null;
            isFreeCameraMode       = true;  // Start in free camera mode
            if (freeCameraPosition === null)
                freeCameraPosition = { x: 0, z: 0 };
            zoomLevels.current = {
                pos   : { ...rightClickZoomLevel.pos },
                lookAt: { ...rightClickZoomLevel.lookAt }
            };

            animateCameraTo(defaultCameraState.pos, defaultCameraState.lookAt, 0);

            showZoomControl(0);
        }

        return;
    }

    // Get current slider value
    const zoomSlider = document.getElementById('zoom-slider');
    const currentSliderValue = zoomSlider ? parseFloat(zoomSlider.value) : (currentZoom * 100);

    // Get mouse position in world space at current zoom level
    const rect = renderer.domElement.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    camera.updateMatrixWorld();

    // Find intersection with the MEA plane (y=0)
    const planeNormal = new THREE.Vector3(0, 1, 0);
    const targetPoint = new THREE.Vector3();
    raycaster.ray.intersectPlane(new THREE.Plane(planeNormal, 0), targetPoint);

    // If the slider is at the minimum, and deltaY is > 0, we zoom out by doing the same thing as a right click
    if (currentSliderValue <= 0 && event.deltaY > 0) {
        zoomOutToDefaultView();
        return;
    }

    // Calculate zoom delta from this event's deltaY
    // Negative deltaY = zoom in (pinch out), positive = zoom out (pinch in)
    const sensitivityScale = 1.0;
    const sliderDelta      = -event.deltaY * sensitivityScale;

    // Calculate new slider value by adding delta to current value
    let newSliderValue = currentSliderValue + sliderDelta;

    // Clamp to bounds (0-100)
    newSliderValue = Math.max(0, Math.min(100, newSliderValue));

    if (newSliderValue === currentSliderValue) {
        return;
    }

    // Update the slider UI
    if (zoomSlider) {
        zoomSlider.value = newSliderValue;
    }

    // Calculate how much to shift the camera to zoom towards pointer
    if (targetPoint && isFreeCameraMode) {
        const oldZoom = interpolateZoom(currentSliderValue / 100);
        const newZoom = interpolateZoom(newSliderValue / 100);

        // The scale factor is the ratio of camera heights
        const oldScale   = oldZoom.pos.y;
        const newScale   = newZoom.pos.y;
        const scaleRatio = newScale / oldScale;

        // Calculate offset from camera to target point in the horizontal plane
        const currentCameraX = freeCameraPosition.x;
        const currentCameraZ = freeCameraPosition.z;

        const targetOffsetX = targetPoint.x - currentCameraX;
        const targetOffsetZ = targetPoint.z - currentCameraZ;

        // Adjust free camera position: new = target - (target - current) * scaleRatio
        freeCameraPosition.x = targetPoint.x - targetOffsetX * scaleRatio;
        freeCameraPosition.z = targetPoint.z - targetOffsetZ * scaleRatio;

        // Clamp to grid boundaries
        freeCameraPosition.x = Math.max(-gridHalfWidth, Math.min(gridHalfWidth, freeCameraPosition.x));
        freeCameraPosition.z = Math.max(-gridHalfDepth, Math.min(gridHalfDepth, freeCameraPosition.z));
    }

    // Apply the zoom
    updateZoomFromSlider(newSliderValue);
}

function onWheel(event) {
    // Handle pinch-to-zoom gestures (ctrlKey is true for those)
    if (event.ctrlKey) {
        handlePinchZoom(event);
        return;
    }

    // Only handle trackpad swipes when zoomed in or in free camera mode
    if (zoomedElectrode === null && !isFreeCameraMode) {
        return;
    }

    // Ignore standard mouse wheel events - only process high-precision trackpad/touchpad
    // deltaMode: 0 = pixels (trackpad), 1 = lines (mouse wheel), 2 = pages
    if (event.deltaMode !== 0) {
        return;
    }

    // Prevent default scrolling
    event.preventDefault();

    // Accumulate wheel delta
    wheelDeltaX += event.deltaX;
    wheelDeltaY += event.deltaY;

    // Debounce for smooth continuous movement
    const now = performance.now();
    if (now - lastNavigationTime < navigationDebounceMs) {
        return;
    }

    // Switch to free camera mode on first trackpad movement
    if (!isFreeCameraMode) {
        isFreeCameraMode = true;
        // Initialize free camera position from current electrode or camera position
        if (zoomedElectrode) {
            const electrodeWorldPos = new THREE.Vector3();
            zoomedElectrode.getWorldPosition(electrodeWorldPos);
            freeCameraPosition = {
                x: electrodeWorldPos.x,
                z: electrodeWorldPos.z
            };
        } else {
            freeCameraPosition = {
                x: camera.position.x - zoomLevels.current.pos.x,
                z: camera.position.z - zoomLevels.current.pos.z
            };
        }
        zoomedElectrode = null;
        zoomedElectrodeGridPos = null;
    }

    // Calculate movement scale (pixels of delta to world units)
    const movementScale = 0.15; // Adjust this for sensitivity

    // Update free camera position
    freeCameraPosition.x += wheelDeltaX * movementScale;
    freeCameraPosition.z += wheelDeltaY * movementScale;

    // Clamp to grid boundaries
    freeCameraPosition.x = Math.max(-gridHalfWidth, Math.min(gridHalfWidth, freeCameraPosition.x));
    freeCameraPosition.z = Math.max(-gridHalfDepth, Math.min(gridHalfDepth, freeCameraPosition.z));

    // Calculate new camera position
    const newCameraPos = {
        x: freeCameraPosition.x + zoomLevels.current.pos.x,
        y: zoomLevels.current.pos.y,
        z: freeCameraPosition.z + zoomLevels.current.pos.z
    };

    const newLookAt = {
        x: freeCameraPosition.x + zoomLevels.current.lookAt.x,
        y: zoomLevels.current.lookAt.y,
        z: freeCameraPosition.z + zoomLevels.current.lookAt.z
    };

    // Smoothly move camera (no animation, just update directly for responsiveness)
    camera.position.x = newCameraPos.x;
    camera.position.y = newCameraPos.y;
    camera.position.z = newCameraPos.z;
    camera.lookAt(newLookAt.x, newLookAt.y, newLookAt.z);

    // Update tooltip if hovering
    if (mousePos !== null) {
        updateTooltipAndOutline(mousePos.x, mousePos.y);
    }

    lastNavigationTime = now;

    // Reset accumulated delta
    wheelDeltaX = 0;
    wheelDeltaY = 0;
}

function onMouseDown(event) {
    // Only handle middle-click (button 1)
    if (event.button !== 1) return;

    // Only handle when zoomed in or in free camera mode
    if (zoomedElectrode === null && !isFreeCameraMode) return;

    event.preventDefault();

    isMiddleClickDragging = true;
    middleClickStartX = event.clientX;
    middleClickStartY = event.clientY;
    middleClickLastX = event.clientX;
    middleClickLastY = event.clientY;

    // Change cursor to grabbing
    renderer.domElement.style.cursor = 'grabbing';
}

function onMouseMove(event) {
    if (!isMiddleClickDragging) return;

    event.preventDefault();

    // Calculate delta from last position (similar to trackpad deltaX/deltaY)
    const deltaX = event.clientX - middleClickLastX;
    const deltaY = event.clientY - middleClickLastY;

    middleClickLastX = event.clientX;
    middleClickLastY = event.clientY;

    // Skip if no movement
    if (deltaX === 0 && deltaY === 0) return;

    // Switch to free camera mode on first drag movement (same as trackpad)
    if (!isFreeCameraMode) {
        isFreeCameraMode = true;
        // Initialize free camera position from current electrode or camera position
        if (zoomedElectrode) {
            const electrodeWorldPos = new THREE.Vector3();
            zoomedElectrode.getWorldPosition(electrodeWorldPos);
            freeCameraPosition = {
                x: electrodeWorldPos.x,
                z: electrodeWorldPos.z
            };
        } else {
            freeCameraPosition = {
                x: camera.position.x - zoomLevels.current.pos.x,
                z: camera.position.z - zoomLevels.current.pos.z
            };
        }
        zoomedElectrode = null;
        zoomedElectrodeGridPos = null;
    }

    // Calculate movement scale (same as trackpad scrolling)
    const movementScale = 0.15;

    // Update free camera position (note: inverted to feel like dragging the view)
    freeCameraPosition.x -= deltaX * movementScale;
    freeCameraPosition.z -= deltaY * movementScale;

    // Clamp to grid boundaries
    freeCameraPosition.x = Math.max(-gridHalfWidth, Math.min(gridHalfWidth, freeCameraPosition.x));
    freeCameraPosition.z = Math.max(-gridHalfDepth, Math.min(gridHalfDepth, freeCameraPosition.z));

    // Calculate new camera position
    const newCameraPos = {
        x: freeCameraPosition.x + zoomLevels.current.pos.x,
        y: zoomLevels.current.pos.y,
        z: freeCameraPosition.z + zoomLevels.current.pos.z
    };

    const newLookAt = {
        x: freeCameraPosition.x + zoomLevels.current.lookAt.x,
        y: zoomLevels.current.lookAt.y,
        z: freeCameraPosition.z + zoomLevels.current.lookAt.z
    };

    // Update camera position directly for responsiveness
    camera.position.x = newCameraPos.x;
    camera.position.y = newCameraPos.y;
    camera.position.z = newCameraPos.z;
    camera.lookAt(newLookAt.x, newLookAt.y, newLookAt.z);

    // Update tooltip if hovering
    if (mousePos !== null) {
        updateTooltipAndOutline(mousePos.x, mousePos.y);
    }
}

function onMouseUp(event) {
    // Only handle middle-click (button 1)
    if (event.button !== 1) return;

    if (isMiddleClickDragging) {
        isMiddleClickDragging = false;
        renderer.domElement.style.cursor = '';
    }
}

function toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.body.requestFullscreen().catch(err => {
            console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
        });
    } else {
        document.exitFullscreen();
    }
}

function onKeyDown(event) {
    console.log(`Key down: ${event.key}`);

    // Toggle full screen with 'f' key, but only for the /mea/ endpoint
    if (event.key === 'f' && isMeaPage) {
        event.preventDefault();
        toggleFullScreen();
        return;
    }

    // Reset view with 'r' key
    if (event.key === 'r') {
        event.preventDefault();
        if (focusOnChannels && focusOnChannels.length > 0) {
            applyFocusOnChannels();
        } else {
            zoomOutToDefaultView();
        }
        return;
    }

    // Disable keyboard shortcuts in Jupyter mode (except fullscreen above)
    if (isJupyterMode)
        return;

    // Only handle arrow keys when zoomed in
    if (zoomedElectrode === null && !isFreeCameraMode) {
        return;
    }

    let deltaRow = 0;
    let deltaCol = 0;

    switch (event.key) {
        case 'ArrowUp':
        case 'w':
            deltaRow = -1;
            break;
        case 'ArrowDown':
        case 's':
            deltaRow = 1;
            break;
        case 'ArrowLeft':
        case 'a':
            deltaCol = -1;
            break;
        case 'ArrowRight':
        case 'd':
            deltaCol = 1;
            break;
        default:
            return;
    }

    event.preventDefault();

    // If in free camera mode, find nearest electrode and switch to snapped mode
    if (isFreeCameraMode) {
        // Find closest electrode to current free camera position
        let minDist    = Infinity;
        let closestRow = 0;
        let closestCol = 0;

        const rect = renderer.domElement.getBoundingClientRect();

        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = - ((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);

        camera.updateMatrixWorld();

        for (let col = 0; col < gridSize; col++) {
            for (let row = 0; row < gridSize; row++) {
                const electrode = grid[col][row].electrode;
                const pos = new THREE.Vector3();
                electrode.getWorldPosition(pos);

                const dist = Math.sqrt(
                    Math.pow(pos.x - freeCameraPosition.x, 2) +
                    Math.pow(pos.z - freeCameraPosition.z, 2)
                );

                if (dist < minDist) {
                    minDist    = dist;
                    closestRow = row;
                    closestCol = col;
                }
            }
        }

        console.log(`Switching to snapped mode at row ${closestRow}, col ${closestCol}`);
        console.log(`Free camera position x=${freeCameraPosition.x}, z=${freeCameraPosition.z}`);

        // Snap to closest electrode
        zoomedElectrodeGridPos = { row: closestRow, col: closestCol };
        zoomedElectrode        = grid[closestCol][closestRow].electrode;
        isFreeCameraMode       = false;
        freeCameraPosition     = null;
    }

    navigateToElectrode(deltaRow, deltaCol);
}

function navigateToElectrode(deltaRow, deltaCol, duration = zoomAnimationDurationMs) {
    if (zoomedElectrode === null) {
        const newCameraPos = {
            x: freeCameraPosition.x + zoomLevels.current.pos.x,
            y: zoomLevels.current.pos.y,
            z: freeCameraPosition.z + zoomLevels.current.pos.z
        };

        const newLookAt = {
            x: freeCameraPosition.x + zoomLevels.current.lookAt.x,
            y: zoomLevels.current.lookAt.y,
            z: freeCameraPosition.z + zoomLevels.current.lookAt.z
        };

        animateCameraTo(newCameraPos, newLookAt, duration);
        return;
    }

    const newRow = Math.max(0, Math.min(gridSize - 1, zoomedElectrodeGridPos.row + deltaRow));
    const newCol = Math.max(0, Math.min(gridSize - 1, zoomedElectrodeGridPos.col + deltaCol));

    // Get the new electrode
    const newElectrodeData = grid[newCol][newRow];
    const newElectrode = newElectrodeData.electrode;

    // Get world position of new electrode
    const electrodeWorldPos = new THREE.Vector3();
    newElectrode.getWorldPosition(electrodeWorldPos);

    // Calculate zoom camera position
    const zoomedCameraPos = {
        x: electrodeWorldPos.x + zoomLevels.current.pos.x,
        y: zoomLevels.current.pos.y,
        z: electrodeWorldPos.z + zoomLevels.current.pos.z
    };

    const zoomedLookAt = {
        x: electrodeWorldPos.x + zoomLevels.current.lookAt.x,
        y: zoomLevels.current.lookAt.y,
        z: electrodeWorldPos.z + zoomLevels.current.lookAt.z
    };

    // Update tracked state
    zoomedElectrode = newElectrode;
    zoomedElectrodeGridPos = { row: newRow, col: newCol };

    // Animate to new position
    animateCameraTo(zoomedCameraPos, zoomedLookAt, duration);
}

function zoomOutToDefaultView() {
    zoomedElectrode        = null;
    zoomedElectrodeGridPos = null;
    isFreeCameraMode       = false;
    freeCameraPosition     = null;
    currentZoom            = 0.0;

    animateCameraTo(defaultCameraState.pos, defaultCameraState.lookAt, zoomAnimationDurationMs);
    hideZoomControl();
}

function onRendererRightClick(event) {
    event.preventDefault();

    // Allow input during animation - it will be interrupted if needed
    const rect = renderer.domElement.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    if (zoomedElectrode !== null || isFreeCameraMode) {
        // Already zoomed in (either snapped or free), zoom back out to default view
        zoomOutToDefaultView();
        return false;
    }

    const intersects = raycaster.intersectObjects(scene.children, true);

    if (intersects.length > 0) {
        // Zoom into the selected electrode
        const clickedElectrode = intersects[0].object;

        // Find the electrode's world position
        const electrodeWorldPos = new THREE.Vector3();

        // Traverse up to find the electrode parent object
        let targetObject = clickedElectrode;
        while (targetObject.parent && targetObject.parent.type !== 'Scene') {
            targetObject = targetObject.parent;
        }

        targetObject.getWorldPosition(electrodeWorldPos);

        // Find grid position of clicked electrode
        let foundRow = -1, foundCol = -1;
        outerLoop:
        for (let col = 0; col < gridSize; col++) {
            for (let row = 0; row < gridSize; row++) {
                if (grid[col][row].electrode === targetObject) {
                    foundRow = row;
                    foundCol = col;
                    break outerLoop;
                }
            }
        }

        // Calculate zoom camera position
        const zoomedCameraPos = {
            x: electrodeWorldPos.x + rightClickZoomLevel.pos.x,
            y: rightClickZoomLevel.pos.y,
            z: electrodeWorldPos.z + rightClickZoomLevel.pos.z
        };

        const zoomedLookAt = {
            x: electrodeWorldPos.x + rightClickZoomLevel.lookAt.x,
            y: rightClickZoomLevel.lookAt.y,
            z: electrodeWorldPos.z + rightClickZoomLevel.lookAt.z
        };

        zoomedElectrode        = targetObject;
        zoomedElectrodeGridPos = { row: foundRow, col: foundCol };
        isFreeCameraMode       = false;  // Start in snapped mode
        freeCameraPosition     = null;
        zoomLevels.current     = {
            pos   : { ...rightClickZoomLevel.pos },
            lookAt: { ...rightClickZoomLevel.lookAt }
        }
        currentZoom            = rightClickZoomLevelPercent / 100.0;

        animateCameraTo(zoomedCameraPos, zoomedLookAt, zoomAnimationDurationMs);

        showZoomControl(rightClickZoomLevelPercent);
    }

    return false;
}

function animateCameraTo(targetPosition, targetLookAt, duration = zoomAnimationDurationMs) {
    // Cancel any existing animation
    if (currentAnimationFrameId !== null) {
        cancelAnimationFrame(currentAnimationFrameId);
        currentAnimationFrameId = null;
    }

    // Use current camera position as start (handles mid-animation interruptions)
    const startPosition = {
        x: camera.position.x,
        y: camera.position.y,
        z: camera.position.z
    };

    const startBloomStrength  = bloomRenderer.compositeMaterial.uniforms.bloomStrength.value / devicePixelRatio;
    const targetBloomStrength = interpolate(minBloomStrength, maxBloomStrength, currentZoom);

    const startTiltShiftStrength  = bloomRenderer.tiltShiftMaterialH.uniforms.blurStrength.value / devicePixelRatio;
    const targetTiltShiftStrength = interpolate(minTiltShiftStrength, maxTiltShiftStrength, currentZoom);

    // Get current lookAt by calculating from camera direction
    const currentLookAt = new THREE.Vector3();
    camera.getWorldDirection(currentLookAt);
    currentLookAt.multiplyScalar(10).add(camera.position);

    const startLookAt = {
        x: currentLookAt.x,
        y: currentLookAt.y,
        z: currentLookAt.z
    };

    if (duration <= 0) {
        // Jump to target immediately
        camera.position.set(targetPosition.x, targetPosition.y, targetPosition.z);
        camera.lookAt(targetLookAt.x, targetLookAt.y, targetLookAt.z);

        setBloomStrength(targetBloomStrength);
        setTiltShiftStrength(targetTiltShiftStrength);

        return;
    }

    const startTime = performance.now();

    function animate(currentTime) {
        const elapsed  = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1.0);

        // Easing function (ease-in-out)
        const eased = progress < 0.5
            ? 2 * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 2) / 2;

        // Interpolate camera position
        camera.position.x = startPosition.x + (targetPosition.x - startPosition.x) * eased;
        camera.position.y = startPosition.y + (targetPosition.y - startPosition.y) * eased;
        camera.position.z = startPosition.z + (targetPosition.z - startPosition.z) * eased;

        // Interpolate lookAt target
        const currentLookAtX = startLookAt.x + (targetLookAt.x - startLookAt.x) * eased;
        const currentLookAtY = startLookAt.y + (targetLookAt.y - startLookAt.y) * eased;
        const currentLookAtZ = startLookAt.z + (targetLookAt.z - startLookAt.z) * eased;

        camera.lookAt(currentLookAtX, currentLookAtY, currentLookAtZ);

        // Interpolate bloom strength
        const currentBloomStrength = interpolate(startBloomStrength, targetBloomStrength, eased);
        setBloomStrength(currentBloomStrength);

        // Interpolate tilt-shift strength
        const currentTiltShiftStrength = interpolate(startTiltShiftStrength, targetTiltShiftStrength, eased);
        setTiltShiftStrength(currentTiltShiftStrength);

        // Update the tooltip and outline position if hovering over an electrode
        // Force update during camera animation to keep tooltip accurate
        if (mousePos !== null) {
            updateTooltipAndOutline(mousePos.x, mousePos.y, true);
        }

        if (progress < 1.0) {
            currentAnimationFrameId = requestAnimationFrame(animate);
        } else {
            currentAnimationFrameId = null;
        }
    }

    currentAnimationFrameId = requestAnimationFrame(animate);
}

let currentMouseOverElectrode = null;

function createTooltip() {
    tooltip                       = document.createElement('div');
    tooltip.style.position        = 'absolute';
    tooltip.style.padding         = '4px 8px';
    tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    tooltip.style.color           = 'white';
    tooltip.style.borderRadius    = '4px';
    tooltip.style.fontSize        = '12px';
    tooltip.style.pointerEvents   = 'none';
    tooltip.style.display         = 'none';
    tooltip.style.zIndex          = '1000';
    document.body.appendChild(tooltip);
}

function onRendererMouseOut(event) {
    if (currentMouseOverElectrode) {
        currentMouseOverElectrode.userData.outline.material.visible = false;
        currentMouseOverElectrode.userData.plane.material.visible = false;
        currentMouseOverElectrode = null;
    }

    if (tooltip)
        tooltip.style.display = 'none';
}

function onRendererMouseOver(event) {
    event.preventDefault();

    // Throttle raycasting for better performance (quality-dependent)
    const now = performance.now();
    if (now - lastRaycastTime < raycastThrottleMs) {
        // Store the pending update to process later
        pendingRaycastUpdate = { x: event.clientX, y: event.clientY };

        // Still update tooltip position smoothly even when raycasting is throttled
        if (currentMouseOverElectrode && tooltip) {
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY + 10) + 'px';
        }
        return;
    }

    lastRaycastTime = now;
    pendingRaycastUpdate = null;
    updateTooltipAndOutline(event.clientX, event.clientY);
}

function updateTooltipAndOutline(mouseX, mouseY, forceUpdate = false) {
    const rect = renderer.domElement.getBoundingClientRect();

    mousePos = { x: mouseX, y: mouseY };

    mouse.x = ((mouseX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((mouseY - rect.top) / rect.height) * 2 + 1;

    // Check if mouse is even within the renderer area, as that's cheaper than raycasting
    if (mouse.x < -1 || mouse.x > 1 || mouse.y < -1 || mouse.y > 1) {
        if (currentMouseOverElectrode) {
            currentMouseOverElectrode.userData.outline.material.visible = false;
            currentMouseOverElectrode.userData.plane.material.visible = false;
            currentMouseOverElectrode = null;
        }

        if (tooltip)
            tooltip.style.display = 'none';

        return;
    }

    camera.updateMatrixWorld();  // Ensure matrices are current
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);

    if (intersects.length == 0) {
        if (currentMouseOverElectrode) {
            currentMouseOverElectrode.userData.outline.material.visible = false;
            currentMouseOverElectrode.userData.plane.material.visible = false;
            currentMouseOverElectrode = null;
        }

        if (tooltip)
            tooltip.style.display = 'none';

        return;
    }

    const electrode = intersects[0].object;

    if (currentMouseOverElectrode == electrode) {
        if (tooltip) {
            tooltip.style.left = (mouseX + 10) + 'px';
            tooltip.style.top = (mouseY + 10) + 'px';
        }

        return;
    }

    if (currentMouseOverElectrode) {
        currentMouseOverElectrode.userData.outline.material.visible = false;
        currentMouseOverElectrode.userData.plane.material.visible = false;
    }

    currentMouseOverElectrode = electrode;
    currentMouseOverElectrode.userData.outline.material.visible = true;

    if (tooltip && electrode.userData.channel !== undefined) {
        tooltip.textContent   = `Channel ${electrode.userData.channel}`;
        tooltip.style.display = 'block';
        tooltip.style.left    = (mouseX + 10) + 'px';
        tooltip.style.top     = (mouseY + 10) + 'px';
    }
}

function resetMeaPlanePositions() {
    for (let i = 0; i < measurementPlanes; i++)
        planePositions[i] = (((i + 1) / measurementPlanes) * electrodeDepth) - (halfElectrodeDepth);

    drawPlane = frontPlaneIndex;
    drawPos = 0;
}

function createMea() {
    setupPlaneMaterialCache();

    // Max instances per row = textureWidth * gridSize (all electrodes in the row)
    const maxInstancesPerRow = textureWidth * gridSize;

    // Total segments per row = textureWidth * gridSize (all electrodes)
    const segmentsPerRow = textureWidth * gridSize;
    const verticesPerSegment = 4;

    // Create row-level marker containers and waveform geometries
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        const rowMarkerPlanes = [];
        const rowWaveformPlanes = [];

        // Calculate row's base Z position
        const rowBaseZ = back + (rowIndex * (electrodeDepth + gapSizeZ)) + (halfElectrodeDepth);

        for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
            // === MARKER INSTANCED MESHES (unchanged) ===
            const markerSize = 6.0;
            const markerGeometry = new THREE.PlaneGeometry(markerSize, markerSize);

            const instancedMeshes = {
                spike: new THREE.InstancedMesh(markerGeometry, planeMaterialCache[planeIndex].spike, maxInstancesPerRow),
                stim : new THREE.InstancedMesh(markerGeometry, planeMaterialCache[planeIndex].stim, maxInstancesPerRow)
            };

            instancedMeshes.spike.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 500);
            instancedMeshes.stim.boundingSphere  = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 500);

            instancedMeshes.spike.frustumCulled = true;
            instancedMeshes.stim.frustumCulled  = true;

            const planeBaseOrder = planeIndex * 1000 + rowIndex * 100000;
            const spikeOrder = planeBaseOrder + 100;
            const stimOrder = planeBaseOrder + 200;

            instancedMeshes.spike.renderOrder = spikeOrder;
            instancedMeshes.spike.count = 0;
            instancedMeshes.spike.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

            instancedMeshes.stim.renderOrder = stimOrder;
            instancedMeshes.stim.count = 0;
            instancedMeshes.stim.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

            const markerContainer = new THREE.Object3D();
            markerContainer.add(instancedMeshes.spike);
            markerContainer.add(instancedMeshes.stim);
            scene.add(markerContainer);

            rowMarkerPlanes[planeIndex] = {
                container: markerContainer,
                instancedMeshes
            };

            // === ROW-LEVEL MERGED WAVEFORM GEOMETRY ===
            // Create a single geometry for all electrodes in this row at this plane depth
            const positions          = new Float32Array(segmentsPerRow * verticesPerSegment * 3);
            const colors             = new Float32Array(segmentsPerRow * verticesPerSegment * 3);
            const uvs                = new Float32Array(segmentsPerRow * verticesPerSegment * 2);
            const brightness         = new Float32Array(segmentsPerRow * verticesPerSegment);      // Per-vertex brightness
            const electrodeIdx       = new Float32Array(segmentsPerRow * verticesPerSegment);      // Which electrode (0-7)
            const electrodeU         = new Float32Array(segmentsPerRow * verticesPerSegment);      // U position within electrode (0-1)
            const channelOpacityAttr = new Float32Array(segmentsPerRow * verticesPerSegment);      // Per-electrode opacity
            brightness.fill(1.0);
            channelOpacityAttr.fill(1.0);

            // Pre-populate electrode index and U for each vertex
            for (let colIdx = 0; colIdx < gridSize; colIdx++) {
                for (let seg = 0; seg < textureWidth; seg++) {
                    const segmentIndex = colIdx * textureWidth + seg;
                    const uValue = (seg + 0.5) / textureWidth;  // Center of segment
                    for (let v = 0; v < 4; v++) {
                        const idx = segmentIndex * 4 + v;
                        electrodeIdx[idx] = colIdx;
                        electrodeU[idx] = uValue;
                        channelOpacityAttr[idx] = 1.0;
                    }
                }
            }

            // Create index buffer: 6 indices per quad (2 triangles)
            // Use Uint32Array since we may exceed 65535 vertices (segmentsPerRow * 4 could be > 65535)
            const indices = segmentsPerRow * verticesPerSegment > 65535
                ? new Uint32Array(segmentsPerRow * 6)
                : new Uint16Array(segmentsPerRow * 6);

            for (let i = 0; i < segmentsPerRow; i++) {
                const baseVertex = i * 4;
                const baseIndex = i * 6;
                indices[baseIndex + 0] = baseVertex + 0;
                indices[baseIndex + 1] = baseVertex + 1;
                indices[baseIndex + 2] = baseVertex + 2;
                indices[baseIndex + 3] = baseVertex + 0;
                indices[baseIndex + 4] = baseVertex + 2;
                indices[baseIndex + 5] = baseVertex + 3;
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));

            const positionAttribute = new THREE.BufferAttribute(positions, 3);
            positionAttribute.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute('position', positionAttribute);

            const colorAttribute = new THREE.BufferAttribute(colors, 3);
            colorAttribute.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute('instanceColor', colorAttribute);

            const uvAttribute = new THREE.BufferAttribute(uvs, 2);
            uvAttribute.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute('uv', uvAttribute);

            const brightnessAttribute = new THREE.BufferAttribute(brightness, 1);
            brightnessAttribute.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute('vertexBrightness', brightnessAttribute);

            // Static attributes for electrode identification (don't need dynamic usage)
            geometry.setAttribute('electrodeIndex', new THREE.BufferAttribute(electrodeIdx, 1));
            geometry.setAttribute('electrodeU', new THREE.BufferAttribute(electrodeU, 1));

            // Dynamic attribute for per-electrode opacity (can be updated for focus dimming)
            const channelOpacityAttribute = new THREE.BufferAttribute(channelOpacityAttr, 1);
            channelOpacityAttribute.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute('channelOpacity', channelOpacityAttribute);

            // Clone the shared material for this specific row/plane mesh
            // Each mesh needs its own material instance for independent uniform updates
            const material = sharedWaveformMaterials[planeIndex].clone();

            // Initialize spike data array for this material's uniforms
            material.uniforms.spikeData = {
                value: new Array(maxFlashingSpikes).fill(null).map(() => new THREE.Vector4(0, 0, 0, -1))
            };
            material.uniforms.activeSpikeCount = { value: 0 };

            const mesh = new THREE.Mesh(geometry, material);
            mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 500);
            mesh.frustumCulled = true;
            mesh.renderOrder = planeBaseOrder;

            const waveformContainer = new THREE.Object3D();
            waveformContainer.position.set(0, 0, rowBaseZ);
            waveformContainer.add(mesh);
            scene.add(waveformContainer);

            rowWaveformPlanes[planeIndex] = {
                container: waveformContainer,
                geometry,
                mesh,
                material,  // Store reference to the cloned material
                spikes: [],  // Track spikes for this row/plane
            };
        }

        rowMarkers[rowIndex] = rowMarkerPlanes;
        rowWaveforms[rowIndex] = rowWaveformPlanes;
    }

    // Create electrode grid for raycasting/interaction (simplified - no waveform geometry)
    for (let colIndex = 0; colIndex < gridSize; colIndex++) {
        const column = [];
        grid[colIndex] = column;

        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
            const electrodeGeometry = new THREE.PlaneGeometry(electrodeWidth + gapSizeX, electrodeDepth);
            electrodeGeometry.rotateX(-Math.PI / 2);

            const outlineMaterial = new THREE.MeshBasicMaterial({ color: cellOutlineColourObject });
            const outline = new THREE.LineSegments(new THREE.EdgesGeometry(electrodeGeometry), outlineMaterial);

            const planeMaterial = new THREE.MeshBasicMaterial({
                color      : 0xffffff,
                opacity    : 0.15,
                transparent: true,
                blending   : THREE.AdditiveBlending,
            });
            const plane = new THREE.Mesh(electrodeGeometry, planeMaterial);

            outlineMaterial.visible = false;
            planeMaterial.visible = false;
            plane.layers.set(1);
            plane.depthWrite = false;
            plane.renderOrder = 1000 + rowIndex * 100000;

            plane.userData = {
                channel: rowColChannelLayout[rowIndex][colIndex],
                outline: outline,
                plane  : plane
            };

            const electrode = new THREE.Object3D();
            electrode.add(outline);
            electrode.add(plane);

            const xPos = left + (colIndex * (electrodeWidth + gapSizeX)) + (halfElectrodeWidth);
            const zPos = back + (rowIndex * (electrodeDepth + gapSizeZ)) + (halfElectrodeDepth);

            electrode.position.set(xPos, 0, zPos);
            scene.add(electrode);
            column[rowIndex] = { electrode };
        }
    }

    resetMeaPlanePositions();

    renderer.domElement.style.filter = windowStyle.getPropertyValue('--canvas-3d-filter');

    renderer.domElement.addEventListener('click', onRendererMouseClick, false);
    renderer.domElement.addEventListener('contextmenu', onRendererRightClick, false);
    renderer.domElement.addEventListener('mousemove', onRendererMouseOver, false);
    renderer.domElement.addEventListener('mouseout', onRendererMouseOut, false);
}

function render(timestamp) {
    if (rendererElement.style.display === 'none' || rendererElement.hidden) {
        stopAnimation();
        return;
    }

    const elapsedTime = (timestamp - firstTimestamp) * 0.001;

    // Timing measurements for debug mode
    const renderStartTime = DEBUG_MODE ? performance.now() : 0;
    let shiftEndTime = 0;
    let addEndTime = 0;

    animationRequestId = requestAnimationFrame(render);

    // Track FPS
    if (DEBUG_MODE && lastTimestamp !== undefined) {
        const frameTime = timestamp - lastTimestamp;
        const fps = 1000 / frameTime;
        fpsHistory.push(fps);
        if (fpsHistory.length > maxHistoryLength) {
            fpsHistory.shift();
        }

        // Calculate exponential moving average FPS (using frame time for stability)
        // EMA smoothing factor: lower = more smoothing (0.05 = ~20 frame smoothing)
        const emaAlpha = 0.05;
        const lastAvgFps = fpsAverageHistory.length > 0 ? fpsAverageHistory[fpsAverageHistory.length - 1] : fps;
        const avgFps = lastAvgFps + emaAlpha * (fps - lastAvgFps);
        fpsAverageHistory.push(avgFps);
        if (fpsAverageHistory.length > maxHistoryLength) {
            fpsAverageHistory.shift();
        }
    }

    const analysisMsPerResult = getAnalysisMsPerResult();
    if (!analysisMsPerResult)
        return;

    if (firstTimestamp === undefined) {
        if (isBuffering())
            return;

        firstTimestamp = timestamp;
        lastTimestamp = timestamp;
        return;
    }

    const relativeTimestamp = timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    const depthPerMs = electrodeDepth / (textureWidth * measurementPlanes * analysisMsPerResult)
    let planesCompleted = 0

    // Check for resuming before the pause check to ensure material cache is restored
    if (window.resuming === true) {
        window.resuming = false;
        clearGeometries();
        resetMeaPlanePositions();
        return;
    }

    if (window.paused !== true) {
        for (let i = 0; i < measurementPlanes; i++)
            planePositions[i] -= depthPerMs * relativeTimestamp;

        while (planePositions[0] < 0 - halfElectrodeDepth) {
            let position = planePositions.shift();
            position += electrodeDepth;
            planePositions.push(position);

            planesCompleted++;
            drawPlane--;
        }

        if (drawPlane < 0) {
            continueAnimation();
            return;
        }
    }

    if (DEBUG_MODE) {
        shiftEndTime = performance.now();
    }

    let frontJourneyProgress = (halfElectrodeDepth - planePositions[frontPlaneIndex]) / (electrodeDepth / measurementPlanes);
    if (frontJourneyProgress < 0.0) {
        console.error(`Front measurement progress is out of range at ${frontJourneyProgress}`);
        frontJourneyProgress = 0.0;
    }
    if (frontJourneyProgress > 1.0) {
        console.error(`Front measurement progress is out of range at ${frontJourneyProgress}`);
        frontJourneyProgress = 1.0;
    }

    const drawUpToPos         = Math.round(textureWidth * frontJourneyProgress);
    const maxAnalysisToRender = ((frontPlaneIndex - drawPlane) * textureWidth) + drawUpToPos - drawPos;

    if (DEBUG_MODE)
    {
        entriesHistory.push(maxAnalysisToRender);
        if (entriesHistory.length > maxHistoryLength) {
            entriesHistory.shift();
        }

        bufferHistory.push(getBufferSize());
        if (bufferHistory.length > maxHistoryLength) {
            bufferHistory.shift();
        }
    }

    const analysisToRender = getAnalysisToRender(maxAnalysisToRender);
    const analysisMissing  = maxAnalysisToRender - analysisToRender.length;

    if (DEBUG_MODE)
    {
        // Get buffering state history AFTER fetching analysis to render
        isBufferingHistory.push(isBuffering());
        if (isBufferingHistory.length > maxHistoryLength) {
            isBufferingHistory.shift();
        }
    }

    if (window.paused === true) {
        bloomRenderer.render();
        return;
    }

    const operations = [];
    let amountToDraw = analysisToRender.length + analysisMissing;

    while (amountToDraw > 0 && drawPlane < measurementPlanes) {
        const thisDrawAmount = Math.min(amountToDraw, textureWidth - drawPos);
        operations.push({ drawPlane, drawPos, thisDrawAmount });
        amountToDraw -= thisDrawAmount;
        drawPos += thisDrawAmount;

        if (drawPos >= textureWidth) {
            drawPlane++;
            drawPos = 0;
        }
    }

    // We always use the 'global' camera position, not the zoomed in version so the perspective is consistent.
    // Use cached vectors to avoid allocations per frame
    _cameraPosition.set(cameraPos.x, cameraPos.y, cameraPos.z);
    _lookAtVector.set(lookAt.x, lookAt.y, lookAt.z);
    _cameraVector.copy(_cameraPosition).sub(_lookAtVector).normalize();
    // _cameraFloorVector is already set to (0, 0, 1) at initialization
    const angleInRadians = _cameraVector.angleTo(_cameraFloorVector);

    // Reset row-level marker and waveform plane data for planes that completed
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        for (let i = 0; i < planesCompleted; i++) {
            // Cycle the row marker plane data (shift and push)
            const markerPlane = rowMarkers[rowIndex].shift();
            markerPlane.instancedMeshes.spike.count = 0;
            markerPlane.instancedMeshes.stim.count = 0;
            rowMarkers[rowIndex].push(markerPlane);

            // Cycle the row waveform plane data (shift and push)
            const waveformPlane = rowWaveforms[rowIndex].shift();
            // Clear the geometry positions
            const positions = waveformPlane.geometry.attributes.position.array;
            positions.fill(0);
            waveformPlane.geometry.attributes.position.needsUpdate = true;
            // Reset brightness
            const brightness = waveformPlane.geometry.attributes.vertexBrightness.array;
            brightness.fill(1.0);
            waveformPlane.geometry.attributes.vertexBrightness.needsUpdate = true;
            // Clear spike flash data
            waveformPlane.spikes = [];
            rowWaveforms[rowIndex].push(waveformPlane);
        }
    }

    // Process analysis data into row-level waveform geometries
    let analysisIndex = 0;

    for (const { drawPlane: currentDrawPlane, drawPos: currentDrawPos, thisDrawAmount } of operations) {
        // For each draw position in this operation
        for (let drawOffset = 0; drawOffset < thisDrawAmount; drawOffset++) {
            const drawX = currentDrawPos + drawOffset;
            const analysis = analysisToRender[analysisIndex++];

            if (!analysis) continue;

            // Process each row and column for this time point
            for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
                const rowWaveformPlane = rowWaveforms[rowIndex][currentDrawPlane];
                const rowMarkerPlane = rowMarkers[rowIndex][currentDrawPlane];

                const positions  = rowWaveformPlane.geometry.attributes.position.array;
                const colors     = rowWaveformPlane.geometry.attributes.instanceColor.array;
                const uvs        = rowWaveformPlane.geometry.attributes.uv.array;
                const brightness = rowWaveformPlane.geometry.attributes.vertexBrightness.array;

                const spikeMesh = rowMarkerPlane.instancedMeshes.spike;
                const stimMesh  = rowMarkerPlane.instancedMeshes.stim;

                for (let colIndex = 0; colIndex < gridSize; colIndex++) {
                    const channel = rowColChannelLayout[rowIndex][colIndex];

                    // Skip disconnected corner electrodes (channels 0, 7, 56, 63)
                    if (disconnectedChannels.has(channel)) continue;

                    const electrodeXPos = left + (colIndex * (electrodeWidth + gapSizeX)) + (halfElectrodeWidth);

                    const entry = analysis[channel];

                    // Calculate x position for this data point (world coordinates)
                    const xPos = electrodeXPos + (drawX / textureWidth) * electrodeWidth - halfElectrodeWidth;
                    const xPosNext = electrodeXPos + ((drawX + 1) / textureWidth) * electrodeWidth - halfElectrodeWidth;

                    // Check for spikes/stims and queue marker data
                    if (entry.hasSpike || entry.hasStim) {
                        const mesh = entry.hasStim ? stimMesh : spikeMesh;

                        dummy.position.set(
                            xPos + (0.5 / textureWidth * electrodeWidth),
                            3.0,
                            0
                        );
                        dummy.updateMatrix();
                        mesh.setMatrixAt(mesh.count, dummy.matrix);
                        mesh.count++;

                        if (entry.hasSpike) {
                            channelBrightness[channel] = flashBrightness;

                            // Record spike for localized flash effect
                            // spikeData: x = U position within electrode, y = unused, z = time, w = electrode index
                            const spikeU = (drawX + 0.5) / textureWidth;

                            // Add spike to this row/plane's spike list (FIFO)
                            if (rowWaveformPlane.spikes.length >= maxFlashingSpikes) {
                                rowWaveformPlane.spikes.shift();
                            }
                            rowWaveformPlane.spikes.push({
                                u: spikeU,
                                time: elapsedTime,
                                electrodeIdx: colIndex
                            });
                        }
                    }

                    // Scale to measurement height
                    const yMin = (entry.min * measurementHeight) - (measurementHeight / 2);
                    let yMax = (entry.max * measurementHeight) - (measurementHeight / 2);

                    const minHeight = 0.5;
                    if (Math.abs(yMax - yMin) < minHeight && yMax !== yMin) {
                        const mid = (yMin + yMax) / 4;
                        yMax = mid + minHeight / 4;
                    }

                    const color = noSpikeColourObject;

                    // Calculate buffer indices for merged row geometry
                    const electrodeSegmentOffset = colIndex * textureWidth;
                    const segmentIndex = electrodeSegmentOffset + drawX;
                    const baseIdx = segmentIndex * 12; // 4 vertices * 3 components

                    // Vertex positions (world coordinates)
                    positions[baseIdx + 0] = xPos;      positions[baseIdx + 1] = yMin; positions[baseIdx + 2] = 0;
                    positions[baseIdx + 3] = xPosNext;  positions[baseIdx + 4] = yMin; positions[baseIdx + 5] = 0;
                    positions[baseIdx + 6] = xPosNext;  positions[baseIdx + 7] = yMax; positions[baseIdx + 8] = 0;
                    positions[baseIdx + 9] = xPos;      positions[baseIdx + 10] = yMax; positions[baseIdx + 11] = 0;

                    // Colors
                    for (let v = 0; v < 4; v++) {
                        const colorIdx = baseIdx + v * 3;
                        colors[colorIdx + 0] = color.b;
                        colors[colorIdx + 1] = color.g;
                        colors[colorIdx + 2] = color.r;
                    }

                    // UVs
                    const uvIdx = segmentIndex * 8;
                    const uMin = drawX / textureWidth;
                    const uMax = (drawX + 1) / textureWidth;
                    uvs[uvIdx + 0] = uMin; uvs[uvIdx + 1] = entry.min;
                    uvs[uvIdx + 2] = uMax; uvs[uvIdx + 3] = entry.min;
                    uvs[uvIdx + 4] = uMax; uvs[uvIdx + 5] = entry.max;
                    uvs[uvIdx + 6] = uMin; uvs[uvIdx + 7] = entry.max;

                    // Per-vertex brightness
                    const brightnessIdx = segmentIndex * 4;
                    const channelBright = channelBrightness[channel];
                    brightness[brightnessIdx + 0] = channelBright;
                    brightness[brightnessIdx + 1] = channelBright;
                    brightness[brightnessIdx + 2] = channelBright;
                    brightness[brightnessIdx + 3] = channelBright;
                }

                // Mark attributes as needing update (once per row per operation batch)
                rowWaveformPlane.geometry.attributes.position.needsUpdate = true;
                rowWaveformPlane.geometry.attributes.instanceColor.needsUpdate = true;
                rowWaveformPlane.geometry.attributes.uv.needsUpdate = true;
                rowWaveformPlane.geometry.attributes.vertexBrightness.needsUpdate = true;
            }
        }
    }

    // Decay brightness for all channels and track which need vertex updates
    const channelsNeedingBrightnessUpdate = [];
    for (let channel = 0; channel < 64; channel++) {
        if (channelBrightness[channel] > 1.0) {
            channelsNeedingBrightnessUpdate.push(channel);
            for (let i = 0; i < analysisToRender.length; i++) {
                channelBrightness[channel] *= flashDecayRate;
            }
            if (channelBrightness[channel] < 1.0) {
                channelBrightness[channel] = 1.0;
            }
        }
    }

    // Update brightness attribute for ALL vertices of electrodes with active flashing
    // This ensures existing waveform data also flashes, not just newly drawn data
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        for (let colIndex = 0; colIndex < gridSize; colIndex++) {
            const channel = rowColChannelLayout[rowIndex][colIndex];

            // Skip disconnected corner electrodes (channels 0, 7, 56, 63)
            if (disconnectedChannels.has(channel)) continue;

            // Only update if this channel needs a brightness update
            if (!channelsNeedingBrightnessUpdate.includes(channel)) continue;

            const channelBright = channelBrightness[channel];
            const electrodeSegmentOffset = colIndex * textureWidth;

            // Update brightness for this electrode across ALL planes in this row
            for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
                const waveformPlane = rowWaveforms[rowIndex][planeIndex];
                const brightness = waveformPlane.geometry.attributes.vertexBrightness.array;

                // Update all segments for this electrode
                for (let seg = 0; seg < textureWidth; seg++) {
                    const segmentIndex = electrodeSegmentOffset + seg;
                    const brightnessIdx = segmentIndex * 4;
                    brightness[brightnessIdx + 0] = channelBright;
                    brightness[brightnessIdx + 1] = channelBright;
                    brightness[brightnessIdx + 2] = channelBright;
                    brightness[brightnessIdx + 3] = channelBright;
                }

                waveformPlane.geometry.attributes.vertexBrightness.needsUpdate = true;
            }
        }
    }

    // Update row-level waveform and marker containers
    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        const rowBaseZ = back + (rowIndex * (electrodeDepth + gapSizeZ)) + (halfElectrodeDepth);

        for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
            // Update waveform container position and rotation
            const waveformPlane = rowWaveforms[rowIndex][planeIndex];
            waveformPlane.container.position.z = rowBaseZ + planePositions[planeIndex];
            waveformPlane.container.rotation.x = -angleInRadians;

            const planeBaseOrder = planeIndex * 1000 + rowIndex * 100000;
            waveformPlane.mesh.renderOrder = planeBaseOrder;

            // Calculate opacity and depth factor based on this plane's actual position
            const fadeOutPeriod = 2 * (electrodeDepth / measurementPlanes);
            let currentOpacity;
            if (planePositions[planeIndex] > 0 - halfElectrodeDepth + fadeOutPeriod) {
                currentOpacity = 1.0;
            } else {
                currentOpacity = (planePositions[planeIndex] + halfElectrodeDepth) / fadeOutPeriod;
            }

            const depthFactor = (planePositions[planeIndex] + halfElectrodeDepth) / electrodeDepth;

            // Update THIS mesh's material uniforms (cloned material, not shared)
            waveformPlane.material.uniforms.opacity.value = Math.max(0.0, currentOpacity);
            waveformPlane.material.uniforms.depthFactor.value = Math.max(0.0, Math.min(1.0, depthFactor));
            waveformPlane.material.uniforms.time.value = elapsedTime;

            // Update spike flash uniforms - filter expired spikes and update uniform array
            waveformPlane.spikes = waveformPlane.spikes.filter(spike => (elapsedTime - spike.time) < 2.0);

            const spikeDataArray = waveformPlane.material.uniforms.spikeData.value;
            const activeCount = Math.min(waveformPlane.spikes.length, maxFlashingSpikes);

            for (let i = 0; i < maxFlashingSpikes; i++) {
                if (i < activeCount) {
                    const spike = waveformPlane.spikes[i];
                    spikeDataArray[i].set(spike.u, 0, spike.time, spike.electrodeIdx);
                } else {
                    spikeDataArray[i].set(0, 0, 0, -1);  // Invalid electrode index
                }
            }
            waveformPlane.material.uniforms.activeSpikeCount.value = activeCount;

            // Update marker container
            const markerPlane = rowMarkers[rowIndex][planeIndex];
            const spikeMesh = markerPlane.instancedMeshes.spike;
            const stimMesh = markerPlane.instancedMeshes.stim;

            markerPlane.container.position.z = rowBaseZ + planePositions[planeIndex];
            markerPlane.container.rotation.x = -angleInRadians;

            const spikeOrder = planeBaseOrder + 100;
            const stimOrder = planeBaseOrder + 200;

            spikeMesh.renderOrder = spikeOrder;
            stimMesh.renderOrder = stimOrder;

            // Flag matrices for update and set visibility
            if (spikeMesh.count > 0) {
                spikeMesh.visible = true;
                spikeMesh.instanceMatrix.needsUpdate = true;
            } else {
                spikeMesh.visible = false;
            }

            if (stimMesh.count > 0) {
                stimMesh.visible = true;
                stimMesh.instanceMatrix.needsUpdate = true;
            } else {
                stimMesh.visible = false;
            }
        }
    }

    if (DEBUG_MODE) {
        addEndTime = performance.now();

        // Count total markers on screen (from row-level meshes)
        let totalMarkers = 0;
        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
            for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
                const rowPlane = rowMarkers[rowIndex][planeIndex];
                totalMarkers += rowPlane.instancedMeshes.spike.count;
                totalMarkers += rowPlane.instancedMeshes.stim.count;
            }
        }
        markerCountHistory.push(totalMarkers);
        if (markerCountHistory.length > maxHistoryLength) {
            markerCountHistory.shift();
        }
    }

    // Update marker opacity directly on each row plane
    const markerFadeOutPeriod = 2 * (electrodeDepth / measurementPlanes);
    const markerFadeOutEnd    = 1 * (electrodeDepth / measurementPlanes);

    for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
        for (let planeIndex = 0; planeIndex < measurementPlanes; planeIndex++) {
            const rowPlane = rowMarkers[rowIndex][planeIndex];
            const planePosition = planePositions[planeIndex];

            const currentMarkerOpacity = Math.max(0, planePosition > 0 - halfElectrodeDepth + markerFadeOutPeriod ? 1.0 :
                (planePosition + halfElectrodeDepth - markerFadeOutEnd) / (markerFadeOutPeriod - markerFadeOutEnd));

            const targetOpacity = currentMarkerOpacity;

            // Update the materials attached to this row plane's instanced meshes
            if (Math.abs(rowPlane.instancedMeshes.stim.material.opacity - targetOpacity) > 0.01) {
                rowPlane.instancedMeshes.stim.material.opacity = targetOpacity;
                rowPlane.instancedMeshes.spike.material.opacity = targetOpacity;
            }
        }
    }

    if (DEBUG_MODE) {
        const renderEndTime   = performance.now();
        const totalRenderTime = renderEndTime - renderStartTime;
        const shiftTime       = shiftEndTime - renderStartTime;
        const addTime         = addEndTime - shiftEndTime;
        const remainingTime   = renderEndTime - addEndTime;

        shiftTimeHistory.push(shiftTime);
        if (shiftTimeHistory.length > maxHistoryLength) {
            shiftTimeHistory.shift();
        }

        addTimeHistory.push(addTime);
        if (addTimeHistory.length > maxHistoryLength) {
            addTimeHistory.shift();
        }

        remainingTimeHistory.push(remainingTime);
        if (remainingTimeHistory.length > maxHistoryLength) {
            remainingTimeHistory.shift();
        }
    }

    // Process any pending raycast update that was throttled
    if (pendingRaycastUpdate !== null) {
        const now = performance.now();
        if (now - lastRaycastTime >= raycastThrottleMs) {
            lastRaycastTime = now;
            updateTooltipAndOutline(pendingRaycastUpdate.x, pendingRaycastUpdate.y, true);
            pendingRaycastUpdate = null;
        }
    }

    bloomRenderer.render();

    // Update adaptive quality based on frame time
    if (lastTimestamp !== undefined) {
        const frameTimeMs = timestamp - lastTimestamp;
        updateAdaptiveQuality(frameTimeMs);
    }
}

function doRendererLayout() {
    // Guard against WebGL not being available
    if (!webglAvailable || !renderer || !bloomRenderer || !camera) {
        return;
    }

    const currentBloomStrength = bloomRenderer.compositeMaterial.uniforms.bloomStrength.value / devicePixelRatio;
    const currentTiltShiftStrength = bloomRenderer.tiltShiftMaterialH.uniforms.blurStrength.value / devicePixelRatio;

    devicePixelRatio = window.devicePixelRatio || 1;

    let width = Math.floor(document.body.clientWidth);
    let height = Math.floor(document.body.clientHeight);

    renderer.setPixelRatio(devicePixelRatio);
    renderer.setSize(width, height);
    bloomRenderer.setSize(width, height);
    if (DEBUG_MODE)
        console.log(`Renderer size: ${width} x ${height}, pixel ratio: ${devicePixelRatio}`);

    const aspect = width / height;
    const horizontalFovRadians = cameraBaseFovDeg * (Math.PI / 180);
    // The original intended aspect ratio is 16:9 (i.e. 9/16 = 0.5625), so we adjust the vertical FOV accordingly to preserve horizontal FOV
    const verticalFovRadians = 2 * Math.atan(Math.tan(horizontalFovRadians / 2) / (aspect * 0.5625));
    camera.fov = Math.max(cameraBaseFovDeg, verticalFovRadians * (180 / Math.PI));
    if (DEBUG_MODE)
        console.log(`Camera FOV set to ${camera.fov} degrees for aspect ratio ${aspect}`);

    camera.aspect = aspect;
    camera.updateProjectionMatrix();

    setBloomStrength(currentBloomStrength);
    setTiltShiftStrength(currentTiltShiftStrength);
}

function createRenderer() {
    // Check for WebGL support first
    if (!window.WebGLRenderingContext) {
        console.error('WebGL is not supported by this browser');
        throw new Error('WebGL not supported');
    }

    // Test if WebGL context can actually be created
    const testCanvas = document.createElement('canvas');
    const testGl = testCanvas.getContext('webgl2') || testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
    if (!testGl) {
        console.error('WebGL context could not be created. Hardware acceleration may be disabled.');
        throw new Error('WebGL context creation failed');
    }

    const width  = visualiserContainer?.clientWidth || 1;
    const height = visualiserContainer?.clientHeight || 1;

    let renderer;
    try {
        renderer = new THREE.WebGLRenderer({
            antialias      : false,
            powerPreference: 'high-performance',
            alpha          : true,
            precision      : 'lowp',
            stencil        : false,
            depth          : false,
        });
    } catch (e) {
        console.error('Failed to create WebGL renderer:', e);
        throw e;
    }

    // Check if renderer was created successfully
    if (!renderer || !renderer.getContext()) {
        console.error('WebGL renderer creation failed - no context available');
        if (renderer) renderer.dispose();
        throw new Error('WebGL renderer creation failed: no context available');
    }

    renderer.setSize(width, height, false);
    renderer.setPixelRatio(devicePixelRatio);

    const gl = renderer.getContext();

    // Verify the context is valid and not lost
    if (gl.isContextLost()) {
        console.error('WebGL context was lost immediately after creation');
        renderer.dispose();
        throw new Error('WebGL context was lost');
    }

    const scene = new THREE.Scene();
    scene.sortObjects = true;  // Enable sorting for better transparency rendering
    const camera =
        new THREE.PerspectiveCamera(
            cameraBaseFovDeg,
            width / height,
            0.1,
            1000
        );

    camera.position.set(cameraPos.x, cameraPos.y, cameraPos.z);
    camera.lookAt(lookAt.x, lookAt.y, lookAt.z);
    camera.layers.enable(1);

    const raycaster = new THREE.Raycaster();
    raycaster.layers.enable(1);
    raycaster.layers.set(1);

    // Create bloom renderer
    const bloomRenderer = createBloomRenderer(renderer, scene, camera, width, height);

    rendererElement.appendChild(renderer.domElement);

    return { gl, renderer, scene, camera, raycaster, bloomRenderer };
}

function drawPlot(debugCtx, pixelRatio, plotX, plotY, plotWidth, plotHeight, dataHistory, label, color, options = {}) {
    const {
        showBufferingRegions = false,
        bufferingHistory = null,
        multiLine = false,
        colors = [color],
        labels = [label]
    } = options;

    // Plot background
    debugCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    debugCtx.fillRect(plotX, plotY, plotWidth, plotHeight);

    // Find max value for scaling (for multi-line, check all datasets)
    let maxValue = 1;
    if (multiLine) {
        for (const data of dataHistory) {
            maxValue = Math.max(maxValue, ...data);
        }
    } else {
        maxValue = Math.max(...dataHistory, 1);
    }

    // Fill buffering regions with red background
    if (showBufferingRegions && bufferingHistory && bufferingHistory.length > 1) {
        let inBufferingRegion = false;
        let regionStartIndex = 0;

        for (let i = 0; i < bufferingHistory.length; i++) {
            if (bufferingHistory[i] && !inBufferingRegion) {
                inBufferingRegion = true;
                regionStartIndex = i;
            }
            else if (!bufferingHistory[i] && inBufferingRegion) {
                inBufferingRegion = false;
                const startX = plotX + (regionStartIndex / maxHistoryLength) * plotWidth;
                const endX = plotX + (i / maxHistoryLength) * plotWidth;
                debugCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
                debugCtx.fillRect(startX, plotY, endX - startX, plotHeight);
            }
        }

        if (inBufferingRegion) {
            const startX = plotX + (regionStartIndex / maxHistoryLength) * plotWidth;
            const endX = plotX + plotWidth;
            debugCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            debugCtx.fillRect(startX, plotY, endX - startX, plotHeight);
        }
    }

    // Draw plot lines
    const datasets = multiLine ? dataHistory : [dataHistory];
    const lineColors = multiLine ? colors : [color];

    for (let datasetIndex = 0; datasetIndex < datasets.length; datasetIndex++) {
        const data = datasets[datasetIndex];
        if (data.length > 1) {
            debugCtx.strokeStyle = lineColors[datasetIndex];
            debugCtx.lineWidth = 1 * pixelRatio;
            debugCtx.lineCap = 'bevel';
            debugCtx.lineJoin = 'bevel';
            debugCtx.beginPath();

            for (let i = 0; i < data.length; i++) {
                const plotXPos = plotX + (i / maxHistoryLength) * plotWidth;
                const plotYPos = plotY + plotHeight - (data[i] / maxValue) * plotHeight;

                if (i === 0) {
                    debugCtx.moveTo(plotXPos, plotYPos);
                } else {
                    debugCtx.lineTo(plotXPos, plotYPos);
                }
            }

            debugCtx.stroke();
        }
    }

    // Draw plot axes and labels
    debugCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    debugCtx.lineWidth = 1 * pixelRatio;
    debugCtx.strokeRect(plotX, plotY, plotWidth, plotHeight);

    // Y-axis labels
    debugCtx.fillStyle = 'white';
    debugCtx.font = `${10 * pixelRatio}px monospace`;
    debugCtx.textAlign = 'right';
    debugCtx.fillText(`${Math.round(maxValue)}`, plotX - 5 * pixelRatio, plotY - 7 * pixelRatio);
    debugCtx.fillText('0', plotX - 5 * pixelRatio, plotY + plotHeight - 10 * pixelRatio);

    // Label(s)
    debugCtx.textAlign = 'left';
    if (multiLine) {
        for (let i = 0; i < labels.length; i++) {
            debugCtx.fillStyle = lineColors[i];
            debugCtx.fillText(labels[i], plotX + 5 * pixelRatio, plotY + plotHeight - 15 * pixelRatio - (i * 12 * pixelRatio));
        }
    } else {
        debugCtx.fillStyle = 'white';
        debugCtx.fillText(label, plotX + 5 * pixelRatio, plotY + plotHeight - 15 * pixelRatio);
    }
}

function createBloomRenderer(renderer, scene, camera, width, height) {
    const pixelRatio = renderer.getPixelRatio();

    // Create a separate 2D canvas for text overlay and plot
    const debugCanvas = document.createElement('canvas');

    debugCanvas.style.position      = 'absolute';
    debugCanvas.style.top           = '0';
    debugCanvas.style.left          = '0';
    debugCanvas.style.pointerEvents = 'none';
    debugCanvas.style.zIndex        = '1000';

    debugCanvas.width        = width * pixelRatio;
    debugCanvas.height       = height * pixelRatio;
    debugCanvas.style.width  = `${width}px`;
    debugCanvas.style.height = `${height}px`;
    const debugCtx = debugCanvas.getContext('2d');

    // Append debug canvas after the WebGL canvas (only in debug mode)
    if (DEBUG_MODE) {
        rendererElement.appendChild(debugCanvas);
    }

    const targetType = THREE.UnsignedByteType;

    // Render targets with hiDPI support
    const msaaSamples = currentQualitySettings.msaaSamples ?? 8;  // Default to 8 if not specified
    const renderTarget = new THREE.WebGLRenderTarget(width * pixelRatio, height * pixelRatio, {
        samples  : Math.min(msaaSamples, renderer.capabilities.maxSamples),   // Quality-dependent MSAA
        type     : targetType,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter
    });
    const blurDivisor = currentQualitySettings.blurTargetDivisor;
    const brightTarget = new THREE.WebGLRenderTarget((width / blurDivisor) * pixelRatio, (height / blurDivisor) * pixelRatio, {
        type: targetType
    });
    const blurTarget1 = new THREE.WebGLRenderTarget((width / blurDivisor) * pixelRatio, (height / blurDivisor) * pixelRatio, {
        type: targetType
    });
    const blurTarget2 = new THREE.WebGLRenderTarget((width / blurDivisor) * pixelRatio, (height / blurDivisor) * pixelRatio, {
        type: targetType
    });
    const tiltShiftTarget = new THREE.WebGLRenderTarget(width * pixelRatio, height * pixelRatio, {
        type: targetType
    });
    const tiltShiftTemp = new THREE.WebGLRenderTarget(width * pixelRatio, height * pixelRatio, {
        type: targetType
    });

    // Brightness extraction shader - extract bright RGB regardless of alpha
    const brightPassShader = {
        uniforms: {
            tDiffuse: { value: null },
            luminosityThreshold: { value: bloomThreshold }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform float luminosityThreshold;
            varying vec2 vUv;

            void main() {
                vec4 texel = texture2D(tDiffuse, vUv);
                // Extract bright areas based on RGB luminosity only
                float luminosity = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
                float brightness = smoothstep(luminosityThreshold, luminosityThreshold + 0.1, luminosity);
                // Output bright color with full opacity so bloom can spread
                gl_FragColor = vec4(texel.rgb * brightness, 1.0);
            }
        `
    };

    // Blur shader - preserve RGB, ignore alpha
    const blurShader = {
        uniforms: {
            tDiffuse  : { value: null },
            resolution: { value: new THREE.Vector2(1.0 / ((width / blurDivisor) * pixelRatio), 1.0 / ((height / blurDivisor) * pixelRatio)) },
            direction : { value: new THREE.Vector2(1, 0) }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform vec2 resolution;
            uniform vec2 direction;
            varying vec2 vUv;

            void main() {
                vec2 offset = direction * resolution;
                vec3 color = vec3(0.0);
                color += texture2D(tDiffuse, vUv - offset * 3.0).rgb * 0.015625;
                color += texture2D(tDiffuse, vUv - offset * 2.0).rgb * 0.09375;
                color += texture2D(tDiffuse, vUv - offset).rgb * 0.234375;
                color += texture2D(tDiffuse, vUv).rgb * 0.3125;
                color += texture2D(tDiffuse, vUv + offset).rgb * 0.234375;
                color += texture2D(tDiffuse, vUv + offset * 2.0).rgb * 0.09375;
                color += texture2D(tDiffuse, vUv + offset * 3.0).rgb * 0.015625;
                gl_FragColor = vec4(color, 1.0);
            }
        `
    };

    // Composite shader with alpha override for bloom
    const compositeShader = {
        uniforms: {
            tDiffuse     : { value: null },
            tBloom       : { value: null },
            bloomStrength: { value: minBloomStrength }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D tDiffuse;
            uniform sampler2D tBloom;
            uniform float bloomStrength;
            varying vec2 vUv;

            void main() {
                vec4 original = texture2D(tDiffuse, vUv);
                vec3 bloom = texture2D(tBloom, vUv).rgb;

                // Add bloom to color
                vec3 color = original.rgb + bloom * bloomStrength;

                // Calculate bloom contribution to alpha
                float bloomLuminosity = dot(bloom, vec3(0.299, 0.587, 0.114));
                float bloomAlpha = bloomLuminosity * bloomStrength;

                // Use maximum of original alpha and bloom alpha
                float finalAlpha = max(original.a, bloomAlpha);

                gl_FragColor = vec4(color, finalAlpha);
            }
        `
    };

    // Create full-screen quads for each pass
    const quadGeometry = new THREE.PlaneGeometry(2, 2);

    const brightPassMaterial = new THREE.ShaderMaterial(brightPassShader);
    const brightPassQuad     = new THREE.Mesh(quadGeometry, brightPassMaterial);
    const brightPassScene    = new THREE.Scene();
    brightPassScene.add(brightPassQuad);

    const blurPassMaterial1 = new THREE.ShaderMaterial(blurShader);
    const blurPassQuad1     = new THREE.Mesh(quadGeometry, blurPassMaterial1);
    const blurPassScene1    = new THREE.Scene();
    blurPassScene1.add(blurPassQuad1);

    const blurPassMaterial2 = new THREE.ShaderMaterial({
        ...blurShader,
        uniforms: {
            tDiffuse  : { value: null },
            resolution: { value: new THREE.Vector2(1.0 / ((width / blurDivisor) * pixelRatio), 1.0 / ((height / blurDivisor) * pixelRatio)) },
            direction : { value: new THREE.Vector2(0, 1) }
        }
    });
    const blurPassQuad2  = new THREE.Mesh(quadGeometry, blurPassMaterial2);
    const blurPassScene2 = new THREE.Scene();
    blurPassScene2.add(blurPassQuad2);

    const compositeMaterial = new THREE.ShaderMaterial(compositeShader);
    const compositeQuad     = new THREE.Mesh(quadGeometry, compositeMaterial);
    const compositeScene    = new THREE.Scene();
    compositeScene.add(compositeQuad);

    const tiltShiftMaterialH = new THREE.ShaderMaterial(tiltShiftShader);
    const tiltShiftQuadH     = new THREE.Mesh(quadGeometry, tiltShiftMaterialH);
    const tiltShiftSceneH    = new THREE.Scene();
    tiltShiftSceneH.add(tiltShiftQuadH);

    const tiltShiftMaterialV = new THREE.ShaderMaterial({
        ...tiltShiftShader,
        uniforms: {
            tDiffuse    : { value: null },
            resolution  : { value: new THREE.Vector2(1.0 / (width * pixelRatio), 1.0 / (height * pixelRatio)) },
            focusHeight : { value: tiltShiftFocusHeight },
            focusWidth  : { value: tiltShiftFocusWidth },
            blurStrength: { value: minTiltShiftStrength },
            direction   : { value: new THREE.Vector2(0, 1) }
        }
    });
    const tiltShiftQuadV  = new THREE.Mesh(quadGeometry, tiltShiftMaterialV);
    const tiltShiftSceneV = new THREE.Scene();
    tiltShiftSceneV.add(tiltShiftQuadV);

    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    return {
        render: function() {
            // Store original clear color and alpha
            const originalClearColor = renderer.getClearColor(new THREE.Color());
            const originalClearAlpha = renderer.getClearAlpha();

            const pipelineStart = DEBUG_MODE ? performance.now() : 0;
            let t0, t1, t2, t3, t4, t5, t6, t7, t8;
            let totalDrawCalls = 0;
            let totalTriangles = 0;

            if (DEBUG_MODE) {
                t0 = performance.now();
                renderer.info.reset();
            }

            // 1. Render scene to render target
            renderer.setRenderTarget(renderTarget);
            renderer.setClearColor(0x000000, 0);
            renderer.clear();
            renderer.render(scene, camera);

            if (DEBUG_MODE) {
                t1 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            // Skip bloom passes entirely if bloom is disabled (low quality mode)
            if (!currentQualitySettings.bloomEnabled) {
                // Skip directly to tilt-shift or final output
                if (DEBUG_MODE) {
                    t2 = t1; t3 = t1; t4 = t1; t5 = t1; // Mark bloom passes as skipped
                }

                // Apply tilt-shift if enabled, otherwise render directly to screen
                if (currentQualitySettings.tiltShiftEnabled) {
                    tiltShiftMaterialH.uniforms.tDiffuse.value = renderTarget.texture;
                    renderer.setRenderTarget(tiltShiftTemp);
                    renderer.setClearColor(0x000000, 0);
                    renderer.clear();
                    renderer.render(tiltShiftSceneH, orthoCamera);

                    if (DEBUG_MODE) {
                        t6 = performance.now();
                        totalDrawCalls += renderer.info.render.calls;
                        totalTriangles += renderer.info.render.triangles;
                    }

                    tiltShiftMaterialV.uniforms.tDiffuse.value = tiltShiftTemp.texture;
                    renderer.setRenderTarget(null);
                    renderer.setClearColor(originalClearColor, originalClearAlpha);
                    renderer.clear();
                    renderer.render(tiltShiftSceneV, orthoCamera);

                    if (DEBUG_MODE) {
                        t7 = performance.now();
                        t8 = t7;
                        totalDrawCalls += renderer.info.render.calls;
                        totalTriangles += renderer.info.render.triangles;
                    }
                } else {
                    // No bloom, no tilt-shift - render scene directly to screen
                    if (DEBUG_MODE) {
                        t6 = t5; t7 = t5;
                    }

                    // Use pass-through shader to copy render target to screen
                    tiltShiftMaterialV.uniforms.tDiffuse.value = renderTarget.texture;
                    tiltShiftMaterialV.uniforms.blurStrength.value = 0;
                    renderer.setRenderTarget(null);
                    renderer.setClearColor(originalClearColor, originalClearAlpha);
                    renderer.clear();
                    renderer.render(tiltShiftSceneV, orthoCamera);

                    if (DEBUG_MODE) {
                        t8 = performance.now();
                        totalDrawCalls += renderer.info.render.calls;
                        totalTriangles += renderer.info.render.triangles;
                    }
                }

                // Skip to debug overlay section
                if (DEBUG_MODE) {
                    lastDrawCalls = totalDrawCalls;
                    lastTriangles = totalTriangles;
                    pipelineSceneHistory.push(t8 - t0);  // Total GPU rendering time
                    if (pipelineSceneHistory.length > maxHistoryLength) pipelineSceneHistory.shift();
                }

                // Render debug overlay if enabled
                if (DEBUG_MODE) {
                    this.renderDebugOverlay();
                }
                return;
            }

            // 2. Extract bright areas (bloom enabled path)
            brightPassMaterial.uniforms.tDiffuse.value = renderTarget.texture;
            renderer.setRenderTarget(brightTarget);
            renderer.setClearColor(0x000000, 0);
            renderer.clear();
            renderer.render(brightPassScene, orthoCamera);

            if (DEBUG_MODE) {
                t2 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            // 3. Blur horizontally
            blurPassMaterial1.uniforms.tDiffuse.value = brightTarget.texture;
            renderer.setRenderTarget(blurTarget1);
            renderer.setClearColor(0x000000, 0);
            renderer.clear();
            renderer.render(blurPassScene1, orthoCamera);

            if (DEBUG_MODE) {
                t3 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            // 4. Blur vertically
            blurPassMaterial2.uniforms.tDiffuse.value = blurTarget1.texture;
            renderer.setRenderTarget(blurTarget2);
            renderer.setClearColor(0x000000, 0);
            renderer.clear();
            renderer.render(blurPassScene2, orthoCamera);

            if (DEBUG_MODE) {
                t4 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            // 5. Composite original + bloom
            compositeMaterial.uniforms.tDiffuse.value = renderTarget.texture;
            compositeMaterial.uniforms.tBloom.value = blurTarget2.texture;
            renderer.setRenderTarget(tiltShiftTarget);
            renderer.setClearColor(0x000000, 0);
            renderer.clear();
            renderer.render(compositeScene, orthoCamera);

            if (DEBUG_MODE) {
                t5 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            // 6. Apply tilt-shift blur horizontally (skip if disabled in quality settings)
            if (currentQualitySettings.tiltShiftEnabled) {
                tiltShiftMaterialH.uniforms.tDiffuse.value = tiltShiftTarget.texture;
                renderer.setRenderTarget(tiltShiftTemp);
                renderer.setClearColor(0x000000, 0);
                renderer.clear();
                renderer.render(tiltShiftSceneH, orthoCamera);
            }

            if (DEBUG_MODE) {
                t6 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            // 7. Apply tilt-shift blur vertically (or output composite directly if tilt-shift disabled)
            if (currentQualitySettings.tiltShiftEnabled) {
                tiltShiftMaterialV.uniforms.tDiffuse.value = tiltShiftTemp.texture;
                renderer.setRenderTarget(null);
                renderer.setClearColor(originalClearColor, originalClearAlpha);
                renderer.clear();
                renderer.render(tiltShiftSceneV, orthoCamera);
            } else {
                // Skip tilt-shift entirely - render composite directly to screen
                renderer.setRenderTarget(null);
                renderer.setClearColor(originalClearColor, originalClearAlpha);
                renderer.clear();
                // Use a simple pass-through for the composite result
                tiltShiftMaterialV.uniforms.tDiffuse.value = tiltShiftTarget.texture;
                tiltShiftMaterialV.uniforms.blurStrength.value = 0; // Disable blur in shader
                renderer.render(tiltShiftSceneV, orthoCamera);
            }

            if (DEBUG_MODE) {
                t7 = performance.now();
                totalDrawCalls += renderer.info.render.calls;
                totalTriangles += renderer.info.render.triangles;
            }

            if (DEBUG_MODE) {
                t8 = performance.now();

                // Store accumulated render stats
                lastDrawCalls = totalDrawCalls;
                lastTriangles = totalTriangles;

                // Record total GPU rendering time
                pipelineSceneHistory.push(t8 - t0);
                if (pipelineSceneHistory.length > maxHistoryLength) pipelineSceneHistory.shift();
            }

            // 8. Draw debug text overlay and plots if in debug mode
            if (DEBUG_MODE) {
                this.renderDebugOverlay();
            }
        },

        // Debug overlay rendering extracted to a separate function for reuse
        renderDebugOverlay: function() {
            // Get current pixel ratio from renderer (not the stale closure variable)
            // This ensures the debug overlay scales correctly when DPI changes
            const currentPixelRatio = renderer.getPixelRatio();

            debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);

            const fontSize = 14 * currentPixelRatio;
            debugCtx.font = `${fontSize}px monospace`;
            debugCtx.fillStyle = 'white';
            debugCtx.textAlign = 'left';
            debugCtx.textBaseline = 'top';

            const adaptiveStatus = adaptiveQualityEnabled ? ` (adaptive: ${adaptiveQualityState.currentLevel})` : ' (fixed)';
            const bloomStatus = currentQualitySettings.bloomEnabled ? '' : ' [no bloom]';
            const avgFps = fpsAverageHistory.at(-1) ? Math.round(fpsAverageHistory.at(-1)) : 0;
            const text = `Quality: ${activeQuality}${adaptiveStatus}${bloomStatus}, Requested: ${entriesHistory.at(-1) || 0}, Buffer: ${bufferHistory.at(-1) || 0}${isBufferingHistory.at(-1) ? ' (Buffering...)' : ''}, FPS: ${avgFps}, Markers: ${markerCountHistory.at(-1) || 0}, Draws: ${lastDrawCalls}, Tris: ${lastTriangles}`;
            const padding = 15 * currentPixelRatio;
            const x = padding;
            const y = padding;

            // Background for text
            const metrics = debugCtx.measureText(text);
            const textWidth = metrics.width;
            const textHeight = fontSize * 1.2;
            debugCtx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            debugCtx.fillRect(x, y, textWidth + padding * 2, textHeight);

            // Text
            debugCtx.fillStyle = isBufferingHistory.at(-1) ? 'red' : 'white';
            debugCtx.fillText(text, x + padding, y);

            // Plot dimensions
            const plotWidth = 300 * currentPixelRatio;
            const plotHeight = 80 * currentPixelRatio;
            const plotSpacing = 5 * currentPixelRatio;

            // Draw Buffer Size plot
            const bufferPlotX = 2 * x;
            const bufferPlotY = y + textHeight + padding;
            drawPlot(debugCtx, currentPixelRatio, bufferPlotX, bufferPlotY, plotWidth, plotHeight, bufferHistory, 'Buffer', '#00ff00', {
                showBufferingRegions: true,
                bufferingHistory: isBufferingHistory
            });

            // Draw Last Entries plot
            const entriesPlotX = 2 * x;
            const entriesPlotY = bufferPlotY + plotHeight + plotSpacing;
            drawPlot(debugCtx, currentPixelRatio, entriesPlotX, entriesPlotY, plotWidth, plotHeight, entriesHistory, 'Requested', '#ff9900', {
                showBufferingRegions: true,
                bufferingHistory: isBufferingHistory
            });

            // Draw FPS plot (raw and smoothed average)
            const fpsPlotX = 2 * x;
            const fpsPlotY = entriesPlotY + plotHeight + plotSpacing;
            drawPlot(debugCtx, currentPixelRatio, fpsPlotX, fpsPlotY, plotWidth, plotHeight,
                [fpsHistory, fpsAverageHistory],
                'FPS',
                '#00ffff',
                {
                    multiLine: true,
                    colors: ['#00ffff', '#ffff00'],
                    labels: ['Raw', 'Avg']
                }
            );

            // Draw Marker Count plot
            const markerPlotX = 2 * x;
            const markerPlotY = fpsPlotY + plotHeight + plotSpacing;
            drawPlot(debugCtx, currentPixelRatio, markerPlotX, markerPlotY, plotWidth, plotHeight, markerCountHistory, 'Markers', '#ff00ff');

            // Draw Render Timing plot (multi-line)
            const timingPlotX = 2 * x;
            const timingPlotY = markerPlotY + plotHeight + plotSpacing;
            drawPlot(debugCtx, currentPixelRatio, timingPlotX, timingPlotY, plotWidth, plotHeight,
                [shiftTimeHistory, addTimeHistory, remainingTimeHistory],
                'Timing (ms)',
                '#ffff00',
                {
                    multiLine: true,
                    colors: ['#ff0000', '#00ff00', '#0000ff'],
                    labels: ['Shift', 'Add', 'Remaining']
                }
            );

            // Draw GPU Timing plot (single line - total rendering time)
            const pipelinePlotX = 2 * x;
            const pipelinePlotY = timingPlotY + plotHeight + plotSpacing;
            drawPlot(debugCtx, currentPixelRatio, pipelinePlotX, pipelinePlotY, plotWidth, plotHeight,
                pipelineSceneHistory,
                'GPU (ms)',
                '#ff00ff'
            );
        },
        setSize: function(w, h) {
            const pixelRatio = renderer.getPixelRatio();
            renderTarget.setSize(w * pixelRatio, h * pixelRatio);
            brightTarget.setSize((w / blurDivisor) * pixelRatio, (h / blurDivisor) * pixelRatio);
            blurTarget1.setSize((w / blurDivisor) * pixelRatio, (h / blurDivisor) * pixelRatio);
            blurTarget2.setSize((w / blurDivisor) * pixelRatio, (h / blurDivisor) * pixelRatio);
            tiltShiftTarget.setSize(w * pixelRatio, h * pixelRatio);
            tiltShiftTemp.setSize(w * pixelRatio, h * pixelRatio);
            blurPassMaterial1.uniforms.resolution.value.set(1.0 / ((w / blurDivisor) * pixelRatio), 1.0 / ((h / blurDivisor) * pixelRatio));
            blurPassMaterial2.uniforms.resolution.value.set(1.0 / ((w / blurDivisor) * pixelRatio), 1.0 / ((h / blurDivisor) * pixelRatio));
            tiltShiftMaterialH.uniforms.resolution.value.set(1.0 / (w * pixelRatio), 1.0 / (h * pixelRatio));
            tiltShiftMaterialV.uniforms.resolution.value.set(1.0 / (w * pixelRatio), 1.0 / (h * pixelRatio));

            if (!DEBUG_MODE)
                return;

            debugCanvas.width        = w * pixelRatio;
            debugCanvas.height       = h * pixelRatio;
            debugCanvas.style.width  = `${w}px`;
            debugCanvas.style.height = `${h}px`;
        },
        compositeMaterial : compositeMaterial,
        brightPassMaterial: brightPassMaterial,
        tiltShiftMaterialH: tiltShiftMaterialH,
        tiltShiftMaterialV: tiltShiftMaterialV,
    };
}

function paramToNumber(name, min, max, defaultValue, allowFloat = false) {
    let value = urlParams.get(name);
    if (!value)
        return defaultValue;

    if (allowFloat)
        value = parseFloat(value);
    else
        value = parseInt(value);
    if (isNaN(value))
        return defaultValue;

    if (value < min || value > max)
        return defaultValue;

    return value;
}
