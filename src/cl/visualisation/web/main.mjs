import { connect, setVisibleAbsUv, getVisibleAbsUv, getAnalysisToRender, getAnalysisMsPerResult, stimulate, reset, setDcOffsetCorrection, getDcOffsetCorrection, setHideCrosstalk, getHideCrosstalk } from '/visualiser/analysis.mjs';
import { rowColChannelLayout } from '/visualiser/layout.mjs';

connect(); // Start analysis module connection

//
// Set up page event handlers
//

const minVisibleRangeAbs = 20;
const maxVisibleRangeAbs = 6400;

// Get URL parameters, supporting iframe embedding
let searchParams;
try {
    searchParams = window.parent?.location.search ?? window.location.search;

    // Merge with own search params to allow overrides
    const parentParams = new URLSearchParams(searchParams);
    const ownParams = new URLSearchParams(window.location.search);
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

// Check if we're in Jupyter mode via the query parameter.
// In Jupyter mode, user controls are hidden, click to stim is disabled, 2D mode is forced, keyboard shortcuts are disabled, and the visible range is fixed to 100uV.
// None of these updates the local storage, so the main web console UI is unaffected.
const isJupyterMode = urlParams.get('jupyterMode') === '1';

const isSideBarMode = urlParams.get('sidebarMode') === '1';

const focusOnChannels = urlParams.get('focusOnChannels')?.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

document.onkeydown =
    event =>
    {
        // Disable keyboard shortcuts in Jupyter mode
        if (isJupyterMode)
            return;

        // Check if the pressed key is the spacebar
        if (event.code === 'Space')
        {
            const target = event.target;
            const isInteractiveElement =
                target.isContentEditable ||
                ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(target.tagName) ||
                target.hasAttribute('tabindex');

            if (isInteractiveElement)
                return;

            // Prevent default page scrolling
            event.preventDefault();

            // Pause / resume the rendering
            pause_toggle();
        }
    };

document.querySelector("#pause_toggle").onclick     = () => pause_toggle();
document.querySelector("#reset").onclick            = () => reset();
document.querySelector("#stimulate").onclick        = () => stimulate(selectedChannel);
document.querySelector("#enter-dark-mode").onclick  = () => setTheme('dark');
document.querySelector("#enter-light-mode").onclick = () => setTheme('light');

// Make paused states globally visible so that the 3D renderer can see them
window.paused   = false;
window.resuming = false;
const pause_toggle_button = document.querySelector("#pause_toggle");

let visibleAbsUv = 50.0;

function pause_toggle()
{
    window.paused = !window.paused;

    if (window.paused)
    {
        pause_toggle_button.innerHTML = 'Resume';
        window.paused   = true;
        window.resuming = false;
    }
    else
    {
        pause_toggle_button.innerHTML = 'Pause';
        window.paused   = false;
        window.resuming = true;
    }
}

let selectedChannel;

function selectChannel(channel)
{
    selectedChannel = channel;
    document.querySelector("#stimulate").innerHTML = 'Stimulate Channel ' + selectedChannel;

    // Disable click-to-stimulate in Jupyter mode
    if (!isJupyterMode && document.querySelector("#click-to-stimulate").checked && !window.paused)
        stimulate(selectedChannel);
};

// selectChannel(27);
selectedChannel = 27;

function setupVisibleRangeControl()
{
    const uiControls          = document.querySelector('#visibleRangeDiv');
    const uiVisibleRangeIn    = document.querySelector('#visible_range_in');
    const uiVisibleRangeValue = document.querySelector('#visible_range_value');
    const uiVisibleRangeInput = document.querySelector('#visible_range_input');
    const uiVisibleRangeForm  = document.querySelector('#visible_range_form');

    if (isJupyterMode)
    {
        setVisibleAbsUv(100.0);
        return;
    }

    const minLog = Math.log(minVisibleRangeAbs);
    const maxLog = Math.log(maxVisibleRangeAbs);
    const scale  = (maxLog - minLog) / (maxVisibleRangeAbs - minVisibleRangeAbs);

    uiVisibleRangeIn.min = minVisibleRangeAbs;
    uiVisibleRangeIn.max = maxVisibleRangeAbs;

    function updateUi()
    {
        visibleAbsUv = getVisibleAbsUv();
        uiVisibleRangeIn.value = (Math.log(visibleAbsUv) - minLog) / scale + minVisibleRangeAbs;
        uiVisibleRangeValue.textContent = visibleAbsUv.toFixed(2);
        uiVisibleRangeInput.value = visibleAbsUv.toFixed(2);
    }

    function setRangeFromUiSlider()
    {
        const uV = Math.exp(minLog + scale * (uiVisibleRangeIn.value - minVisibleRangeAbs));
        // console.log("From UI: " + uV);
        setVisibleAbsUv(uV);
        updateUi();
    }

    function setRangeFromUiForm() {
        const uV = parseFloat(uiVisibleRangeInput.value);
        if (!isNaN(uV)) {
            setVisibleAbsUv(uV);
            updateUi();
        }
    }

    function setRangeFromStorage()
    {
        uiVisibleRangeIn.value          = (Math.log(getVisibleAbsUv()) - minLog) / scale + minVisibleRangeAbs;
        // console.log("From Storage: " + getVisibleAbsUv());
        updateUi();
    }

    uiVisibleRangeIn.addEventListener('input', setRangeFromUiSlider, false);

    window.addEventListener(
        'pageshow',
        (event) =>
        {
            setRangeFromStorage();

            uiControls.style.removeProperty('display');
        });

    window.addEventListener(
        'storage',
        (event) =>
        {
            // Sync the whole UI if any other tab changes the local storage
            if (event.storageArea === localStorage)
            {
                setRangeFromStorage();
            }
        });

    uiVisibleRangeForm.addEventListener('submit', (event) => {
        event.preventDefault();
        setRangeFromUiForm();
        });
}

function setupDcOffsetCorrectionControl()
{
    const uiDcOffsetCorrection = document.querySelector('#dc-offset-correction');

    function updateUi()
    {
        uiDcOffsetCorrection.checked = getDcOffsetCorrection();
    }

    function setFromUi()
    {
        const isEnabled = uiDcOffsetCorrection.checked;
        setDcOffsetCorrection(isEnabled);
        updateUi();
    }

    function setFromStorage()
    {
        const isEnabled = getDcOffsetCorrection();
        updateUi();
    }

    uiDcOffsetCorrection.onchange =
        () =>
        {
            setFromUi();
        };

    window.addEventListener(
        'pageshow',
        (event) =>
        {
            setFromStorage();
        });

    window.addEventListener(
        'storage',
        (event) =>
        {
            // Sync the whole UI if any other tab changes the local storage
            if (event.storageArea === localStorage)
            {
                setFromStorage();
            }
        });
}

function setupHideCrosstalkControl()
{
    // const uiHideCrosstalk = document.querySelector('#hide-crosstalk');

    // function updateUi()
    // {
    //     uiHideCrosstalk.checked = getHideCrosstalk();
    // }

    // function setFromUi()
    // {
    //     const isEnabled = uiHideCrosstalk.checked;
    //     setHideCrosstalk(isEnabled);
    //     updateUi();
    // }

    // function setFromStorage()
    // {
    //     const isEnabled = getHideCrosstalk(); // Yuck - get updates from local storage
    //     updateUi();
    // }

    // uiHideCrosstalk.onchange =
    //     () =>
    //     {
    //         setFromUi();
    //     };

    // window.addEventListener(
    //     'pageshow',
    //     (event) =>
    //     {
    //         setFromStorage();
    //     });

    // window.addEventListener(
    //     'storage',
    //     (event) =>
    //     {
    //         // Sync the whole UI if any other tab changes the local storage
    //         if (event.storageArea === localStorage)
    //         {
    //             setFromStorage();
    //         }
    //     });

    // Disable crosstalk by default for now
    setHideCrosstalk(true);
}

function setupClickToStimulateControl()
{
    const uiClickToStimulate = document.querySelector('#click-to-stimulate');

    function updateUi()
    {
        uiClickToStimulate.checked = localStorage.getItem('click-to-stimulate') === 'true';
    }

    function setFromUi()
    {
        const isEnabled = uiClickToStimulate.checked;
        localStorage.setItem('click-to-stimulate', isEnabled ? 'true' : 'false');
        updateUi();
    }

    function setFromStorage()
    {
        const isEnabled = localStorage.getItem('click-to-stimulate') === 'true';
        updateUi();
    }

    uiClickToStimulate.onchange =
        () =>
        {
            setFromUi();
        };

    window.addEventListener(
        'pageshow',
        (event) =>
        {
            setFromStorage();
        });

    window.addEventListener(
        'storage',
        (event) =>
        {
            // Sync the whole UI if any other tab changes the local storage
            if (event.storageArea === localStorage)
            {
                setFromStorage();
            }
        });
}

function setupPlotModeControl()
{
    const uiPlotMode = document.querySelector('#plot-mode');

    // In Jupyter mode, force 2D mode
    if (isJupyterMode)
    {
        return;
    }

    function updateUi()
    {
        uiPlotMode.checked = localStorage.getItem('plot-mode') === '3d';

        // Toggle the 2D/3D display depending on the setting
        if (uiPlotMode.checked)
        {
            meaGrid.style.display  = 'none';
            meaGrid.hidden = true;
            renderer.style.display = 'flex';
            renderer.hidden = false;

            // Stop 2D animation and start 3D
            meaGrid.dispatchEvent(new Event('visibilitychange'));
            renderer.dispatchEvent(new Event('visibilitychange'));
        }
        else
        {
            renderer.style.display = 'none';
            renderer.hidden = true;
            meaGrid.style.display  = 'grid';
            meaGrid.hidden = false;

            // Clear 3D textures, stop 3D animation and start 2D
            renderer.dispatchEvent(new Event('visibilitychange'));
            meaGrid.dispatchEvent(new Event('visibilitychange'));
        }
    }

    function setFromUi()
    {
        const isEnabled = uiPlotMode.checked;
        localStorage.setItem('plot-mode', isEnabled ? '3d' : '2d');
        updateUi();
    }

    function setFromStorage()
    {
        updateUi();
    }

    uiPlotMode.onchange =
        () =>
        {
            setFromUi();
        };

    window.addEventListener(
        'pageshow',
        (event) =>
        {
            setFromStorage();
        });

    // We don't currently sync plot mode between tabs, as it would be disruptive
    // window.addEventListener(
    //     'storage',
    //     (event) =>
    //     {
    //         // Sync the whole UI if any other tab changes the local storage
    //         if (event.storageArea === localStorage)
    //         {
    //             setFromStorage();
    //         }
    //     });
}

setupVisibleRangeControl();
setupDcOffsetCorrectionControl();
setupHideCrosstalkControl();
setupClickToStimulateControl();
setupPlotModeControl();

//
// Rendering
//

function parseRgb(colorString)
{
    if (!colorString) return null;

    // Create a temporary element to resolve the color
    const d = document.createElement("div");
    d.style.color = colorString;
    document.body.appendChild(d);
    const computedColor = getComputedStyle(d).color;
    document.body.removeChild(d);

    const match = computedColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (match)
    {
        return {
            r: parseInt(match[1], 10),
            g: parseInt(match[2], 10),
            b: parseInt(match[3], 10)
        };
    }
    return null;
}

function setTheme(mode)
{
    document.querySelector("html").setAttribute("data-theme", mode);

    const oldMinMaxColour = minMaxColour;
    const oldSpikeColour  = spikeColour;
    const oldStimColour   = stimColour;

    updateColoursFromTheme();

    localStorage.setItem('theme', mode);

    const oldMinMaxRgb = parseRgb(oldMinMaxColour);
    const oldSpikeRgb  = parseRgb(oldSpikeColour);
    const oldStimRgb   = parseRgb(oldStimColour);
    const newMinMaxRgb = parseRgb(minMaxColour);
    const newSpikeRgb  = parseRgb(spikeColour);
    const newStimRgb   = parseRgb(stimColour);

    if (!oldMinMaxRgb && !oldStimRgb && !oldSpikeRgb) return;

    for (let i = 0; i < contextAll.length; i++)
    {
        const context = contextAll[i];
        const canvas = canvasAll[i];

        if (canvas.width === 0 || canvas.height === 0) continue;

        const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        for (let p = 0; p < data.length; p += 4)
        {
            const r = data[p];
            const g = data[p + 1];
            const b = data[p + 2];
            const a = data[p + 3];

            if (a === 0) continue;

            // Heuristic to find pixels matching a color, accounting for anti-aliasing
            // This works by checking which color channels are active.
            const isMinMax = oldMinMaxRgb && (r > 0 === oldMinMaxRgb.r > 0) && (g > 0 === oldMinMaxRgb.g > 0) && (b > 0 === oldMinMaxRgb.b > 0);
            const isStim = oldStimRgb && (r > 0 === oldStimRgb.r > 0) && (g > 0 === oldStimRgb.g > 0) && (b > 0 === oldStimRgb.b > 0);
            const isSpike = oldSpikeRgb && (r > 0 === oldSpikeRgb.r > 0) && (g > 0 === oldSpikeRgb.g > 0) && (b > 0 === oldSpikeRgb.b > 0);

            if (isMinMax)
            {
                data[p]     = newMinMaxRgb.r;
                data[p + 1] = newMinMaxRgb.g;
                data[p + 2] = newMinMaxRgb.b;
            }
            else if (isSpike)
            {
                data[p]     = newSpikeRgb.r;
                data[p + 1] = newSpikeRgb.g;
                data[p + 2] = newSpikeRgb.b;
            }
            else if (isStim)
            {
                data[p]     = newStimRgb.r;
                data[p + 1] = newStimRgb.g;
                data[p + 2] = newStimRgb.b;
            }
        }
        context.putImageData(imageData, 0, 0);
    }
}

let minMaxColour;
let spikeColour;
let stimColour;

// Setup colour change on prefers-color-scheme change
let isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
mediaQuery.addEventListener('change', (e) => {
    isDarkMode = e.matches;
    updateColoursFromTheme();
    window.resuming = true;
});

function updateColoursFromTheme()
{
    const style = getComputedStyle(document.documentElement);

    minMaxColour  = style.getPropertyValue('--electrode-min-max-colour');
    spikeColour   = style.getPropertyValue(isDarkMode ? '--electrode-spike-indicator-colour' : '--electrode-spike-fill-colour');
    stimColour    = style.getPropertyValue(isDarkMode ? '--electrode-stim-colour' : '--electrode-stim-fill-colour');
}

if (localStorage.getItem('theme'))
    setTheme(localStorage.getItem('theme'));
else
    updateColoursFromTheme();

const rowCount      = rowColChannelLayout.length;
const colCount      = rowColChannelLayout[0].length;
const channelCount  = rowCount * colCount;

// Create unwrapped channel -> canvas index
const channelCanvasLayout = new Array(64);

// Create canvas index -> channel
const canvasIndexToChannel = new Array(64);

for (let row = 0; row < rowCount; row++)
    for (let col = 0; col < colCount; col++)
    {
        if (rowColChannelLayout[row].length != colCount)
            throw `row ${row} has colCount of ${rowColChannelLayout[row].length}, expecting ${colCount}`;

        const channel       = rowColChannelLayout[row][col];
        const channelIndex  = (row * rowCount) + col;

        if (channelCanvasLayout[channel] !== undefined)
            throw `duplicate layout data at row ${row} col ${col}`;
        else
        {
            channelCanvasLayout[channel]        = channelIndex;
            canvasIndexToChannel[channelIndex]  = channel;
        }
    }

const canvasAll     = [];
const contextAll    = [];
const meaGrid       = document.getElementById('mea-grid');
const renderer      = document.getElementById('renderer');

if (isJupyterMode) {
    renderer.style.display = 'none';
    renderer.hidden = true;
    meaGrid.style.display = 'grid';
    meaGrid.hidden = false;
    renderer.dispatchEvent(new Event('visibilitychange'));
    meaGrid.dispatchEvent(new Event('visibilitychange'));

    if (isSideBarMode) {
        // In sidebar mode, the external border acts as the outer border, so remove internal padding
        meaGrid.style.padding = '0';
    }
}

for (let i = 0; i < channelCount; i++)
{
    const canvas = document.createElement('canvas');
    canvas.id = 'electrode_' + i;

    // 'willReadFrequently' keeps it out of GPU memory, needed for performance
    // when drawing with putImageData on Chome.
    const context   = canvas.getContext('2d', { willReadFrequently: true });

    // We're rendering pixel perfect
    context.imageSmoothingEnabled = false;

    canvasAll.push(canvas);
    contextAll.push(context);

    canvas.onclick = selectChannel.bind(undefined, canvasIndexToChannel[i]);

    const div = document.createElement('div');
    div.dataset.tooltip = 'Channel ' + canvasIndexToChannel[i];
    div.appendChild(canvas);
    meaGrid.appendChild(div);
}

let firstTimestamp;
let lastTimestamp;

let animationRequestId;

// Start the animation loop
continueAnimation();

meaGrid.addEventListener(
    'visibilitychange',
    function()
    {
        if (meaGrid.style.display === 'none' || meaGrid.hidden)
        {
            console.log("Stopping 2D animation");
            stopAnimation();
        }
        else
        {
            console.log("Starting 2D animation");
            window.resuming = true;
            continueAnimation();
        }
    }
);

function continueAnimation()
{
    stopAnimation();

    animationRequestId = requestAnimationFrame(render);
}

function stopAnimation()
{
    if (!animationRequestId)
        return;

    cancelAnimationFrame(animationRequestId);
    animationRequestId = undefined;
}

let canvasWidth   = undefined;
let canvasHeight  = undefined;
let resized       = true;

//
// Wake Lock Management
//

let wakeLock = null;

async function requestWakeLock() {
    if ('wakeLock' in navigator) {
        try {
            wakeLock = await navigator.wakeLock.request('screen');
            console.log('Wake lock acquired');

            wakeLock.addEventListener('release', () => {
                console.log('Wake lock released');
            });
        } catch (err) {
            console.error(`Failed to acquire wake lock: ${err.name}, ${err.message}`);
        }
    } else {
        console.warn('Wake Lock API not supported in this browser');
    }
}

async function releaseWakeLock() {
    if (wakeLock !== null) {
        try {
            await wakeLock.release();
            wakeLock = null;
        } catch (err) {
            console.error(`Failed to release wake lock: ${err.name}, ${err.message}`);
        }
    }
}

// Request wake lock when page becomes visible
document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState === 'visible') {
        await requestWakeLock();
    } else {
        await releaseWakeLock();
    }
});

// Initial wake lock request when the page loads
requestWakeLock();

function doLayout()
{
    resized = true;
    const mainWidth  = Math.floor(document.body.clientWidth);
    const mainHeight =
        Math.floor(
            document.body.clientHeight -
            document.querySelector('header').clientHeight
        );

    const widthBasedOnHeight = Math.floor(mainHeight / 9 * 16);

    let height;
    let width;
    if (mainWidth >= widthBasedOnHeight)
    {
        width  = widthBasedOnHeight + 'px';
        height = Math.floor(mainHeight) + 'px';
    }
    else
    {
        width  = mainWidth + 'px';
        height = Math.floor(mainWidth / 16 * 9) + 'px';
    }


    meaGrid.style.width   = width;
    meaGrid.style.height  = height;
}
doLayout();
window.addEventListener('resize', doLayout);

function render(timestamp)
{
    if (meaGrid.style.display === 'none' || meaGrid.hidden)
        return;

    // Continue rendering
    animationRequestId = requestAnimationFrame(render);

    const analysisMsPerResult = getAnalysisMsPerResult();
    if (!analysisMsPerResult)
        return;

    if (firstTimestamp === undefined)
    {
        firstTimestamp = timestamp;
        lastTimestamp  = timestamp
        return;
    }

    // Determine how many pixels to shift the rendering by, based on the time delta
    const pixelDelta      = Math.floor((timestamp - lastTimestamp) / analysisMsPerResult);
    const pixelDeltaMs    = pixelDelta * analysisMsPerResult;
    lastTimestamp        += pixelDeltaMs;

    const analysisToRender  = getAnalysisToRender(pixelDelta);

    for (let channel = 0; channel < channelCount; channel++)
    {
        const canvasIndex   = channelCanvasLayout[channel];
        const canvas        = canvasAll[canvasIndex];
        const context       = contextAll[canvasIndex];

        if (resized)
        {
            //
            // The canvases parent divs 'should' all have the same size,
            // and getting the dimensions is expensive in Safari.
            //

            canvasWidth     = Math.ceil(canvas.parentElement.getBoundingClientRect().width);
            canvasHeight    = Math.ceil(canvas.parentElement.getBoundingClientRect().height);
            resized         = false;
        }

        if (window.resuming)
        {
            // Clear the previous data when resuming
            context.clearRect(0, 0, canvasWidth, canvasHeight);
        }

        if (window.paused)
            continue;

        if (canvas.width != canvasWidth || canvas.height != canvasHeight)
        {
            canvas.width = canvasWidth;
            canvas.height = canvasHeight;

            context.clearRect(0, 0, canvasWidth, canvasHeight);
        }

        // Scroll to left
        const imageData = context.getImageData(analysisToRender.length, 0, canvasWidth - analysisToRender.length, canvasHeight);
        context.clearRect(canvasWidth - analysisToRender.length, 0, analysisToRender.length, canvasHeight);
        context.putImageData(imageData, 0, 0);

        // Prepare for drawing vertical min max lines with fills
        context.fillStyle   = minMaxColour;
        context.strokeStyle = spikeColour;

        const alpha = (focusOnChannels && !focusOnChannels.includes(channel)) ? 0.35 : 1.0;
        context.globalAlpha = alpha;

        for (let i = 0; i < analysisToRender.length; i++)
        {
            const drawX = canvasWidth - analysisToRender.length + i;
            const entry = analysisToRender[i][channel];

            if (!entry.outOfRange || entry.hasStim)
            {
                // Scale to canvas height, and invert
                let renderMin     = canvasHeight - 1 - Math.floor(entry.min * canvasHeight);
                let renderMax     = canvasHeight - 1 - Math.ceil(entry.max  * canvasHeight);

                if (entry.hasStim)
                {
                    context.fillStyle = stimColour;
                    context.globalAlpha = 1.0;
                    // Simulate stim artefact by drawing a full height line
                    renderMin = 0;
                    renderMax = canvasHeight;
                }

                // Draw min->max line
                context.fillRect(
                    drawX,
                    renderMin,
                    1,                      // width
                    renderMax - renderMin   // height
                );

                if (entry.hasStim) {
                    context.fillStyle = minMaxColour;
                    context.globalAlpha = alpha;
                }
            }

            // Highlight detected spikes
            if (entry.hasSpike)
            {
                context.globalAlpha = 1.0;
                context.beginPath();
                context.moveTo(drawX, canvasHeight - 1 - 6);
                context.lineTo(drawX, canvasHeight);
                context.lineTo(drawX - 2, canvasHeight);
                context.lineTo(drawX, canvasHeight - 1 - 6);
                context.stroke();

                // Flash the background
                canvas.classList.remove("spike-flash");
                requestAnimationFrame(() => canvas.classList.add("spike-flash"));
                context.globalAlpha = alpha;
            }
        }
    }

    if (window.resuming)
        window.resuming = false;
}