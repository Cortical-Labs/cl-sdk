const defaults =
    {
        'visibleAbsUv':          100,
        'dcOffsetCorrection':    true,
        'hideCrosstalk':         true
    };

const channelCount              = 64;

const initialConnectIntervalMs  = 500;
const maxConnectIntervalMs      = 1000;

const targetBufferSizeMs        = 100;
const maxBufferSizeMs           = 250;

const analysis                  = [];

const originalChannelMean       = new Float32Array(channelCount);
const channelMean               = new Float32Array(channelCount);
const channelStdDev             = new Float32Array(channelCount);

const previousChannelMin        = new Int16Array(channelCount);
const previousChannelMax        = new Int16Array(channelCount);

const uvPerRawAdc               = 0.195;

const FLAG_HAS_SPIKE            = 1 << 0;
const FLAG_HAS_STIM             = 1 << 1;

let ws;
let connectInterval     = initialConnectIntervalMs;
let scheduledReconnect;

// Cached WebSocket URL from /_/config endpoint (for mock server support)
let cachedWsUrl         = null;

let visibleAbsRaw;
let isDcOffsetCorrectionEnabled;
let isHideCrosstalkEnabled;

const stimHideChunks                = 2;
let replaceWithMeanCount            = 0;

const shouldHideCrosstalk           = new Uint32Array(channelCount);
const shouldInjectStimFlag          = new Uint32Array(channelCount);

// Initialise from local storage
getVisibleAbsUv();
getDcOffsetCorrection();
getHideCrosstalk();

// Set during a reset message, lets us know how many ms are processed
// to produce each analysis result.
let analysisMs = 0;

export function getAnalysisMsPerResult()
{
    return analysisMs;
}

export function stimulate(channel)
{
    ws.send(
        JSON.stringify(
            {
                "action": "stimulate",
                "channel": channel
            }));
}

export function reset()
{
    ws.send(
        JSON.stringify(
            {
                "action": "reset"
            }));
}

export function setVisibleAbsUv(uV)
{
    localStorage.setItem('visibleRangeAbsUv', uV);
    visibleAbsRaw = uV / uvPerRawAdc;
}

export function getVisibleAbsUv()
{
    visibleAbsRaw = (parseFloat(localStorage.getItem('visibleRangeAbsUv')) || defaults.visibleAbsUv) / uvPerRawAdc
    return visibleAbsRaw * uvPerRawAdc;
}

export function setDcOffsetCorrection(enable)
{
    if (enable === isDcOffsetCorrectionEnabled)
        return;

    isDcOffsetCorrectionEnabled = enable;
    localStorage.setItem('isDcOffsetCorrectionEnabled', enable ? 'true' : 'false');
}

export function getDcOffsetCorrection()
{
    const settingValue = localStorage.getItem('isDcOffsetCorrectionEnabled');

    if (settingValue === 'true')
        isDcOffsetCorrectionEnabled = true;
    else if (settingValue === 'false')
        isDcOffsetCorrectionEnabled = false;
    else
        // If not set, use the default
        isDcOffsetCorrectionEnabled = defaults.dcOffsetCorrection;

    return isDcOffsetCorrectionEnabled;
}

export function setHideCrosstalk(enable)
{
    if (enable === isHideCrosstalkEnabled)
        return;

    isHideCrosstalkEnabled = enable;
    localStorage.setItem('isHideCrosstalkEnabled', enable ? 'true' : 'false');
}

export function getHideCrosstalk()
{
    const settingValue = localStorage.getItem('isHideCrosstalkEnabled');

    if (settingValue === 'true')
        isHideCrosstalkEnabled = true;
    else if (settingValue === 'false')
        isHideCrosstalkEnabled = false;
    else
        // If not set, use the default
        isHideCrosstalkEnabled = defaults.hideCrosstalk;

    return isHideCrosstalkEnabled;
}

export function connect()
{
    console.log('connect: ' + document.visibilityState);

    if (document.visibilityState != 'visible')
        return;

    if (ws)
    {
        if (ws.readyState == WebSocket.CONNECTING || ws.readyState == WebSocket.OPEN)
            return;

        ws.close();
        ws = undefined;
    }

    function scheduleReconnect()
    {
        console.log('Scheduling reconnect in ' + connectInterval + ' ms');

        if (scheduledReconnect)
        {
            clearTimeout(scheduledReconnect);
            scheduledReconnect = undefined;
        }

        scheduledReconnect =
            setTimeout(
                () =>
                {
                    disconnect();
                    connect();
                },
                connectInterval);

        // Increment the interval in case reconnecting fails
        connectInterval *= 1.25;
        if (connectInterval > maxConnectIntervalMs)
            connectInterval = maxConnectIntervalMs;
    }

    // Handle timeout ourselves
    scheduleReconnect();

    // Try to fetch WebSocket URL from /_/config endpoint first (for mock server support)
    // This allows the MEA page served from a different port to correctly connect to the WebSocket
    async function getWebSocketUrl()
    {
        // Use cached URL if available
        if (cachedWsUrl)
            return cachedWsUrl;

        try
        {
            const response = await fetch('/_/config');
            if (response.ok)
            {
                const config = await response.json();
                // Use overview_url if available (includes full path), otherwise construct from websocket_url
                if (config.overview_url)
                {
                    cachedWsUrl = config.overview_url;
                    console.log('Using WebSocket URL from /_/config:', cachedWsUrl);
                    return cachedWsUrl;
                }
                else if (config.websocket_url)
                {
                    cachedWsUrl = config.websocket_url + '/_/ws/overview';
                    console.log('Using WebSocket URL from /_/config:', cachedWsUrl);
                    return cachedWsUrl;
                }
            }
        }
        catch (e)
        {
            // Ignore errors - fall back to default URL construction
            console.log('Failed to fetch /_/config, using default URL:', e.message);
        }

        // Default URL construction for normal operation (real system)
        // Uses location.host which includes port if present
        const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
        cachedWsUrl = `${protocol}://${location.host}/_/ws/overview`;
        console.log('Using default WebSocket URL:', cachedWsUrl);
        return cachedWsUrl;
    }

    getWebSocketUrl().then(wsUrl =>
    {
        ws = new WebSocket(wsUrl);

        // We want ArrayBuffer objects when receiving binary messages
        ws.binaryType = 'arraybuffer';

        ws.onopen =
            function(event)
            {
                console.log('Connected to WebSocket server');

                clearTimeout(scheduledReconnect);
                scheduledReconnect = undefined;

                // Reset the reconnection interval
                connectInterval = initialConnectIntervalMs;
            };

        ws.onmessage =
            function(event)
            {
                if (event.data instanceof ArrayBuffer)
                {
                    const int16Array = new Int16Array(event.data);

                    // We recieve more than one analysis result at a time,
                    // and split them here.
                    for (let i = 0; i < int16Array.length; i += channelCount * 3)
                    {
                        const view = int16Array.subarray(i, i + channelCount * 3);
                        analysis.push(view);

                        if (analysis.length > maxBufferSizeMs / analysisMs)
                            analysis.splice(0, 1);
                    }

                    if (buffering && analysis.length * analysisMs >= targetBufferSizeMs)
                    {
                        console.log(`Buffering complete, ${analysis.length * analysisMs} ms, targetBufferSizeMs: ${targetBufferSizeMs}`);
                        buffering = false;
                    }
                }
                else
                {
                    // console.log('Received text message: ' + event.data);

                    const message = JSON.parse(event.data);

                    if (message.status == 'reset')
                    {
                        if (message.analysisMs)
                            analysisMs = message.analysisMs;

                        // Use the long term mean as the DC offset
                        if (message.channel_mean)
                            for (let i = 0; i < channelCount; i++)
                            {
                                // For DC offset correction, we need the original channel mean
                                originalChannelMean[i]  = message.channel_mean[i];

                                // For hiding stim crosstalk, we need the current channel mean
                                channelMean[i]          = message.channel_mean[i];
                            }

                        if (message.channel_stddev)
                            for (let i = 0; i < channelCount; i++)
                                channelStdDev[i] = message.channel_stddev[i];

                        shouldHideCrosstalk.fill(0);
                        shouldInjectStimFlag.fill(0);
                        replaceWithMeanCount = 0;

                        analysis.splice(0, analysis.length);
                        buffering = true;
                    }
                    else if (message.status == 'status')
                    {
                        //
                        // The server periodically updates the rolling channel mean and stddev.
                        //

                        if (message.channel_mean)
                            for (let i = 0; i < channelCount; i++)
                                channelMean[i] = message.channel_mean[i];

                        if (message.channel_stddev)
                            for (let i = 0; i < channelCount; i++)
                                channelStdDev[i] = message.channel_stddev[i];
                    }
                }
            };

        ws.onclose =
            function()
            {
                console.log('WebSocket connection closed');
                disconnect();

                analysis.splice(0, analysis.length);
                buffering = true;

                if (document.visibilityState == 'visible')
                    scheduleReconnect();
            };

        ws.onerror =
            function(error)
            {
                console.log('WebSocket error');
                disconnect();

                if (document.visibilityState == 'visible')
                    scheduleReconnect();
            };
    });

    // Setup visibility change handling
    window.addEventListener('visibilitychange', considerVisibility);
    window.addEventListener('pageshow', considerVisibility);
}

function disconnect(removeListeners = false)
{
    if (ws)
    {
        ws.close();
        ws = undefined;
    }

    clearInterval(scheduledReconnect);
    scheduledReconnect = undefined;

    if (!removeListeners)
        return;

    window.removeEventListener('visibilitychange', considerVisibility);
    window.removeEventListener('pageshow', considerVisibility);
}

window.addEventListener('beforeunload', () => disconnect(true));

function considerVisibility()
{
    if (document.visibilityState == 'visible')
    {
        // Reset the reconnection interval and connect
        connectInterval = initialConnectIntervalMs;
        connect();
    }
    else
        disconnect();
}

let buffering = true;

export function isBuffering()
{
    return buffering;
}

export function getBufferSize()
{
    return analysis.length;
}

export function getAnalysisToRender(maxEntries)
{
    if (!maxEntries)
        return [];

    const analysisToRender = Math.min(maxEntries, analysis.length);

    if (analysisToRender < maxEntries)
        buffering = true;

    return processAnalysisToRender(analysis.splice(0, analysisToRender));
}

function preprocessAnalysis(int16Array)
{
    //
    // First we need to know if ANY channel is stimming,
    // as all* non-stimming channels will be replaced with the rolling mean.
    //
    // * except any that have stimmed during the period that we're replacing with mean ...
    //

    if (isHideCrosstalkEnabled)
        for (let channel = 0; channel < channelCount; channel++)
            if (int16Array[channel * 3 + 2] & FLAG_HAS_STIM)
            {
                // We'll be replacing the min/max data with the rolling mean.
                // Each channel that stims will be excepted from this.
                if (stimHideChunks)
                {
                    replaceWithMeanCount = stimHideChunks;
                    shouldHideCrosstalk.fill(1);
                }
                break;
            }

    for (let channel = 0; channel < channelCount; channel++)
    {
        let     min       = int16Array[channel * 3 + 0];
        let     max       = int16Array[channel * 3 + 1];
        const   flags     = int16Array[channel * 3 + 2];
        let     hasStim   = flags & FLAG_HAS_STIM;

        // Except any stimming channel from having its min/max replaced
        // This has been commented out to allow crosstalk hiding on stim channels too for cleaner display.
        // if (hasStim)
        //     shouldHideCrosstalk[channel] = 0;

        //
        // If we should, replace measured values with the rolling mean.
        //

        if (shouldHideCrosstalk[channel])
        {
            // Re-use previous non-stim min/max values
            min = previousChannelMin[channel];
            max = previousChannelMax[channel];
        }
        else
        {
            if (!hasStim)
            {
                // Remember the previous min/max for this channel
                previousChannelMin[channel] = min;
                previousChannelMax[channel] = max;
            }
        }

        if (isDcOffsetCorrectionEnabled)
        {
            //
            // Apply DC offset correction
            //

            min -= originalChannelMean[channel];
            max -= originalChannelMean[channel];
        }

        int16Array[channel * 3 + 0] = min;
        int16Array[channel * 3 + 1] = max;
    }

    if (replaceWithMeanCount)
        if (--replaceWithMeanCount == 0)
            shouldHideCrosstalk.fill(0);
}

function processAnalysisToRender(analysisToRender)
{
    //
    // Apply DC offset correction, if enabled
    // Apply normalisation, if enabled
    // Scale the data to range 0.0 to 1.0, ready for easy scaled rendering.
    //

    for (let i = 0; i < analysisToRender.length; i++)
    {
        const int16Array        = analysisToRender[i];
        const processedResult   = [];

        // Might as well reuse the same array
        analysisToRender[i]     = processedResult;

        // Update rolling averages, apply stateful adjustments, etc.
        preprocessAnalysis(int16Array);

        for (let channel = 0; channel < channelCount; channel++)
        {
            const visibleMin    = 0 - visibleAbsRaw;
            const visibleMax    = visibleAbsRaw;
            const visibleRange  = visibleMax - visibleMin;

            const min           = int16Array[channel * 3 + 0];
            const max           = int16Array[channel * 3 + 1];
            const flags         = int16Array[channel * 3 + 2];

            // Ignore spikes on unconnected ADC channels
            const hasSpike      = !!(flags & FLAG_HAS_SPIKE);
            let hasStim         = !!(flags & FLAG_HAS_STIM);

            // shouldInjectStimFlag is used to retain the "hasStim" flag for the set number of updates.
            // Currently set to 0 so only the update that actually had the stim pulse has the flag.
            if (hasStim)
                shouldInjectStimFlag[channel] = 0; // Previously 2;
            else if (shouldInjectStimFlag[channel])
            {
                shouldInjectStimFlag[channel]--;
                hasStim = true;
            }

            // Is this data visible at all with this range?
            if (max < visibleMin || min > visibleMax)
            {
                processedResult.push(
                    {
                        outOfRange: true,
                        hasSpike:   hasSpike,
                        hasStim:    hasStim
                    });
                continue;
            }

            // Clamp to visible range
            const clampedMin    = Math.min(visibleMax, Math.max(visibleMin, min));
            const clampedMax    = Math.min(visibleMax, Math.max(visibleMin, max));

            // Shift into 0 <--> visibleRange - 1
            const shiftedMin    = clampedMin - visibleMin;
            const shiftedMax    = clampedMax - visibleMin;

            // Scale into floating point 0.0 <-> 1.0
            const scaledMin     = shiftedMin / visibleRange;
            const scaledMax     = shiftedMax / visibleRange;

            processedResult.push(
                {
                    min:        scaledMin,
                    max:        scaledMax,
                    hasSpike:   hasSpike,
                    hasStim:    hasStim
                });
        }
    }

    return analysisToRender;
}