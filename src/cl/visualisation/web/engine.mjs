//
// This code assumes that the visualiser has been injected above,
// and that a function called 'createVisualiser' has been defined.
// 'createVisualiser' is the visualisation-specific implementation
// created by the user.
//
// These values are injected from Python:
//
//  const uniqueId      = '{unique_id}';              // ID of containing div
//  const dataStreams   = json.dumps(data_streams);   // Array of data stream names
//  const sshIp         = '{ssh_ip}';                 // SSH device IP, only used in VSCode Server mode (null if not used)
//  const websocketPort = 1025;                       // WebSocket server port
//

const visualisationDiv  = document.getElementById(uniqueId);
const vis               = createVisualiser(uniqueId, visualisationDiv);

function isUnderVsCode()
{
    return true;
}

function getRobustHostname()
{
    return sshIp ?? 'localhost';
}

function getWsProtocol()
{
    if (isUnderVsCode())
        return 'ws';

    return location.protocol === 'https:' ? 'wss' : 'ws';
}

const initialConnectIntervalMs  = 500;
const maxConnectIntervalMs      = 1000;

//
// This should be set large enough to account for network jitter.
// It will smooth the rendering when updates arrive late.
//

const bufferMs = vis.bufferMs !== undefined ? vis.bufferMs : 1000 / 60 * 1;

//
// Different jupyter environments / versions use different classes for the cell output.
// Basically we're looking for a parent that doesn't get replaced when the cell output
// is regenerated. We watch this element for mutation events so that we can detect when
// our content is removed, and then disconnect and disable the reconnection loop.
//
// If none of our classes match, we fall back to the body tag and log a warning.
//

const outputContainerClasses =
    [
        '.output_container',        // VSCode Jupyter
        '.output_wrapper',          // Jupyter on Gershom image
        '.jp-Cell-outputWrapper'    // Jupyter installed locally on macos via pip
    ];
let outputContainer;

while (!outputContainer && outputContainerClasses.length)
{
    const parentClass   = outputContainerClasses.shift();
    outputContainer     = visualisationDiv.closest(parentClass);
}

if (!outputContainer)
{
    // We didn't find a parent element. Log all parent candicate classes then fall back to the body tag.
    console.warn("Did not find visusalisation parent element by class, falling back to document.body. Candidate classes follow:")

    let element = visualisationDiv.parentElement;
    while (element)
    {
        console.warn('Candidate parent element classlist: ' + element.classList);
        element = element.parentElement;
    }

    outputContainer = document.body;
}

let ws;
let connectInterval = initialConnectIntervalMs;
let scheduledReconnect;

let hasBeenRemoved      = false;
let animationRequestId  = null;

let framesPerSecond;
let updateQueues;
let latestTimestamps;
let baseTimestampMs;
let baseTimestampFrames;

function reset()
{
    console.log('Resetting visualiser state');

    updateQueues        = {};
    latestTimestamps    = {};
    baseTimestampMs     = null;
    baseTimestampFrames = null;

    for (const dataStreamName of dataStreams)
    {
        latestTimestamps[dataStreamName]    = -1;
        updateQueues[dataStreamName]        = [];
    }
}

// If the visualiser is removed from the DOM, close the connection
const mutationObserver = new MutationObserver(
    (mutationsList) =>
    {
        if (!outputContainer.contains(visualisationDiv))
        {
            console.log('Visualiser has been removed from the DOM, closing WebSocket connection');
            hasBeenRemoved = true;
            ws.close();
            mutationObserver.disconnect();
            cancelAnimationFrame(animationRequestId);
            animationRequestId = null;
        }
    });
mutationObserver.observe(outputContainer, { childList: true, subtree: true });

function render(timestampMs)
{
    // Keep the render loop going
    animationRequestId = requestAnimationFrame(render);

    if (!baseTimestampMs)
    {
        // Check for an update in each queue, if we have any we sync on the lowest timestamp
        let lowestTimestamp = Number.MAX_SAFE_INTEGER;
        for (const dataStreamName of dataStreams)
        {
            const updateQueue = updateQueues[dataStreamName];
            if (updateQueue.length)
                lowestTimestamp = Math.min(lowestTimestamp, updateQueue[0].timestamp);
        }

        // No updates yet
        if (lowestTimestamp == Number.MAX_SAFE_INTEGER)
            return;

        console.log('Resetting render base timestamp with a buffer of ' + bufferMs + 'ms');

        // Synchronise based on this render timestamp + buffer ms, and the first recieved update
        baseTimestampMs     = timestampMs + bufferMs
        baseTimestampFrames = lowestTimestamp;
        return;
    }

    if (timestampMs < baseTimestampMs)
    {
        // We're still doing our initialbuffering
        return;
    }

    //
    // Determine our buffered render time, process updates with a lower
    // timestamp, and then invoke the visualisation draw function.
    //

    const relativeMs        = timestampMs - baseTimestampMs;
    const relativeFrames    = Math.floor(relativeMs * framesPerSecond / 1000);
    const renderUpToFrame   = baseTimestampFrames + relativeFrames;

    for (const dataStreamName of dataStreams)
    {
        const updateQueue = updateQueues[dataStreamName];
        while (updateQueue.length && updateQueue[0].timestamp < renderUpToFrame)
        {
            const dataToProcess = updateQueue.shift();
            vis.process(
                dataToProcess.data_stream,
                dataToProcess.timestamp,
                dataToProcess.data);
        }
    }

    //
    // Pass both the browser animation timestamp and the
    // appropriate data-stream-relative timestamp to the visualiser.
    //
    // The latter can be used to animate things relative to data stream
    // timestamps passed to process().
    //

    vis.draw(timestampMs, renderUpToFrame);
}

function connect()
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

    // Don't connect if our visualiser has been removed from the DOM
    if (!document.body.contains(visualisationDiv))
    {
        console.log('Visualiser has been removed from the DOM, not connecting');
        return;
    }

    // The hostname is not valid when running under vscode jupyter
    // TODO: Provide way to override hostname from the notebook
    const hostname          = getRobustHostname();
    const protocol          = getWsProtocol();
    const port              = websocketPort || 1025;
    const connectionString  = `${protocol}://${hostname}:${port}/_/ws/live_streaming`;

    ws = new WebSocket(connectionString);

    // We want ArrayBuffer objects when receiving binary messages
    ws.binaryType = 'arraybuffer';

    ws.onopen =
        function (event)
        {
            console.log(`Connected to WebSocket server at ${connectionString}`);

            clearTimeout(scheduledReconnect);
            scheduledReconnect = undefined;

            reset();
            vis.reset();

            // Reset the reconnection interval
            connectInterval = initialConnectIntervalMs;

            // Subscribe to the streams
            for (const dataStreamName of dataStreams)
                ws.send(
                    JSON.stringify(
                        {
                            action: 'subscribe',
                            type:   'data_stream',
                            name:   dataStreamName
                        }));

            // Start our rendering loop
            animationRequestId = requestAnimationFrame(render);
        };

    let pendingNewDataMessage;
    let binaryHandler;

    function handleDataStreamUpdate(data)
    {
        //
        // We previously recieved a new_data message,
        // and we now expect a msgpack encoded payload.
        //

        const message           = pendingNewDataMessage;
        pendingNewDataMessage   = null;

        const updateQueue = updateQueues[message.data_stream];
        if (!updateQueue)
            return;

        message.data = msgpack.decode(new Uint8Array(data));
        updateQueue.push(message);

        // TODO: Don't let the update queue grow indefinitely
    }

    function handleSpikes(data)
    {
        const message           = pendingNewDataMessage;
        pendingNewDataMessage   = null;

        const updateQueue = updateQueues['cl_spikes'];
        if (!updateQueue)
            return;

        const samplesPerSpike       = 75;
        const dataView              = new DataView(data);
        const timestampSizeBytes    = 8; // uint64
        const channelSizeBytes      = 1; // uint8
        const sampleSizeBytes       = 4; // float32

        const channelsOffset    = message.spike_count * timestampSizeBytes;
        const channelsSize      = message.spike_count * channelSizeBytes;
        const channelsPadding   = (timestampSizeBytes - (channelsSize & (timestampSizeBytes - 1))) & (timestampSizeBytes - 1);
        const samplesOffset     = channelsOffset + channelsSize + channelsPadding;

        for (let i = 0; i < message.spike_count; i++)
        {
            const tsOffset  = i * timestampSizeBytes;
            const tsLow     = dataView.getUint32(tsOffset, true);
            const tsHigh    = dataView.getUint32(tsOffset + 4, true);
            const timestamp = Number(BigInt(tsHigh) << 32n | BigInt(tsLow));

            const channel   = dataView.getUint8(channelsOffset + (i * channelSizeBytes), true);
            const samples   = new Float32Array(data, samplesOffset + (i * samplesPerSpike * sampleSizeBytes), samplesPerSpike);

            updateQueue.push({ data_stream: 'cl_spikes', timestamp, data: { channel, samples } });
        }
    }

    function handleStims(data) {
        const message = pendingNewDataMessage;
        pendingNewDataMessage = null;

        const updateQueue = updateQueues['cl_stims'];
        if (!updateQueue)
            return;

        const dataView = new DataView(data);
        const timestampSizeBytes = 8; // uint64
        const channelSizeBytes = 1; // uint8

        const channelsOffset = message.stim_count * timestampSizeBytes;

        for (let i = 0; i < message.stim_count; i++)
        {
            const tsOffset = i * timestampSizeBytes;
            const tsLow = dataView.getUint32(tsOffset, true);
            const tsHigh = dataView.getUint32(tsOffset + 4, true);
            const timestamp = Number(BigInt(tsHigh) << 32n | BigInt(tsLow));

            const channel = dataView.getUint8(channelsOffset + (i * channelSizeBytes), true);

            updateQueue.push({ data_stream: 'cl_stims', timestamp, data: { channel } });
        }
    }

    ws.onmessage =
        function (event)
        {
            if (event.data instanceof ArrayBuffer)
            {
                if (binaryHandler)
                {
                    const handler = binaryHandler;
                    binaryHandler = undefined;

                    handler(event.data);
                }
                else
                    console.log('Received unexpected binary message: ' + event.data);

                return;
            }

            const message = JSON.parse(event.data);

            if (message.status == 'new_data')
            {
                // The data itself will arrive in binary form as the next message
                pendingNewDataMessage   = message;
                binaryHandler           = handleDataStreamUpdate;
            }
            else if (message.status == 'cl_spikes')
            {
                // The data itself will arrive in binary form as the next message
                pendingNewDataMessage   = message;
                binaryHandler           = handleSpikes;
            }
            else if (message.status == 'cl_stims')
            {
                // The data itself will arrive in binary form as the next message
                pendingNewDataMessage   = message;
                binaryHandler           = handleStims;
            }
            else if (message.status == 'reset')
            {
                if (framesPerSecond != message.frames_per_second)
                {
                    console.log('Frames per second set to ' + message.frames_per_second + 'Hz');
                    framesPerSecond = message.frames_per_second;
                }

                reset();
                vis.reset();
            }
            else if (message.status == 'attributes_reset')
            {
                // console.log(`Data stream [${message.data_stream}]: attributes reset to: ${JSON.stringify(message.attributes)}`);

                if (vis.attributesReset)
                    vis.attributesReset(message.data_stream, message.attributes);
            }
            else if (message.status == 'attributes_updated')
            {
                // console.log(`Data stream [${message.data_stream}]: attributes updated: ${JSON.stringify(message.attributes)}`);

                if (vis.attributesUpdated)
                    vis.attributesUpdated(message.data_stream, message.attributes);
            }
            else
                console.log('Unhandled text message: ' + event.data);
        };

    ws.onclose =
        function ()
        {
            console.log('WebSocket connection closed');
            cancelAnimationFrame(animationRequestId);
            animationRequestId = null;
            if (!hasBeenRemoved && document.visibilityState == 'visible')
                scheduleReconnect();
        };

    ws.onerror =
        function (error)
        {
            console.log('WebSocket error, closing and scheduling reconnect');
            ws.close();

            if (document.visibilityState == 'visible')
                scheduleReconnect();
        };
}

function disconnect() {
    if (ws) {
        ws.close();
        ws = undefined;
    }
    clearInterval(scheduledReconnect);
    scheduledReconnect = undefined;
}

window.addEventListener('beforeunload', disconnect);

function considerVisibility() {
    if (document.visibilityState === "visible") {
        // Reset the reconnection interval and connect
        connectInterval = initialConnectIntervalMs;
        connect();
    } else {
        disconnect();
    }
}

window.addEventListener("visibilitychange", () => {
    considerVisibility();
});

window.addEventListener("pageshow", () => {
    considerVisibility();
});

considerVisibility();
