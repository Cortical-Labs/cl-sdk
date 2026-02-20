"""
WebSocket server subprocess wrapper.

This module provides a subprocess wrapper for the WebSocket server,
isolating it from the main process GIL.

The subprocess connects to the shared memory buffer and runs the
WebSocket server independently.
"""
from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import multiprocessing
import os
import signal
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, override
from urllib.parse import urlparse

import numpy as np
import websockets as ws

from .. import Spike, Stim
from .._data_buffer import DataStreamEventRecord, SharedDataBuffer
from .._data_producer import STALE_THRESHOLD_NS
from ._http_server import StaticHttpServer

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing import Queue
    from multiprocessing.process import BaseProcess

class WebSocketHandshakeFilter(logging.Filter):
    """Filter to suppress cosmetic handshake errors when clients probe before server is ready."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress "connection closed while reading HTTP request" errors
        msg = record.getMessage()

        if "opening handshake failed" in msg or "connection closed while reading HTTP request" in msg or "did not receive a valid HTTP request" in msg:
            return False

        # Also check the exception info
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type and exc_type.__name__ in {"InvalidMessage", "EOFError"}:
                # Check if it's a handshake-related error
                exc_str = str(record.exc_info[1])
                if "did not receive a valid HTTP request" in exc_str:
                    return False
                if "connection closed while reading HTTP request" in exc_str:
                    return False
                if "stream ends after 0 bytes" in exc_str:
                    return False

        return True

_logger = logging.getLogger("cl.websocket.subprocess")
_logger.addFilter(WebSocketHandshakeFilter())

# Protocol constants
ANALYSIS_SIZE_MS = 5
FLAG_HAS_SPIKE   = 1 << 0
FLAG_HAS_STIM    = 1 << 1

def _to_json(obj: dict) -> str:
    """Serialize dict to compact JSON string."""
    return json.dumps(obj, separators=(',', ':'))

def _make_status(status: str, data: dict | None = None) -> str:
    """Create a status message matching the system protocol."""
    message = {"status": status}
    if data:
        message.update(data)
    return _to_json(message)

def _run_websocket_subprocess(config: SubprocessConfig, ready_queue: Queue, command_queue: Queue) -> None:
    """
    Entry point for WebSocket subprocess.

    Args:
        config: Subprocess configuration
        ready_queue: Queue to signal when server is ready
        command_queue: Queue for receiving commands from main process
    """

    # First thing: print to confirm we're running
    _logger.info("Subprocess entry point reached")

    try:
        # Lower process priority to avoid competing with the main loop
        with contextlib.suppress(OSError, AttributeError):
            os.nice(10)

        _logger.info("WebSocket subprocess starting, connecting to buffer: %s", config.buffer_name)

        # Connect to existing shared memory buffer
        buffer = SharedDataBuffer.attach(as_producer=False, name_prefix=config.buffer_name)
        _logger.info("Connected to shared buffer")

        # Create lightweight reader
        reader = BufferReader(buffer, config.frames_per_second, config.channel_count)

        # Create a modified WebSocket server that uses our reader
        server = SubprocessWebSocketServer(
            reader        = reader,
            port          = config.port,
            host          = config.host,
            serve_vis     = config.serve_vis,
            web_directory = config.web_directory,
            app_html      = config.app_html,
            ready_queue   = ready_queue,    # Pass queue so server can signal when actually ready
            command_queue = command_queue,  # Pass command queue for attribute updates
        )

        # Run forever until terminated (ready signal sent from within _serve)
        server.run_forever()

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _logger.exception("WebSocket subprocess error: %s", e)
        with contextlib.suppress(Exception):
            ready_queue.put({"status": "error", "error": f"{e}\n{tb}"})
    finally:
        _logger.info("WebSocket subprocess shutting down")

@dataclass
class SubprocessConfig:
    """Configuration for WebSocket subprocess."""
    buffer_name      : str
    frames_per_second: int
    channel_count    : int
    port             : int        = 1025
    host             : str        = "127.0.0.1"
    serve_vis        : bool       = True
    web_directory    : str | None = None
    app_html         : str | None = None

@dataclass
class AttributeUpdateCommand:
    """Command to update data stream attributes."""
    stream_name       : str
    updated_attributes: dict

@dataclass
class AttributeResetCommand:
    """Command to reset data stream attributes."""
    stream_name: str
    attributes : dict

@dataclass(eq=False)
class ClientSession:
    """Represents a connected WebSocket client."""

    websocket              : ws.ServerConnection
    path                   : str
    subscribed_data_streams: set[str] = field(default_factory=set)

    async def send_text(self, message: str) -> bool:
        """Send text message to client. Returns False if send fails."""
        try:
            await self.websocket.send(message)
            return True
        except Exception as e:
            _logger.warning("Failed to send text to %s: %s", self.path, e)
            return False

    async def send_binary(self, data: bytes) -> bool:
        """Send binary message to client. Returns False if send fails."""
        try:
            await self.websocket.send(data)
            return True
        except Exception as e:
            _logger.debug("Failed to send binary to %s: %s", self.path, e)
            return False

    @property
    def is_open(self) -> bool:
        """Check if the WebSocket connection is still open."""
        try:
            state = self.websocket.state
            if state is not None:
                return state == ws.protocol.State.OPEN

            # Assume open if no way to check
            return True
        except Exception as e:
            _logger.debug("is_open check failed: %s", e)
            return False

class BufferReader:
    """
    Lightweight reader for SharedDataBuffer in subprocess.

    Provides a similar interface to Neurons for reading data,
    but operates directly on the shared memory buffer.
    """

    def __init__(self, buffer: SharedDataBuffer, frames_per_second: int, channel_count: int):
        self._buffer            = buffer
        self._frames_per_second = frames_per_second
        self._channel_count     = channel_count
        self._frame_duration_us = 1_000_000 / frames_per_second

    @property
    def heartbeat_ns(self) -> int:
        return self._buffer.main_process_heartbeat_ns

    def get_frames_per_second(self) -> int:
        return self._frames_per_second

    def get_channel_count(self) -> int:
        return self._channel_count

    def get_frame_duration_us(self) -> float:
        return self._frame_duration_us

    def timestamp(self) -> int:
        """Get current timestamp from buffer."""
        return self._buffer.write_timestamp

    def start_timestamp(self) -> int:
        """Get oldest available timestamp from buffer."""
        return self._buffer.start_timestamp

    def read(self, frame_count: int, from_timestamp: int) -> np.ndarray:
        """Read frames from buffer."""
        return self._buffer.read_frames(from_timestamp, frame_count)

    def read_spikes(self, frame_count: int, from_timestamp: int) -> list[Spike]:
        """Read spikes from buffer."""
        to_ts         = from_timestamp + frame_count
        spike_records = self._buffer.read_spikes(from_timestamp, to_ts)

        # Convert SpikeRecords to Spike objects for compatibility
        return [
            Spike(
                timestamp           = s.timestamp,
                channel             = s.channel,
                channel_mean_sample = s.channel_mean_sample,
                samples             = s.samples,
            )
            for s in spike_records
        ]

    def read_stims(self, from_timestamp: int, to_timestamp: int) -> list[Stim]:
        """Read stims from buffer."""
        stim_records = self._buffer.read_stims(from_timestamp, to_timestamp)

        # Convert StimRecords to Stim objects for compatibility
        return [Stim(timestamp=s.timestamp, channel=s.channel) for s in stim_records]

    def read_datastream_events(self, from_timestamp: int, to_timestamp: int) -> list[DataStreamEventRecord]:
        """Read datastream events from buffer."""
        return self._buffer.read_datastream_events(from_timestamp, to_timestamp)

class OverviewProtocol:
    """Protocol for the /_/ws/overview WebSocket endpoint."""

    RESET : Final = "reset"
    STATUS: Final = "status"

    @staticmethod
    def make_reset(analysis_ms: int, channel_mean: list[float], channel_stddev: list[float]) -> str:
        return _make_status(
            OverviewProtocol.RESET,
            {
                "analysisMs"    : analysis_ms,
                "channel_mean"  : channel_mean,
                "channel_stddev": channel_stddev,
            }
        )

    @staticmethod
    def make_status(channel_mean: list[float], channel_stddev: list[float]) -> str:
        return _make_status(
            OverviewProtocol.STATUS,
            {
                "channel_mean"  : channel_mean,
                "channel_stddev": channel_stddev,
            }
        )

class LiveStreamingProtocol:
    """Protocol for the /_/ws/live_streaming WebSocket endpoint."""

    RESET             : Final = "reset"
    SUBSCRIBE         : Final = "subscribe"
    TYPE_DATA_STREAM  : Final = "data_stream"
    NEW_DATA          : Final = "new_data"
    CL_SPIKES         : Final = "cl_spikes"
    CL_STIMS          : Final = "cl_stims"
    ATTRIBUTES_RESET  : Final = "attributes_reset"
    ATTRIBUTES_UPDATED: Final = "attributes_updated"
    SAMPLES_PER_SPIKE : Final = 75

    @staticmethod
    def make_reset(frames_per_second: int) -> str:
        return _make_status(
            LiveStreamingProtocol.RESET,
            {"frames_per_second": frames_per_second}
        )

    @staticmethod
    def make_cl_spikes(spike_count: int) -> str:
        return _make_status(
            LiveStreamingProtocol.CL_SPIKES,
            {"spike_count": spike_count}
        )

    @staticmethod
    def make_cl_stims(stim_count: int) -> str:
        return _make_status(
            LiveStreamingProtocol.CL_STIMS,
            {"stim_count": stim_count}
        )

    @staticmethod
    def make_spikes_payload(spikes: list[Spike]) -> bytes:
        """
        Create binary payload for spikes.

        Format (each section starts on 64-bit boundary):
        - Timestamps: n × uint64
        - Channels: n × uint8 (padded to 8-byte boundary)
        - Samples: n × 75 × float32
        """
        n = len(spikes)
        if n == 0:
            return b""

        timestamp_size       = n * 8  # uint64
        channel_size         = n * 1  # uint8
        channel_padding_size = (8 - (n & 7)) & 7
        samples_size         = n * LiveStreamingProtocol.SAMPLES_PER_SPIKE * 4  # float32
        total_size           = timestamp_size + channel_size + channel_padding_size + samples_size

        payload = np.empty(total_size, dtype=np.uint8)

        # Timestamps section
        offset      = 0
        timestamps  = payload[offset:offset + timestamp_size].view(dtype=np.uint64)
        offset     += timestamp_size

        # Channels section
        channels  = payload[offset:offset + channel_size]
        offset   += channel_size + channel_padding_size

        # Samples section
        samples = payload[offset:offset + samples_size].view(dtype=np.float32)

        for i, spike in enumerate(spikes):
            timestamps[i]                     = spike.timestamp
            channels[i]                       = spike.channel
            sample_offset                     = i * LiveStreamingProtocol.SAMPLES_PER_SPIKE
            sample_end                        = sample_offset + LiveStreamingProtocol.SAMPLES_PER_SPIKE
            samples[sample_offset:sample_end] = spike.samples

        return payload.tobytes()

    @staticmethod
    def make_stims_payload(stims: list[Stim]) -> bytes:
        """
        Create binary payload for stims.

        Format (each section starts on 64-bit boundary):
        - Timestamps: n × uint64
        - Channels: n × uint8 (padded to 8-byte boundary)
        """
        n = len(stims)
        if n == 0:
            return b""

        timestamp_size       = n * 8  # uint64
        channel_size         = n * 1  # uint8
        channel_padding_size = (8 - (n & 7)) & 7
        total_size           = timestamp_size + channel_size + channel_padding_size

        payload = np.empty(total_size, dtype=np.uint8)

        offset      = 0
        timestamps  = payload[offset:offset + timestamp_size].view(dtype=np.uint64)
        offset     += timestamp_size
        channels    = payload[offset:offset + channel_size]

        for i, stim in enumerate(stims):
            timestamps[i] = stim.timestamp
            channels[i]   = stim.channel

        return payload.tobytes()

    @staticmethod
    def make_new_data(stream_name: str, timestamp: int) -> str:
        """Create header for a custom data stream event."""
        return _make_status(
            LiveStreamingProtocol.NEW_DATA,
            {
                "data_stream": stream_name,
                "timestamp"  : timestamp
            }
        )

    @staticmethod
    def make_attributes_reset(stream_name: str, attributes: dict) -> str:
        """Create message for attribute reset."""
        return _make_status(
            LiveStreamingProtocol.ATTRIBUTES_RESET,
            {
                "data_stream": stream_name,
                "attributes" : attributes
            }
        )

    @staticmethod
    def make_attributes_updated(stream_name: str, updated_attributes: dict) -> str:
        """Create message for attribute update."""
        return _make_status(
            LiveStreamingProtocol.ATTRIBUTES_UPDATED,
            {
                "data_stream": stream_name,
                "attributes" : updated_attributes
            }
        )

class SubprocessWebSocketServer:
    """
    WebSocket server that runs in subprocess using BufferReader.
    """

    def __init__(
        self,
        reader       : BufferReader,
        port         : int                                                          = 1025,
        host         : str                                                          = "127.0.0.1",
        serve_vis    : bool                                                         = True,
        web_directory: str | None                                                   = None,
        ready_queue  : Queue[dict[str, Any]] | None                                 = None,
        command_queue: Queue[AttributeUpdateCommand | AttributeResetCommand] | None = None,
        app_html     : str | None                                                   = None,
    ):
        self._reader        = reader
        self._port          = port
        self._host          = host
        self._serve_vis     = serve_vis
        self._web_directory = web_directory
        self._web_server    = None
        self._running       = False
        self._running_event = asyncio.Event()
        self._ready_queue   = ready_queue
        self._command_queue = command_queue

        # Track data stream attributes for sending to newly subscribed clients
        # Use an OrderedDict for LRU-style cleanup (oldest entries removed first)
        self._data_stream_attributes: OrderedDict[str, dict] = OrderedDict()
        self._max_data_stream_attributes = 1000  # Limit to prevent unbounded growth

        # Track pending broadcast task to prevent accumulation
        self._pending_broadcast_task: asyncio.Task | None = None

        # Start web server if enabled
        if serve_vis:
            self._web_server = StaticHttpServer(websocket_host=host, websocket_port=port, host=host, app_html=app_html)
            self._web_server.start()

    @property
    def web_port(self) -> int | None:
        if self._web_server:
            return self._web_server.port
        return None

    @property
    def web_url(self) -> str | None:
        if self._web_server:
            return self._web_server.url
        return None

    @property
    def app_url(self) -> str | None:
        if self._web_server:
            return self._web_server.app_url
        return None

    def run_forever(self) -> None:
        """Run the WebSocket server forever."""

        self._running = True
        self._running_event.clear()

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, _):
            _logger.info("Received signal %s, shutting down", signum)
            self._running = False
            self._running_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Run the async server
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        """Main async server coroutine."""

        # Client tracking
        self._overview_clients      : set[ClientSession] = set()
        self._live_streaming_clients: set[ClientSession] = set()

        # Create live reader
        self._live_reader = SubprocessLiveReader(
            self._reader,
            self._on_new_frames,
            self._on_seek_detected,
        )

        # Server options
        server_options = {
            "ping_interval": 30,
            "ping_timeout" : 10,
            "close_timeout": 5,
            "reuse_address": True,  # Allow immediate port reuse
            "logger"       : _logger,
        }

        _logger.info("Starting WebSocket server on %s:%d", self._host, self._port)

        try:
            async with ws.serve(
                self._handle_connection,
                self._host,
                self._port,
                **server_options,
            ):
                # Signal ready NOW - after server is actually listening
                if self._ready_queue is not None:
                    self._ready_queue.put({
                        "status"  : "ready",
                        "port"    : self._port,
                        "web_port": self.web_port,
                        "web_url" : self.web_url,
                        "app_url" : self.app_url,
                    })
                await self._server_loop()
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                _logger.warning("Port %d already in use", self._port)
            else:
                _logger.exception("Server error: %s", e)

    async def _server_loop(self) -> None:
        """Main server loop."""

        _logger.info("WebSocket server ready on ws://%s:%d", self._host, self._port)

        # Start live reader
        reader_task = asyncio.create_task(self._live_reader.run())

        # Start command processor
        command_task = asyncio.create_task(self._process_commands())

        # Keep running until stopped
        await self._running_event.wait()

        # Cancel tasks
        reader_task.cancel()
        command_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader_task
        with contextlib.suppress(asyncio.CancelledError):
            await command_task

        # Stop web server
        if self._web_server:
            self._web_server.stop()

    async def _process_commands(self) -> None:
        """Process commands from the main process."""

        while self._running:
            try:
                if self._command_queue is None:
                    continue

                while True:
                    try:
                        # Use get_nowait to avoid blocking the event loop
                        cmd = self._command_queue.get_nowait()
                        match cmd:
                            case AttributeUpdateCommand():
                                # Update tracked attributes (move to end for LRU ordering)
                                if cmd.stream_name in self._data_stream_attributes:
                                    self._data_stream_attributes.move_to_end(cmd.stream_name)
                                else:
                                    self._data_stream_attributes[cmd.stream_name] = {}
                                self._data_stream_attributes[cmd.stream_name].update(cmd.updated_attributes)
                                # Enforce size limit by removing oldest entries
                                while len(self._data_stream_attributes) > self._max_data_stream_attributes:
                                    self._data_stream_attributes.popitem(last=False)
                                # Broadcast to subscribers
                                await self._broadcast_attribute_update(cmd.stream_name, cmd.updated_attributes)
                            case AttributeResetCommand():
                                # Set full attribute set (move to end for LRU ordering)
                                if cmd.stream_name in self._data_stream_attributes:
                                    self._data_stream_attributes.move_to_end(cmd.stream_name)
                                self._data_stream_attributes[cmd.stream_name] = cmd.attributes.copy()
                                # Enforce size limit by removing oldest entries
                                while len(self._data_stream_attributes) > self._max_data_stream_attributes:
                                    self._data_stream_attributes.popitem(last=False)
                                # Broadcast to subscribers
                                await self._broadcast_attribute_reset(cmd.stream_name, cmd.attributes)
                    except Exception:
                        # Queue is empty or error occurred
                        break
            finally:
                # Sleep briefly before checking again
                await asyncio.sleep(0.05)  # Check every 50ms

    async def _broadcast_attribute_update(self, stream_name: str, updated_attributes: dict) -> None:
        """Broadcast attribute update to subscribed clients."""

        message = LiveStreamingProtocol.make_attributes_updated(stream_name, updated_attributes)

        # Send to all live streaming clients subscribed to this stream
        for client in list(self._live_streaming_clients):
            if stream_name in client.subscribed_data_streams:
                with contextlib.suppress(Exception):
                    await client.send_text(message)

    async def _broadcast_attribute_reset(self, stream_name: str, attributes: dict) -> None:
        """Broadcast full attribute set to subscribed clients."""

        message = LiveStreamingProtocol.make_attributes_reset(stream_name, attributes)

        # Send to all live streaming clients subscribed to this stream
        for client in list(self._live_streaming_clients):
            if stream_name in client.subscribed_data_streams:
                with contextlib.suppress(Exception):
                    await client.send_text(message)

    async def _send_initial_attributes(self, client: ClientSession, stream_name: str) -> None:
        """Send initial attributes for a data stream to a newly subscribed client."""

        # Get the data stream from neurons (need to access via shared state or queue)
        # For now, we'll need to track data stream attributes in the subprocess
        # This will be populated via a command when attributes are set
        if hasattr(self, '_data_stream_attributes'):
            attributes = self._data_stream_attributes.get(stream_name)
            if attributes:
                message = LiveStreamingProtocol.make_attributes_reset(stream_name, attributes)
                try:
                    await client.send_text(message)
                except Exception as e:
                    _logger.debug("Failed to send initial attributes for %s: %s", stream_name, e)

    async def _handle_connection(self, websocket: ws.ServerConnection) -> None:
        """Handle incoming WebSocket connection."""

        if websocket.request is None:
            await websocket.close(ws.CloseCode.PROTOCOL_ERROR, "No request information")
            return

        path = websocket.request.path

        if path and path.startswith("http"):
            path = str(urlparse(path).path)

        _logger.info("New connection: %s", path)

        try:
            if path == "/_/ws/overview":
                await self._handle_overview(websocket)
            elif path == "/_/ws/live_streaming":
                await self._handle_live_streaming(websocket)
            else:
                await websocket.close(ws.CloseCode.PROTOCOL_ERROR, f"Unknown path: {path}")
        except Exception as e:
            _logger.warning("Connection error on %s: %s", path, e, exc_info=True)

    async def _handle_overview(self, websocket) -> None:
        """Handle overview connection."""

        client = ClientSession(websocket=websocket, path="/_/ws/overview")
        self._overview_clients.add(client)

        try:
            # Send reset message
            channel_mean, channel_stddev = self._live_reader.get_channel_stats()
            reset_msg = OverviewProtocol.make_reset(
                ANALYSIS_SIZE_MS,
                channel_mean,
                channel_stddev
            )
            await client.send_text(reset_msg)

            # Handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get("action")

                    if action == "stimulate":
                        channel = data.get("channel", 0)
                        _logger.debug("Stimulate request for channel %d (ignored in subprocess)", channel)

                    elif action == "reset":
                        channel_mean, channel_stddev = self._live_reader.get_channel_stats()
                        await client.send_text(
                            OverviewProtocol.make_reset(
                                ANALYSIS_SIZE_MS,
                                channel_mean,
                                channel_stddev
                            )
                        )

                except json.JSONDecodeError:
                    pass

        except Exception as e:
            _logger.warning("Overview client error: %s", e, exc_info=True)
        finally:
            self._overview_clients.discard(client)

    async def _handle_live_streaming(self, websocket: ws.ServerConnection) -> None:
        """Handle live streaming connection."""

        client = ClientSession(websocket=websocket, path="/_/ws/live_streaming")
        self._live_streaming_clients.add(client)

        try:
            # Send reset message
            reset_msg = LiveStreamingProtocol.make_reset(self._reader.get_frames_per_second())
            await client.send_text(reset_msg)

            # Handle messages
            async for message in websocket:
                try:
                    data  : dict[str, Any] = json.loads(message)
                    action: str | None     = data.get("action")

                    if action == LiveStreamingProtocol.SUBSCRIBE:
                        sub_type: str | None = data.get("type")
                        name    : str | None = data.get("name")
                        if sub_type == LiveStreamingProtocol.TYPE_DATA_STREAM and name:
                            client.subscribed_data_streams.add(name)
                            # Send initial attributes for this data stream if it exists
                            # Await directly instead of fire-and-forget task
                            await self._send_initial_attributes(client, name)

                except json.JSONDecodeError:
                    pass

        except Exception as e:
            _logger.warning("Live streaming client error: %s", e, exc_info=True)
        finally:
            self._live_streaming_clients.discard(client)

    def _on_new_frames(
        self,
        frames_ts        : int,
        frames           : np.ndarray,
        spikes           : list[Spike],
        stims            : list[Stim],
        datastream_events: list[DataStreamEventRecord]
    ) -> None:
        """Called by live reader when new frames are available."""

        # Get the running event loop
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Cancel any pending broadcast task to prevent task accumulation.
        # This ensures only one broadcast is in-flight at a time, providing
        # natural backpressure when broadcasts take longer than the interval.
        if self._pending_broadcast_task is not None and not self._pending_broadcast_task.done():
            self._pending_broadcast_task.cancel()

        # Schedule new broadcast
        self._pending_broadcast_task = asyncio.create_task(
            self._broadcast_frames(frames_ts, frames, spikes, stims, datastream_events)
        )
        self._pending_broadcast_task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() else None
        )

    def _on_seek_detected(self) -> None:
        """Called by live reader when a seek/jump is detected."""

        # Get the running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Schedule reset broadcast (fire-and-forget with done callback to suppress exceptions)
        task = asyncio.ensure_future(self._broadcast_seek_reset(), loop=loop)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def _broadcast_seek_reset(self) -> None:
        """Broadcast reset messages to all clients after a seek."""
        _logger.debug("Broadcasting seek reset to clients")

        # Get updated channel stats
        channel_mean, channel_stddev = self._live_reader.get_channel_stats()

        # Send reset to overview clients
        reset_msg = OverviewProtocol.make_reset(ANALYSIS_SIZE_MS, channel_mean, channel_stddev)
        for client in list(self._overview_clients):
            with contextlib.suppress(Exception):
                await client.send_text(reset_msg)

        # Send reset to live streaming clients
        live_reset_msg = LiveStreamingProtocol.make_reset(self._reader.get_frames_per_second())
        for client in list(self._live_streaming_clients):
            with contextlib.suppress(Exception):
                await client.send_text(live_reset_msg)

        # Send initial attributes for all data streams to live streaming clients
        for stream_name, attributes in self._data_stream_attributes.items():
            message = LiveStreamingProtocol.make_attributes_reset(stream_name, attributes)
            for client in list(self._live_streaming_clients):
                if stream_name in client.subscribed_data_streams:
                    with contextlib.suppress(Exception):
                        await client.send_text(message)

    async def _broadcast_frames(
        self,
        frames_ts        : int,
        frames           : np.ndarray,
        spikes           : list[Spike],
        stims            : list[Stim],
        datastream_events: list[DataStreamEventRecord]
    ) -> None:
        """Broadcast frame data to connected clients."""
        try:
            # Broadcast to overview clients
            if self._overview_clients:
                await self._broadcast_overview(frames_ts, frames, spikes, stims)

            # Broadcast to live streaming clients
            if self._live_streaming_clients:
                await self._broadcast_live_streaming(spikes, stims, datastream_events)
        except Exception as e:
            _logger.debug("Broadcast error: %s", e)

    async def _broadcast_overview(
        self,
        frames_ts: int,
        frames   : np.ndarray,
        spikes   : list[Spike],
        stims    : list[Stim]
    ) -> None:
        """Broadcast to overview clients."""

        fps = self._reader.get_frames_per_second()
        analysis_size_frames = fps * ANALYSIS_SIZE_MS // 1000
        chunk_count          = len(frames) // analysis_size_frames
        channel_count        = frames.shape[1]

        if chunk_count == 0:
            return

        # Build payload
        payload = np.empty((chunk_count, channel_count, 3), dtype=np.int16)

        chunk_frames_offset = 0
        chunk_end_ts        = frames_ts + analysis_size_frames
        spike_offset        = 0
        stim_offset         = 0

        for i in range(chunk_count):
            chunk_frames = frames[chunk_frames_offset:chunk_frames_offset + analysis_size_frames]

            min_values = np.min(chunk_frames, axis=0)
            max_values = np.max(chunk_frames, axis=0)
            flags      = np.zeros(channel_count, dtype=np.int16)

            while spike_offset < len(spikes):
                spike = spikes[spike_offset]
                if spike.timestamp >= chunk_end_ts:
                    break
                flags[spike.channel] |= FLAG_HAS_SPIKE
                spike_offset += 1

            while stim_offset < len(stims):
                stim = stims[stim_offset]
                if stim.timestamp >= chunk_end_ts:
                    break
                flags[stim.channel] |= FLAG_HAS_STIM
                stim_offset += 1

            payload[i] = np.column_stack((min_values, max_values, flags))

            chunk_frames_offset += analysis_size_frames
            chunk_end_ts        += analysis_size_frames

        # Send to clients
        binary_data  = payload.tobytes()
        disconnected = []

        for client in list(self._overview_clients):
            if not client.is_open:
                disconnected.append(client)
                continue
            if not await client.send_binary(binary_data):
                disconnected.append(client)

        for client in disconnected:
            self._overview_clients.discard(client)

    @staticmethod
    async def send_spikes(client: ClientSession, spikes: list[Spike]) -> bool:
        """Send spikes to a client. Returns False if send fails."""
        header  = LiveStreamingProtocol.make_cl_spikes(len(spikes))
        payload = LiveStreamingProtocol.make_spikes_payload(spikes)
        if not await client.send_text(header):
            return False
        return not (payload and not await client.send_binary(payload))

    @staticmethod
    async def send_stims(client: ClientSession, stims: list[Stim]) -> bool:
        """Send stims to a client. Returns False if send fails."""
        header  = LiveStreamingProtocol.make_cl_stims(len(stims))
        payload = LiveStreamingProtocol.make_stims_payload(stims)
        if not await client.send_text(header):
            return False
        return not (payload and not await client.send_binary(payload))

    @staticmethod
    async def send_datastream_events(client: ClientSession, events: list[DataStreamEventRecord]) -> bool:
        for event in events:
            if event.stream_name in client.subscribed_data_streams:
                header  = LiveStreamingProtocol.make_new_data(event.stream_name, event.timestamp)
                payload = event.data
                if not await client.send_text(header):
                    return False
                if payload and not await client.send_binary(payload):
                    return False

        return True

    async def _broadcast_live_streaming(self, spikes: list[Spike], stims: list[Stim], datastream_events: list[DataStreamEventRecord]) -> None:
        """Broadcast to live streaming clients."""
        disconnected: set[ClientSession] = set()

        for client in list(self._live_streaming_clients):
            try:
                if spikes and "cl_spikes" in client.subscribed_data_streams and not await self.send_spikes(client, spikes):
                    disconnected.add(client)
                    continue

                if stims and "cl_stims" in client.subscribed_data_streams and not await self.send_stims(client, stims):
                    disconnected.add(client)
                    continue

                if datastream_events and not await self.send_datastream_events(client, datastream_events):
                    continue

            except Exception:
                disconnected.add(client)

        self._live_streaming_clients -= disconnected

class SubprocessLiveReader:
    """
    Live reader for subprocess that reads from BufferReader.

    Uses a simple throttled approach: reads and broadcasts at a fixed interval
    rather than trying to keep up with real-time data. This prevents CPU
    spinning and provides consistent data flow to clients.
    """

    BROADCAST_INTERVAL_MS         = 50  # Broadcast every 50ms (20 Hz)
    BURST_MS                      = 50  # How much data to read per broadcast (match interval to avoid gaps/overlaps)
    CAPTURED_SPIKE_SAMPLES_BEFORE = 25
    CAPTURED_SPIKE_SAMPLES_AFTER  = 49
    CAPTURED_SPIKE_SAMPLES        = CAPTURED_SPIKE_SAMPLES_BEFORE + 1 + CAPTURED_SPIKE_SAMPLES_AFTER

    def __init__(
        self,
        reader          : BufferReader,
        on_new_frames   : Callable[[int, np.ndarray, list[Spike], list[Stim], list[DataStreamEventRecord]], None],
        on_seek_detected: Callable[[], None] | None = None,
    ):
        self._reader                   = reader
        self._on_new_frames            = on_new_frames
        self._on_seek_detected         = on_seek_detected
        self._frames_per_second        = reader.get_frames_per_second()
        self._channel_count            = reader.get_channel_count()
        self._read_size_frames         = int(self._frames_per_second * self.BURST_MS / 1000)
        self._running                  = False
        self._next_read_ts: int | None = None

        # Running stats
        self._channel_sum      = np.zeros(self._channel_count, dtype=np.float64)
        self._channel_sum_sq   = np.zeros(self._channel_count, dtype=np.float64)
        self._sample_count     = 0
        self._spikes_broadcast = 0
        self._zero_spike_reads = 0

    def reset_stats(self) -> None:
        """Reset running channel statistics."""
        self._channel_sum    = np.zeros(self._channel_count, dtype=np.float64)
        self._channel_sum_sq = np.zeros(self._channel_count, dtype=np.float64)
        self._sample_count   = 0

    def get_channel_stats(self) -> tuple[list[float], list[float]]:
        if self._sample_count == 0:
            return [0.0] * self._channel_count, [1.0] * self._channel_count

        mean     = self._channel_sum / self._sample_count
        variance = (self._channel_sum_sq / self._sample_count) - (mean ** 2)
        stddev   = np.sqrt(np.maximum(variance, 0))
        return mean.tolist(), stddev.tolist()

    async def run(self) -> None:
        """Main read loop - throttled to fixed interval."""

        self._running = True

        # Start reading from slightly behind current position
        now                = self._reader.timestamp()
        safety_margin      = 500  # frames
        self._next_read_ts = now - safety_margin

        _logger.info("SubprocessLiveReader started, initial ts: %d", self._next_read_ts)
        broadcast_count = 0

        # Interval in seconds
        interval = self.BROADCAST_INTERVAL_MS / 1000.0

        # Drift compensation: track when each iteration should start
        loop_scheduled_time = time.monotonic()

        while self._running:
            # Check for debugger pause via heartbeat
            if self._check_heartbeat_stale():
                await asyncio.sleep(0.01)  # 10ms pause check interval
                # Reset scheduled time when resuming from pause to avoid catching up too fast
                loop_scheduled_time = time.monotonic()
                continue

            try:
                await self._do_broadcast()
                broadcast_count += 1

                if broadcast_count == 1 or broadcast_count % 100 == 0:
                    now = self._reader.timestamp()
                    lag = now - self._next_read_ts if self._next_read_ts else 0
                    _logger.info("Broadcast #%d, lag=%d frames, spikes_total=%d", broadcast_count, lag, self._spikes_broadcast)

            except Exception as e:
                _logger.debug("Broadcast error: %s", e)

            # Drift compensation: advance scheduled time and calculate sleep to stay on schedule
            loop_scheduled_time += interval
            now                  = time.monotonic()
            drift                = now - loop_scheduled_time
            sleep_time           = max(0.01, interval - drift)
            await asyncio.sleep(sleep_time)

    def _check_heartbeat_stale(self) -> bool:
        """
        Check if the main process heartbeat has gone stale (indicating debugger pause).

        Returns True if the heartbeat hasn't been updated in over 200ms, suggesting
        the main process is paused at a breakpoint.
        """
        heartbeat_ns = self._reader.heartbeat_ns
        if heartbeat_ns == 0:
            return False  # Not yet initialized

        current_ns = time.perf_counter_ns()
        elapsed_ns = current_ns - heartbeat_ns

        # If heartbeat hasn't updated in 200ms, consider it stale
        return elapsed_ns > STALE_THRESHOLD_NS

    async def _do_broadcast(self) -> None:
        """Read latest data and broadcast to clients."""
        extended_read_size = (
            self.CAPTURED_SPIKE_SAMPLES_BEFORE +
            self._read_size_frames +
            self.CAPTURED_SPIKE_SAMPLES_AFTER
        )

        now = self._reader.timestamp()
        buffer_start = self._reader.start_timestamp()

        # Track where we should read next sequentially
        # Handle various cases: initial read, falling behind, backward seeks
        max_lag = self._read_size_frames * 4  # Allow up to 4 chunks of lag before jumping
        # Threshold for detecting significant backward jump (e.g., seek) vs just catching up
        backward_jump_threshold = self._read_size_frames * 2
        seek_detected = False

        if self._next_read_ts is None:
            # First read - start slightly behind live
            self._next_read_ts = now - extended_read_size - 100
        elif self._next_read_ts > now + backward_jump_threshold:
            # Significant backward jump detected (e.g., seek backward in playback)
            # Reset to slightly behind the new current position
            _logger.debug("Backward jump: next_read=%d, now=%d, resetting", self._next_read_ts, now)
            self._next_read_ts = now - extended_read_size - 100
            seek_detected = True
        elif self._next_read_ts > now:
            # Minor drift ahead (e.g., catching up or paused) - wait for data
            return
        elif now - self._next_read_ts > max_lag:
            # We've fallen too far behind, jump to near-live
            old_ts = self._next_read_ts
            self._next_read_ts = now - extended_read_size - 100
            _logger.debug("Jumped from %d to %d (was %d frames behind)", old_ts, self._next_read_ts, now - old_ts)
            seek_detected = True

        # Reset stats and notify on seek
        if seek_detected:
            self.reset_stats()
            if self._on_seek_detected:
                self._on_seek_detected()

        # Ensure we don't read before buffer start
        self._next_read_ts = max(self._next_read_ts, buffer_start + 100)

        from_ts = self._next_read_ts - self.CAPTURED_SPIKE_SAMPLES_BEFORE

        try:
            extended_frames = self._reader.read(extended_read_size, from_ts)
        except ValueError:
            return  # Data not ready yet

        this_read_ts = self._next_read_ts
        self._next_read_ts += self._read_size_frames  # Advance for next read

        start = self.CAPTURED_SPIKE_SAMPLES_BEFORE
        end = start + self._read_size_frames
        frames = extended_frames[start:end]

        # Update stats
        self._channel_sum    += frames.sum(axis=0)
        self._channel_sum_sq += (frames.astype(np.float64) ** 2).sum(axis=0)
        self._sample_count   += len(frames)

        # Get spikes and stims
        spikes = self._reader.read_spikes(self._read_size_frames, this_read_ts)
        to_ts  = this_read_ts + self._read_size_frames
        stims  = self._reader.read_stims(this_read_ts, to_ts)

        # Get datastream events
        datastream_events = self._reader.read_datastream_events(this_read_ts, to_ts)

        self._spikes_broadcast += len(spikes)  # Debug counter

        # Add spike samples
        transposed = extended_frames.T.copy()
        for spike in spikes:
            if spike.samples is not None and len(spike.samples) > 0:
                spike.samples = (spike.samples.astype(np.float32) - spike.channel_mean_sample) * 0.195
            else:
                samples_start = spike.timestamp - this_read_ts
                samples_end   = samples_start + self.CAPTURED_SPIKE_SAMPLES
                if 0 <= samples_start < transposed.shape[1] - self.CAPTURED_SPIKE_SAMPLES:
                    samples       = transposed[spike.channel, samples_start:samples_end]
                    spike.samples = (samples.astype(np.float32) - spike.channel_mean_sample) * 0.195

        # Broadcast (synchronously await to provide backpressure)
        await self._broadcast_sync(this_read_ts, frames, spikes, stims, datastream_events)

    async def _broadcast_sync(
            self,
            frames_ts        : int,
            frames           : np.ndarray,
            spikes           : list[Spike],
            stims            : list[Stim],
            datastream_events: list[DataStreamEventRecord],
        ) -> None:
        """Synchronous broadcast - waits for completion before returning."""
        # Call the server's broadcast method directly
        # The server should have set up an async-compatible callback
        self._on_new_frames(frames_ts, frames, spikes, stims, datastream_events)
        # Yield to let any scheduled tasks run
        await asyncio.sleep(0)

    def stop(self) -> None:
        self._running = False

class WebSocketProcessManager:
    """
    Manager for WebSocket subprocess.

    Usage:
        manager = WebSocketProcessManager(buffer_name, fps, channels)
        manager.start()
        # ... do work ...
        manager.stop()
    """

    def __init__(
        self,
        buffer_name      : str,
        frames_per_second: int,
        channel_count    : int,
        port             : int  = 1025,
        host             : str  = "127.0.0.1",
        serve_vis        : bool = True,
        app_html         : str | None = None,
    ):
        self._config = SubprocessConfig(
            buffer_name       = buffer_name,
            frames_per_second = frames_per_second,
            channel_count     = channel_count,
            port              = port,
            host              = host,
            serve_vis         = serve_vis,
            app_html          = app_html,
        )
        self._process      : BaseProcess | None                                           = None
        self._ready_queue  : Queue[dict[str, Any]] | None                                 = None
        self._command_queue: Queue[AttributeResetCommand | AttributeUpdateCommand] | None = None
        self._web_port     : int | None                                                   = None
        self._web_url      : str | None                                                   = None
        self._app_url      : str | None                                                   = None

    @property
    def port(self) -> int:
        return self._config.port

    @property
    def web_port(self) -> int | None:
        return self._web_port

    @property
    def web_url(self) -> str | None:
        return self._web_url

    @property
    def app_url(self) -> str | None:
        return self._app_url

    def _check_port_available(self, timeout_sec: float = 2.0) -> bool:
        """Wait for port to become available."""

        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind((self._config.host, self._config.port))
                    return True
            except OSError:
                time.sleep(0.1)
        return False

    def _wait_for_port_listening(self, timeout_sec: float = 5.0) -> bool:
        """Wait for port to start accepting connections."""

        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            # Check if process is still alive
            if self._process is None or not self._process.is_alive():
                return False
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect((self._config.host, self._config.port))
                    return True
            except (OSError, ConnectionRefusedError):
                time.sleep(0.1)
        return False

    def start(self, timeout_sec: float = 10.0) -> None:
        """Start the WebSocket subprocess."""
        if self._process is not None and self._process.is_alive():
            _logger.info("WebSocket subprocess already running")
            return

        # Wait for port to be available (previous subprocess may still be shutting down)
        port_available = self._check_port_available(timeout_sec=2.0)
        if not port_available:
            _logger.warning("Port %d still in use, attempting to continue anyway", self._config.port)

        # Use spawn context explicitly for clean subprocess
        ctx = multiprocessing.get_context('spawn')

        self._ready_queue   = ctx.Queue()
        self._command_queue = ctx.Queue()
        self._process       = ctx.Process(
            target = _run_websocket_subprocess,
            args   = (self._config, self._ready_queue, self._command_queue),
            daemon = True,
            name   = "cl-websocket-subprocess",
        )
        _logger.info("Starting subprocess...")
        self._process.start()
        _logger.info("Subprocess started, PID=%d, waiting for server to be ready...", self._process.pid)

        # First, wait for a message from the subprocess (error or ready)
        # This is more reliable than checking port connectivity
        start_time = time.monotonic()
        ready_received = False
        error_message = None

        while (time.monotonic() - start_time) < timeout_sec:
            # Check if process is still alive
            if not self._process.is_alive():
                exitcode = self._process.exitcode
                _logger.error("WebSocket subprocess crashed with exit code %d", exitcode)

                # Try to get any error message from queue
                try:
                    result = self._ready_queue.get(timeout=0.1)
                    if result.get("status") == "error":
                        error_message = result.get("error", "Unknown error")
                except Exception:
                    _logger.warning("Failed to get error message from subprocess after crash")

                if error_message:
                    raise RuntimeError(f"WebSocket subprocess initialization failed: {error_message}")
                else:
                    raise RuntimeError(f"WebSocket subprocess crashed with exit code {exitcode}")

            # Try to get ready message from queue (non-blocking)
            try:
                result = self._ready_queue.get(timeout=0.1)
                if result.get("status") == "error":
                    error_message = result.get("error", "Unknown error")
                    raise RuntimeError(f"WebSocket subprocess initialization failed: {error_message}")
                elif result.get("status") == "ready":
                    self._web_port = result.get("web_port")
                    self._web_url  = result.get("web_url")
                    self._app_url  = result.get("app_url")
                    ready_received = True
                    break
            except Exception as e:
                _logger.debug("Waiting for subprocess ready signal... (%s)", e)

            time.sleep(0.1)

        if not ready_received:
            elapsed = time.monotonic() - start_time
            _logger.error("WebSocket subprocess did not signal ready after %.1fs", elapsed)

            # Check if process is still alive
            if self._process.is_alive():
                _logger.error("Subprocess still alive (PID=%d) but didn't send ready message", self._process.pid)
                # Try to get error from queue one more time
                try:
                    result = self._ready_queue.get(timeout=0.5)
                    if result.get("status") == "error":
                        error_message = result.get("error", "Unknown error")
                        raise RuntimeError(f"WebSocket subprocess initialization failed: {error_message}")
                except Exception as e:
                    _logger.warning("Failed to get error message from subprocess after timeout: %s", e)

                raise RuntimeError("WebSocket subprocess failed to start: no ready signal received")
            else:
                raise RuntimeError("WebSocket subprocess crashed before sending ready signal")

        _logger.info("WebSocket subprocess started on port %d", self._config.port)
        if self._web_url:
            _logger.info("Data visualization at %s", self._web_url)
        if self._app_url:
            _logger.info("Application visualisation at %s", self._app_url)

    def send_attribute_update(self, stream_name: str, updated_attributes: dict) -> None:
        """Send an attribute update command to the subprocess."""
        if self._command_queue is not None:
            with contextlib.suppress(Exception):
                self._command_queue.put_nowait(
                    AttributeUpdateCommand(
                        stream_name        = stream_name,
                        updated_attributes = updated_attributes
                    )
                )

    def send_attribute_reset(self, stream_name: str, attributes: dict) -> None:
        """Send an attribute reset command to the subprocess."""
        if self._command_queue is not None:
            with contextlib.suppress(Exception):
                self._command_queue.put_nowait(
                    AttributeResetCommand(
                        stream_name = stream_name,
                        attributes  = attributes
                    )
                )

    def stop(self) -> None:
        """Stop the WebSocket subprocess."""
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)  # Wait for kill to complete
            self._process = None

        if self._ready_queue is not None:
            self._ready_queue = None

        _logger.info("WebSocket subprocess stopped")

    def is_alive(self) -> bool:
        """Check if subprocess is running."""
        return self._process is not None and self._process.is_alive()
