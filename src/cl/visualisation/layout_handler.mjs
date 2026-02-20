/**
 * Layout Handler for Visualisations
 *
 * Automatically detects the Jupyter environment and repositions visualisations
 * to a right-hand side panel when running in Jupyter Notebook or JupyterLab (if enabled via optional arg).
 * In VS Code, visualisations remain in their default position (below the cell).
 */

class LayoutHandler {
    static PANEL_ID                    = 'cl-visualiser-panel';
    static PANEL_DIV_ID                = 'cl-visualiser-panel-div';
    static PANEL_CONTAINER_ID          = 'cl-visualiser-panel-container';
    static STYLE_ID                    = 'cl-visualiser-panel-styles';
    static DEFAULT_PANEL_WIDTH_PERCENT = 20;
    static MIN_PANEL_WIDTH_PERCENT     = 20;
    static MAX_PANEL_WIDTH_PERCENT     = 80;
    static STORAGE_KEY                 = 'cl-visualiser-panel-width';

    constructor() {
        // Initialize global active wrapper tracking on window object
        if (typeof window.clActiveWrapper === 'undefined') {
            window.clActiveWrapper = null;
        }
        this.dragState = {
            isDragging: false,
            startX: 0,
            startPanelWidth: 0,
            containerWidth: 0,
            handle: null
        };

        this.setupGlobalListeners();
    }

    setupGlobalListeners() {
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        document.addEventListener('mouseup', (e) => this.handleMouseUp(e));
    }

    handleMouseMove(e) {
        if (!this.dragState.isDragging) return;

        const deltaX = this.dragState.startX - e.clientX;
        const deltaPercent = (deltaX / this.dragState.containerWidth) * 100;
        let newPanelWidth = this.dragState.startPanelWidth + deltaPercent;

        // Clamp to min/max
        newPanelWidth = Math.max(
            LayoutHandler.MIN_PANEL_WIDTH_PERCENT,
            Math.min(LayoutHandler.MAX_PANEL_WIDTH_PERCENT, newPanelWidth)
        );

        this.updatePanelWidth(newPanelWidth);
        e.preventDefault();
    }

    handleMouseUp() {
        if (!this.dragState.isDragging) return;

        this.dragState.isDragging = false;
        if (this.dragState.handle) {
            this.dragState.handle.classList.remove('dragging');
        }

        // Re-enable pointer events on all iframes
        document.querySelectorAll('iframe').forEach(iframe => {
            iframe.style.pointerEvents = '';
        });

        document.body.style.cursor = '';
        document.body.style.userSelect = '';

        // Force a layout recalculation to ensure ResizeObserver fires in iframes
        requestAnimationFrame(() => {
            const panel = document.getElementById(LayoutHandler.PANEL_ID);
            if (panel) {
                // Trigger a reflow by reading offsetHeight
                void panel.offsetHeight;
            }
        });
    }

    detectEnvironment() {
        // VS Code Jupyter
        if (typeof location !== 'undefined' && location.protocol === 'vscode-webview:') {
            return 'vscode';
        }

        // JupyterLab 4.x - check for JupyterLab-specific elements
        if (document.querySelector('.jp-Notebook') ||
            document.body.classList.contains('jp-LabShell') ||
            document.querySelector('#jp-top-panel')) {
            return 'jupyterlab';
        }

        // Classic Jupyter Notebook 6.x - check for notebook-specific elements
        // User specified: div#site > div#ipython-main-app
        if (document.querySelector('#site #ipython-main-app') ||
            document.getElementById('notebook') ||
            document.body.classList.contains('notebook_app')) {
            return 'notebook';
        }

        return 'unknown';
    }

    getSavedPanelWidth() {
        try {
            const saved = localStorage.getItem(LayoutHandler.STORAGE_KEY);
            if (saved) {
                const width = parseFloat(saved);
                if (!isNaN(width) && width >= LayoutHandler.MIN_PANEL_WIDTH_PERCENT && width <= LayoutHandler.MAX_PANEL_WIDTH_PERCENT) {
                    return width;
                }
            }
        } catch (e) {
            // localStorage might not be available
        }
        return LayoutHandler.DEFAULT_PANEL_WIDTH_PERCENT;
    }

    savePanelWidth(width) {
        try {
            localStorage.setItem(LayoutHandler.STORAGE_KEY, width.toString());
        } catch (e) {
            // localStorage might not be available
        }
    }

    injectStyles() {
        if (document.getElementById(LayoutHandler.STYLE_ID)) {
            return; // Already injected
        }

        const panelWidth = this.getSavedPanelWidth();
        const notebookWidth = 100 - panelWidth;

        // Always inject styles to ensure correct widths are applied
        const styles = document.createElement('style');
        styles.id = LayoutHandler.STYLE_ID;
        styles.textContent = `
            .${LayoutHandler.PANEL_CONTAINER_ID} > .cl-notebook-area {
                flex: 0 0 ${notebookWidth}%;
            }
            #${LayoutHandler.PANEL_ID} {
                flex: 0 0 calc(${panelWidth}% - 6px);
                overflow-y: scroll;
            }
        `;
        document.head.appendChild(styles);
    }

    updatePanelWidth(panelWidthPercent) {
        const notebookWidth = 100 - panelWidthPercent;

        const panel = document.getElementById(LayoutHandler.PANEL_ID);
        const container = document.querySelector(`.${LayoutHandler.PANEL_CONTAINER_ID}`);
        const notebookArea = container?.querySelector('.cl-notebook-area');

        if (panel) {
            panel.style.flex = `0 0 calc(${panelWidthPercent}% - 6px)`;
        }
        if (notebookArea) {
            notebookArea.style.flex = `0 0 ${notebookWidth}%`;
        }

        this.savePanelWidth(panelWidthPercent);
    }

    createResizeHandle() {
        const handle = document.createElement('div');
        handle.className = 'cl-panel-resize-handle';

        handle.addEventListener('mousedown', (e) => {
            this.dragState.isDragging = true;
            this.dragState.handle = handle;
            handle.classList.add('dragging');
            this.dragState.startX = e.clientX;

            const container = document.querySelector(`.${LayoutHandler.PANEL_CONTAINER_ID}`);
            const panel = document.getElementById(LayoutHandler.PANEL_ID);

            if (container && panel) {
                this.dragState.containerWidth = container.getBoundingClientRect().width;
                this.dragState.startPanelWidth = (panel.getBoundingClientRect().width / this.dragState.containerWidth) * 100;
            }

            // Disable pointer events on all iframes to prevent them from capturing mouse events
            document.querySelectorAll('iframe').forEach(iframe => {
                iframe.style.pointerEvents = 'none';
            });

            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';

            e.preventDefault();
            e.stopPropagation();
        });

        return handle;
    }

    getOrCreatePanelForNotebook() {
        let panel = document.getElementById(LayoutHandler.PANEL_ID);
        if (panel) {
            return panel;
        }

        // Find the main app container
        const mainApp = document.getElementById('ipython-main-app');
        if (!mainApp) {
            console.warn('[CL Layout] Could not find #ipython-main-app');
            return null;
        }

        // Get the parent (should be #site)
        const site = mainApp.parentElement;
        if (!site) {
            console.warn('[CL Layout] Could not find parent of #ipython-main-app');
            return null;
        }

        // Save scroll position from #site before restructuring
        const savedScrollTop = site.scrollTop;
        const savedScrollLeft = site.scrollLeft;

        // Create the flex container
        const container = document.createElement('div');
        container.className = LayoutHandler.PANEL_CONTAINER_ID;

        // Wrap the main app in a notebook area div
        const notebookArea = document.createElement('div');
        notebookArea.className = 'cl-notebook-area';

        // Insert container where mainApp was
        site.insertBefore(container, mainApp);

        // Move mainApp into notebookArea
        notebookArea.appendChild(mainApp);

        // Add notebook area to container
        container.appendChild(notebookArea);

        // Create resize handle
        const resizeHandle = this.createResizeHandle();
        container.appendChild(resizeHandle);

        // Create the panel
        panel = document.createElement('div');
        panel.id = LayoutHandler.PANEL_ID;

        // TODO: Decide whether to keep header for the sidebar
        // panel.innerHTML = `
        //     <div class="cl-panel-header">
        //         <span>Visualisation Sidebar</span>
        //     </div>
        // `;
        
        // Create the div container for visualiser wrappers
        const panelDiv = document.createElement('div');
        panelDiv.id = LayoutHandler.PANEL_DIV_ID;
        panel.appendChild(panelDiv);
        
        container.appendChild(panel);

        // Transfer scroll position to the new scroll container (.cl-notebook-area)
        requestAnimationFrame(() => {
            notebookArea.scrollTop = savedScrollTop;
            notebookArea.scrollLeft = savedScrollLeft;
            console.log(`[CL Layout] Restored scroll position (${savedScrollLeft}, ${savedScrollTop}) for notebook area`);
        });

        return panel;
    }

    getOrCreatePanelForJupyterLab() {
        let panel = document.getElementById(LayoutHandler.PANEL_ID);
        if (panel) {
            return panel;
        }

        // Find the notebook area in JupyterLab
        const notebook = document.querySelector('.jp-Notebook');
        if (!notebook) {
            console.warn('[CL Layout] Could not find .jp-Notebook');
            return null;
        }

        // Find a suitable parent to wrap
        // In JupyterLab, the structure is typically:
        // .jp-NotebookPanel > .jp-Toolbar + .jp-NotebookPanel-notebook (contains .jp-Notebook)
        const notebookPanelNotebook = notebook.closest('.jp-NotebookPanel-notebook');
        const targetElement = notebookPanelNotebook || notebook;
        const parent = targetElement.parentElement;

        if (!parent) {
            console.warn('[CL Layout] Could not find parent for JupyterLab notebook');
            return null;
        }

        // Check if we already have a container (might happen on re-run)
        if (parent.classList.contains(LayoutHandler.PANEL_CONTAINER_ID)) {
            panel = document.getElementById(LayoutHandler.PANEL_ID);
            if (panel) return panel;
        }

        // Save scroll position from the original scroll container before restructuring
        const savedScrollTop = targetElement.scrollTop;
        const savedScrollLeft = targetElement.scrollLeft;

        // Create the flex container
        const container = document.createElement('div');
        container.className = LayoutHandler.PANEL_CONTAINER_ID;

        // Wrap the notebook area
        const notebookArea = document.createElement('div');
        notebookArea.className = 'cl-notebook-area';

        // Insert container where target was
        parent.insertBefore(container, targetElement);

        // Move target into notebookArea
        notebookArea.appendChild(targetElement);

        // Add notebook area to container
        container.appendChild(notebookArea);

        // Create resize handle
        const resizeHandle = this.createResizeHandle();
        container.appendChild(resizeHandle);

        // Create the panel
        panel = document.createElement('div');
        panel.id = LayoutHandler.PANEL_ID;
        panel.innerHTML = `
            <div class="cl-panel-header">
                <span>Visualisations</span>
            </div>
        `;
        
        // Create the div container for visualiser wrappers
        const panelDiv = document.createElement('div');
        panelDiv.id = LayoutHandler.PANEL_DIV_ID;
        panel.appendChild(panelDiv);
        
        container.appendChild(panel);

        // Transfer scroll position to the new scroll container (.cl-notebook-area)
        requestAnimationFrame(() => {
            notebookArea.scrollTop = savedScrollTop;
            notebookArea.scrollLeft = savedScrollLeft;
        });

        return panel;
    }

    deactivateWrapper(wrapper) {
        if (!wrapper) return;

        const iframe = wrapper.querySelector('iframe');
        if (iframe) {
            // Remove iframe key listener if it exists
            if (wrapper._iframeKeyHandler && iframe.contentWindow) {
                try {
                    iframe.contentWindow.document.removeEventListener('keydown', wrapper._iframeKeyHandler, true);
                    wrapper._iframeKeyHandler = null;
                } catch (e) {
                    console.debug('[CL Layout] Could not remove iframe key listener:', e);
                }
            }

            // Fire a final mousemove event with out-of-bounds coordinates
            // to allow the visualisation to clean up hover states, tooltips, etc.
            try {
                if (iframe.contentWindow) {
                    const mouseEvent = new MouseEvent('mousemove', {
                        clientX: -1,
                        clientY: -1,
                        bubbles: true,
                        cancelable: true
                    });
                    iframe.contentWindow.document.dispatchEvent(mouseEvent);
                }
            } catch (e) {
                // Ignore if cross-origin or other issues
                console.debug('[CL Layout] Could not dispatch cleanup mousemove event:', e);
            }

            iframe.style.pointerEvents = 'none';
        }
        wrapper.classList.remove('cl-interactive');

        if (window.clActiveWrapper === wrapper) {
            window.clActiveWrapper = null;
        }
    }

    activateWrapper(wrapper) {
        // Deactivate previously active wrapper
        if (window.clActiveWrapper && window.clActiveWrapper !== wrapper) {
            this.deactivateWrapper(window.clActiveWrapper);
        }

        const iframe = wrapper.querySelector('iframe');
        if (iframe) {
            iframe.style.pointerEvents = 'auto';
            // Focus the iframe so keyboard events work immediately
            iframe.focus();

            // Listen for Escape key inside the iframe
            try {
                if (iframe.contentWindow && !wrapper._iframeKeyHandler) {
                    const iframeKeyHandler = (e) => {
                        if (e.key === 'Escape') {
                            e.preventDefault();
                            e.stopPropagation();
                            this.deactivateWrapper(wrapper);
                        }
                    };
                    wrapper._iframeKeyHandler = iframeKeyHandler;
                    iframe.contentWindow.document.addEventListener('keydown', iframeKeyHandler, true);
                }
            } catch (e) {
                console.debug('[CL Layout] Could not add iframe key listener:', e);
            }
        }
        wrapper.classList.add('cl-interactive');
        window.clActiveWrapper = wrapper;
    }

    setupInteractionOverlay(wrapper) {
        // Create overlay
        const overlay = document.createElement('div');
        overlay.className = 'cl-interaction-overlay';
        overlay.innerHTML = '<span>Click to interact</span>';

        // Find the iframe
        const iframe = wrapper.querySelector('iframe');
        if (!iframe) return;

        // Block pointer events by default
        iframe.style.pointerEvents = 'none';

        // Add overlay to wrapper
        wrapper.appendChild(overlay);

        // Show overlay on mouseover (add hover class to wrapper for border highlight)
        wrapper.addEventListener('mouseenter', () => {
            if (!wrapper.classList.contains('cl-interactive')) {
                overlay.classList.add('visible');
                wrapper.classList.add('cl-hover');
            }
        });

        // Hide overlay on mouseout (if not active)
        wrapper.addEventListener('mouseleave', () => {
            if (!wrapper.classList.contains('cl-interactive')) {
                overlay.classList.remove('visible');
                wrapper.classList.remove('cl-hover');
            }
        });

        // Activate on overlay click
        overlay.addEventListener('click', (e) => {
            this.activateWrapper(wrapper);
            overlay.classList.remove('visible');
            wrapper.classList.remove('cl-hover');
            e.stopPropagation();
        });

        // Activate on pinch gesture (wheel event with ctrlKey = pinch-to-zoom on trackpad)
        const pinchHandler = (e) => {
            if (e.ctrlKey && !wrapper.classList.contains('cl-interactive')) {
                e.preventDefault(); // Prevent browser zoom

                this.activateWrapper(wrapper);
                overlay.classList.remove('visible');
                wrapper.classList.remove('cl-hover');

                // Forward the event to the now-active iframe
                const iframe = wrapper.querySelector('iframe');
                if (iframe && iframe.contentWindow) {
                    try {
                        const iframeRect = iframe.getBoundingClientRect();
                        const wheelEvent = new WheelEvent('wheel', {
                            deltaX: e.deltaX,
                            deltaY: e.deltaY,
                            deltaZ: e.deltaZ,
                            deltaMode: e.deltaMode,
                            ctrlKey: e.ctrlKey,
                            shiftKey: e.shiftKey,
                            altKey: e.altKey,
                            metaKey: e.metaKey,
                            bubbles: true,
                            cancelable: true,
                            clientX: e.clientX - iframeRect.left,
                            clientY: e.clientY - iframeRect.top
                        });
                        iframe.contentWindow.document.dispatchEvent(wheelEvent);
                    } catch (err) {
                        console.debug('[CL Layout] Could not forward wheel event to iframe:', err);
                    }
                }
            }
        };
        wrapper.addEventListener('wheel', pinchHandler, { passive: false });

        // Deactivate when clicking outside
        const deactivateHandler = (e) => {
            if (!wrapper.contains(e.target)) {
                this.deactivateWrapper(wrapper);
            }
        };

        // Deactivate when scrolling outside
        const scrollHandler = (e) => {
            // Only deactivate if the scroll event is outside the wrapper and the mouse is not over it
            if (!wrapper.contains(e.target) && !wrapper.matches(':hover')) {
                this.deactivateWrapper(wrapper);
            }
        };

        const keyHandler = (e) => {
            // Deactivate on Escape key (only if this wrapper is active)
            if (e.key === 'Escape' && window.clActiveWrapper === wrapper) {
                e.preventDefault();
                e.stopPropagation();
                this.deactivateWrapper(wrapper);
            }
        }

        // Store handler reference for cleanup
        wrapper._deactivateHandler = deactivateHandler;
        wrapper._scrollHandler = scrollHandler;
        wrapper._keyHandler = keyHandler;
        // Use capture phase (true) to catch clicks before they're stopped by CodeMirror/other elements
        document.addEventListener('click', deactivateHandler, true);
        // Also listen for mousedown as a backup in case click events are prevented
        document.addEventListener('mousedown', deactivateHandler, true);
        wrapper._deactivateHandlerMousedown = deactivateHandler;
        // Listen for scroll events (scroll doesn't bubble, so we need capture phase)
        document.addEventListener('scroll', scrollHandler, true);
        // Listen for keydown events to handle Escape key
        document.addEventListener('keydown', keyHandler, true);
    }

    moveVisualiserToPanel(elementId) {
        const environment = this.detectEnvironment();

        // Only reposition for Jupyter environments
        if (environment !== 'notebook' && environment !== 'jupyterlab') {
            return;
        }

        const visualiserDiv = document.getElementById(elementId);
        if (!visualiserDiv) {
            console.warn(`[CL Layout] Could not find visualiser element #${elementId}`);
            return;
        }

        // Inject styles if not already done
        this.injectStyles();

        // Get or create the panel
        let panel = null;
        if (environment === 'notebook') {
            panel = this.getOrCreatePanelForNotebook();
        } else if (environment === 'jupyterlab') {
            panel = this.getOrCreatePanelForJupyterLab();
        }

        if (!panel) {
            console.warn('[CL Layout] Could not create panel, keeping visualiser in place');
            return;
        }

        // Create a wrapper for this visualiser
        const wrapper = document.createElement('div');
        wrapper.className = 'cl-visualiser-wrapper';
        wrapper.dataset.sourceId = elementId;

        // Create a placeholder in the original location for cleanup detection
        // This allows us to detect when the cell is cleared/re-run
        const placeholder = document.createElement('div');
        placeholder.className = 'cl-visualiser-placeholder';
        placeholder.dataset.visualiserId = elementId;
        placeholder.style.display = 'none';
        visualiserDiv.parentNode.insertBefore(placeholder, visualiserDiv);
        visualiserDiv.parentNode.style.padding = '0 0.4em';

        // Move the actual visualiser element to the panel (not clone)
        // This preserves all JavaScript bindings, canvas contexts, etc.
        wrapper.appendChild(visualiserDiv);

        // Add to panel div (container for all wrappers)
        const panelDiv = document.getElementById(LayoutHandler.PANEL_DIV_ID);
        if (!panelDiv) {
            console.warn('[CL Layout] Could not find panel div, keeping visualiser in place');
            return;
        }
        panelDiv.appendChild(wrapper);

        // Setup interaction overlay
        this.setupInteractionOverlay(wrapper);

        // Force iframe to resize after being moved
        // Use requestAnimationFrame to ensure the DOM has updated
        requestAnimationFrame(() => {
            const iframe = visualiserDiv.querySelector('iframe');
            if (iframe && iframe.contentWindow) {
                // Force a layout recalculation
                void iframe.offsetHeight;

                // Dispatch resize event to iframe's window
                // This should trigger any ResizeObservers inside the iframe
                try {
                    iframe.contentWindow.dispatchEvent(new Event('resize'));
                } catch (e) {
                    // Ignore if cross-origin
                    console.warn('[CL Layout] Could not dispatch resize event to iframe:', e);
                }
            }
        });

        // Set up cleanup when the placeholder is removed (cell deleted/re-run)
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const node of mutation.removedNodes) {
                    if (node === placeholder || (node.contains && node.contains(placeholder))) {
                        // Placeholder was removed, clean up the panel copy
                        // Remove global click listener
                        if (wrapper._deactivateHandler) {
                            document.removeEventListener('click', wrapper._deactivateHandler, true);
                            document.removeEventListener('mousedown', wrapper._deactivateHandlerMousedown, true);
                        }
                        if (wrapper._scrollHandler) {
                            document.removeEventListener('scroll', wrapper._scrollHandler, true);
                        }
                        if (wrapper._keyHandler) {
                            document.removeEventListener('keydown', wrapper._keyHandler, true);
                        }

                        // Remove the wrapper from the panel
                        wrapper.remove();
                        observer.disconnect();

                        // Check if panel div is now empty
                        const panelDiv = document.getElementById(LayoutHandler.PANEL_DIV_ID);
                        const remainingVisualisers = panelDiv ? panelDiv.querySelectorAll('.cl-visualiser-wrapper') : [];
                        if (remainingVisualisers.length === 0) {
                            // Remove the entire panel structure and restore original DOM
                            const container = document.querySelector(`.${LayoutHandler.PANEL_CONTAINER_ID}`);
                            if (container) {
                                // Save scroll position before DOM restructuring
                                const notebookArea = document.querySelector('.cl-notebook-area');
                                const scrollTop = notebookArea ? notebookArea.scrollTop : 0;
                                const scrollLeft = notebookArea ? notebookArea.scrollLeft : 0;
                                const parent = container.parentElement;

                                if (notebookArea && parent) {
                                    // Move the notebook content back to its original position
                                    while (notebookArea.firstChild) {
                                        parent.insertBefore(notebookArea.firstChild, container);
                                    }
                                }

                                // Remove the container (includes panel and resize handle)
                                container.remove();

                                // Remove the dynamic styles if present
                                const dynamicStyles = document.getElementById(LayoutHandler.STYLE_ID);
                                if (dynamicStyles) {
                                    dynamicStyles.remove();
                                }

                                // Restore scroll position to the original scroll container
                                requestAnimationFrame(() => {
                                    parent.scrollTop = scrollTop;
                                    parent.scrollLeft = scrollLeft;
                                    console.log(`[CL Layout] Restored scroll position (${scrollLeft}, ${scrollTop}) after removing panel`);
                                });
                            }
                        }
                        return;
                    }
                }
            }
        });

        // Find the output container to observe (where the placeholder is)
        const outputContainer = placeholder.closest('.output_area, .jp-OutputArea-output, .output_wrapper');
        if (outputContainer && outputContainer.parentElement) {
            observer.observe(outputContainer.parentElement, { childList: true, subtree: true });
        } else {
            // Fallback: observe body
            observer.observe(document.body, { childList: true, subtree: true });
        }
    }
}

// Create instance
const layoutHandler = new LayoutHandler();

// Automatically move the current visualiser (iframe) to the panel
// The iframeId variable is injected by visualisation.py
if (typeof iframeId !== 'undefined') {
    // Use requestAnimationFrame to ensure the DOM is ready
    requestAnimationFrame(() => {
        layoutHandler.moveVisualiserToPanel(iframeId);
    });
}
