window.addEventListener("keydown", (event) => {
    switch (event.key) {
        case "Escape":
            exitAllSingleFieldForms();
            break;
        }
    },
    true,
);

function exitAllSingleFieldForms() {
    const editables = document.querySelectorAll('.single-field .editable');
    const forms = document.querySelectorAll('.single-field form');
    for (let i = 0; i < editables.length; i++) {
        editables[i].classList.remove('hidden');
        forms[i].classList.add('hidden');
    }
}

window.addEventListener('DOMContentLoaded', () => {
    initSingleFieldForms();
    initFieldMultis();
});

export function initSingleFieldForms() {
    const containers = document.querySelectorAll('.single-field');
    for (const container of Array.from(containers)) {

        // Skip containers that have already been initialized
        if (container.dataset.initialized === 'true') {
            continue;
        }
        container.dataset.initialized = 'true';

        const editable = container.querySelector('.editable');
        const display = editable.querySelector('.value');
        const form = container.querySelector('form');
        const cancelButton = container.querySelector('.cancel');
        const input = form.querySelector('input, select, textarea');
        const formError = container.querySelector('.form-error');

        // note(jake): 2025-06-06
        // Remove success class after its animation finishes so it doesn't
        // accidentally get triggered again if the element is modified.
        editable.addEventListener('animationend', (e) => {
            editable.classList.remove('success');
        });

        if (cancelButton) {
            cancelButton.addEventListener('click', (e) => {
                e.preventDefault();
                editable.classList.remove('hidden');
                form.classList.add('hidden');
                formError.textContent = "";
            });
        }
        else {
            // If there's no cancel button, allow clicking outside the form to cancel.
            document.addEventListener('click', (e) => {
                if (!container.contains(e.target) && !form.classList.contains('hidden')) {
                    editable.classList.remove('hidden');
                    form.classList.add('hidden');
                    formError.textContent = "";
                }
            });
        }

        editable.addEventListener('click', () => {

            // note(jake): 2025-06-05
            // Close all other single-field-forms.
            const otherEditables = document.querySelectorAll('.single-field .editable');
            const otherForms = document.querySelectorAll('.single-field form');
            for (let i = 0; i < otherEditables.length; i++) {
                if (otherEditables[i] !== editable) {
                    otherEditables[i].classList.remove('hidden');
                    otherForms[i].classList.add('hidden');
                }
            }

            input.classList.remove('invalid');
            editable.classList.add('hidden');
            form.classList.remove('hidden');

            // For duration fields, populate with formatted display value instead of raw seconds
            if (form.dataset.fieldType === 'duration-seconds') {
                input.value = display.textContent.trim();
            } else {
                input.value = editable.textContent.trim();
            }

            input.focus();
            input.select();
        });

        form.addEventListener('submit', async (e) => {

            e.preventDefault();

            const restoreInteractivity = () => {
                // note(jake): 2025-06-06
                // Enable inputs again. The form is display: "none" so it's non
                // interactive but ready to be used again once it's revealed.
                input.removeAttribute('disabled');
                if (cancelButton)
                    cancelButton.removeAttribute('disabled');
                e.submitter.removeAttribute('disabled');
            }


            // note(jake): 2025-06-06
            // Send form data in the request body with proper Content-Type
            form.classList.add('submitting');

            // Handle JSON body forms (for PATCH/PUT requests)
            let requestBody;
            let contentType;
            let displayValue = input.value;

            // note(jake): 2025-06-06
            // We must disable inputs AFTER we have read the form data,
            // otherwise disabled inputs will not be included in the FormData.
            const formData = new FormData(form);

            input.setAttribute('disabled', 'disabled');
            if (cancelButton) cancelButton.setAttribute('disabled', 'disabled');
            e.submitter.setAttribute('disabled', 'disabled');

            // Block SSE updates to this field while waiting for server response
            display.dataset.pending = 'true';

            if (form.dataset.jsonBody === 'true') {
                const fieldName = form.dataset.fieldName;
                const fieldType = form.dataset.fieldType;
                const rawValue = form.dataset.rawValue;

                let parsedValue;

                // Parse duration format (e.g., "5m", "1h 30m", "1h 30m 20s", "90") into seconds
                // Supports: h/hr/hours, m/min/minutes, s/sec/seconds
                // Also supports: hh:mm:ss, mm:ss, or h:mm:ss formats
                if (fieldType === 'duration-seconds') {
                    const trimmedValue = input.value.trim();

                    // Try hh:mm:ss or mm:ss or h:mm:ss format first
                    const colonMatch = trimmedValue.match(/^(\d+):(\d{1,2})(?::(\d{1,2}))?$/);

                    if (colonMatch) {
                        if (colonMatch[3] !== undefined) {
                            // hh:mm:ss format
                            const hours = parseInt(colonMatch[1]);
                            const minutes = parseInt(colonMatch[2]);
                            const seconds = parseInt(colonMatch[3]);
                            parsedValue = hours * 3600 + minutes * 60 + seconds;
                        } else {
                            // mm:ss format
                            const minutes = parseInt(colonMatch[1]);
                            const seconds = parseInt(colonMatch[2]);
                            parsedValue = minutes * 60 + seconds;
                        }
                    } else {
                        // Try existing format: "5m", "1h 30m", "1h 30m 20s"
                        const durationMatch = trimmedValue.match(/^(?:(\d+)\s*(?:h|hr|hours?)\s*)?(?:(\d+)\s*(?:m|min|minutes?)\s*)?(?:(\d+)\s*(?:s|sec|seconds?))?$/i);

                        if (durationMatch && (durationMatch[1] || durationMatch[2] || durationMatch[3])) {
                            const hours = parseInt(durationMatch[1] || 0);
                            const minutes = parseInt(durationMatch[2] || 0);
                            const seconds = parseInt(durationMatch[3] || 0);
                            parsedValue = hours * 3600 + minutes * 60 + seconds;
                        } else {
                            // Try parsing as plain seconds (must be purely numeric)
                            if (/^\d+$/.test(trimmedValue)) {
                                parsedValue = parseInt(trimmedValue);
                            } else {
                                parsedValue = NaN;
                            }
                            if (isNaN(parsedValue) || parsedValue < 0) {
                                input.classList.add('invalid');
                                formError.textContent = "Invalid duration. Use format like '5m', '1h 30m', '1:30:00', '5:30', or '90' (seconds).";
                                restoreInteractivity();
                                return;
                            }
                        }
                    }

                    // Update display value to show formatted duration
                    const h = Math.floor(parsedValue / 3600);
                    const m = Math.floor((parsedValue % 3600) / 60);
                    const s = parsedValue % 60;

                    if (h > 0 && m > 0 && s > 0) {
                        displayValue = `${h}h ${m}m ${s}s`;
                    } else if (h > 0 && m > 0) {
                        displayValue = `${h}h ${m}m`;
                    } else if (h > 0 && s > 0) {
                        displayValue = `${h}h ${s}s`;
                    } else if (m > 0 && s > 0) {
                        displayValue = `${m}m ${s}s`;
                    } else if (h > 0) {
                        displayValue = `${h}h`;
                    } else if (m > 0) {
                        displayValue = `${m}m`;
                    } else {
                        displayValue = `${s}s`;
                    }
                } else {
                    parsedValue = input.value;
                }

                requestBody = JSON.stringify({ [fieldName]: parsedValue });
                contentType = 'application/json';
            } else {
                requestBody = formData;
                contentType = undefined; // Let browser set it for FormData
            }

            // Use data-http-method if available, otherwise fall back to form.method
            const httpMethod = (form.dataset.httpMethod || form.method || 'POST').toUpperCase();

            const fetchOptions = {
                method: httpMethod,
                headers: contentType ? { 'Content-Type': contentType } : {},
            };

            // Only add body for methods that support it (not GET or HEAD)
            if (httpMethod !== 'GET' && httpMethod !== 'HEAD') {
                fetchOptions.body = requestBody;
            }

            const res = await fetch(form.action, fetchOptions);

            // Clear pending flag now that we have a response
            delete display.dataset.pending;

            if (res.status >= 400) {
                input.classList.add('invalid');
                restoreInteractivity();
                if (res.status >= 500) {
                    formError.textContent = "Server error.";
                } else {
                    const errorData = await res.json().catch(() => ({}));
                    formError.textContent = errorData.error || "Invalid input.";
                }
                return;
            } else {
                formError.textContent = "";
                display.textContent = displayValue;
                display.dataset.lastUpdated = new Date().getTime();
                editable.classList.remove('success');
                requestAnimationFrame(() => {
                    editable.classList.add('success');
                });

                // Update raw value if it's a duration field
                if (form.dataset.fieldType === 'duration-seconds') {
                    const parsedValue = parseInt(JSON.parse(requestBody)[form.dataset.fieldName]);
                    form.dataset.rawValue = parsedValue;
                }

                // Dispatch custom event for successful form submission
                // This allows other modules to react to field updates (e.g., recalculating estimated times)
                const fieldUpdateEvent = new CustomEvent('singleFieldUpdated', {
                    bubbles: true,
                    detail: {
                        form: form,
                        fieldName: form.dataset.fieldName,
                        fieldType: form.dataset.fieldType,
                        newValue: form.dataset.fieldType === 'duration-seconds' ? parseInt(form.dataset.rawValue) : input.value,
                        displayValue: displayValue
                    }
                });
                form.dispatchEvent(fieldUpdateEvent);
            }

            // note(jake): 2025-06-06
            // Success or failure, exit the form.
            form.classList.add('hidden');
            editable.classList.remove('hidden');

            restoreInteractivity();

            form.classList.remove('submitting');
        });
    }
}

function initFieldMultis() {
    const fieldMultis = document.querySelectorAll('.field-multi');
    for (const fieldMulti of Array.from(fieldMultis)) {

        const addButton = fieldMulti.querySelector('.add-field');
        addButton.addEventListener('click', () => {
            fieldMultiAdd(fieldMulti);
        });

        const removeButtons = fieldMulti.querySelectorAll('.remove-field');
        for (const removeButton of Array.from(removeButtons)) {
            removeButton.addEventListener('click', () => {
                fieldMultiRemove(removeButton);
            });
        }
    }
}

function fieldMultiAdd(fieldMulti) {
    const fields = fieldMulti.querySelector('.field-rows');
    const field = fieldMulti.querySelector('.field-row');
    const newField = field.cloneNode(true);
    const input = newField.querySelector('input');
    const removeButton = newField.querySelector('.remove-field');
    removeButton.addEventListener('click', () => {
        fieldMultiRemove(removeButton);
    });
    input.value = "";
    fields.appendChild(newField);
}

function fieldMultiRemove(button) {
    const fieldRow = button.closest('.field-row');
    if (fieldRow) {
        fieldRow.remove();
    }
}