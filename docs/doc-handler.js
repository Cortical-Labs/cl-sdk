/**
 * Adds a "Copy" button to code blocks
 */
const addCodeCopyButton = () => {
    const codeBlocks = document.querySelectorAll("div.pdoc-code");

    codeBlocks.forEach(pre => {
        // Strictly apply to div.pdoc-code > pre
        if (!pre.querySelector("pre")) { return; }

        // Clipboard API is not available if not https
        if (!(navigator.clipboard && window.isSecureContext)) { return; }

        // Create the button
        const button        = document.createElement("button");
        button.className    = "copy-button";
        button.textContent  = "Copy";
        button.type         = "button";

        // Add click event
        button.addEventListener("click", () => {
            const code = pre.querySelector("code").innerText;

            navigator.clipboard.writeText(code).then(() => {
                button.textContent = "Copied!";
                button.classList.add("copied");
                setTimeout(() => {
                    button.textContent = "Copy";
                    button.classList.remove("copied");
                }, 2000);
            });
        });

        // Append button to the pre block
        pre.appendChild(button);
    });
}

/**
 * Close the navbar when a link is clicked, only relevant on mobile.
 */
const closeNavbarOnLinkClick = () => {
    const navLinks    = document.querySelectorAll("nav a");
    const toggleState = document.querySelector("#togglestate");
    navLinks.forEach(link => {
        link.addEventListener("click", event => {
            // Navbar show state is handled by #togglestate.checked
            toggleState.checked = false;
        });
    });
}

document.addEventListener("DOMContentLoaded", () => {
    addCodeCopyButton();
    closeNavbarOnLinkClick();

    const nav = document.querySelector('nav.pdoc');
    if (nav) {
        nav.addEventListener('scroll', () => {
            nav.classList.toggle('is-scrolled', nav.scrollTop > 0);
        });
    }
});