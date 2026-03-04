"""
Browser automation layer for the SpaceX ISS Docking Simulator.

Connects to an already-opened Chrome browser via the Chrome DevTools Protocol
(CDP) and provides methods to read simulator state and trigger control actions.

To use this module, start Chrome with remote debugging enabled::

    google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/

Then instantiate :class:`SimulatorBrowser`, call :meth:`connect`, and use
:meth:`read_state` / :meth:`click_action` to interact with the page.

.. note::
    The CSS selectors in :attr:`SimulatorBrowser.BUTTON_SELECTORS` and
    :attr:`SimulatorBrowser.STATE_SELECTORS` are placeholders.  Inspect the
    simulator page (F12 → DevTools) and replace each empty string with the
    correct CSS selector or DOM ID before running the agent.
"""

import logging
import time
from typing import Optional

from playwright.sync_api import Browser, Page, sync_playwright

logger = logging.getLogger(__name__)


class SimulatorBrowser:
    """
    Controls the SpaceX ISS Docking Simulator via Chrome DevTools Protocol.

    Parameters
    ----------
    cdp_url:
        URL of the Chrome remote-debugging endpoint
        (default: ``http://localhost:9222``).
    page_load_timeout:
        Seconds to wait for a page reload before raising a timeout error.
    """

    CDP_URL: str = "http://localhost:9222"
    SIMULATOR_URL: str = "https://iss-sim.spacex.com/"

    # ------------------------------------------------------------------
    # CSS selectors for control buttons.
    # TODO: Inspect the simulator page (DevTools → Elements) and replace
    #       each empty string with the correct CSS selector for that button.
    # ------------------------------------------------------------------
    BUTTON_SELECTORS: dict[str, str] = {
        "translate_forward":  "",  # TODO: e.g. "#translate-forward-button"
        "translate_backward": "",  # TODO: e.g. "#translate-backward-button"
        "translate_up":       "",  # TODO: e.g. "#translate-up-button"
        "translate_down":     "",  # TODO: e.g. "#translate-down-button"
        "translate_left":     "",  # TODO: e.g. "#translate-left-button"
        "translate_right":    "",  # TODO: e.g. "#translate-right-button"
        "roll_left":          "",  # TODO: e.g. "#roll-left-button"
        "roll_right":         "",  # TODO: e.g. "#roll-right-button"
        "pitch_up":           "",  # TODO: e.g. "#pitch-up-button"
        "pitch_down":         "",  # TODO: e.g. "#pitch-down-button"
        "yaw_left":           "",  # TODO: e.g. "#yaw-left-button"
        "yaw_right":          "",  # TODO: e.g. "#yaw-right-button"
    }

    # ------------------------------------------------------------------
    # CSS selectors for state readout elements.
    # TODO: Inspect the simulator page and replace each empty string with
    #       the CSS selector of the DOM element whose text content gives
    #       the numeric reading for that state variable.
    # ------------------------------------------------------------------
    STATE_SELECTORS: dict[str, str] = {
        "x":     "",  # TODO: e.g. "#x-error-number"    (metres)
        "y":     "",  # TODO: e.g. "#y-error-number"    (metres)
        "z":     "",  # TODO: e.g. "#z-error-number"    (metres)
        "roll":  "",  # TODO: e.g. "#roll-error-number" (degrees)
        "range": "",  # TODO: e.g. "#range-number"      (metres)
        "yaw":   "",  # TODO: e.g. "#yaw-error-number"  (degrees)
        "rate":  "",  # TODO: e.g. "#rate-number"       (m/s)
        "pitch": "",  # TODO: e.g. "#pitch-error-number"(degrees)
    }

    def __init__(
        self,
        cdp_url: str = CDP_URL,
        page_load_timeout: float = 30.0,
    ) -> None:
        self._cdp_url = cdp_url
        self._page_load_timeout = page_load_timeout
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to an already-opened Chrome instance via CDP."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.connect_over_cdp(self._cdp_url)

        contexts = self._browser.contexts
        if not contexts:
            raise RuntimeError(
                "No browser contexts found. "
                f"Open the simulator at {self.SIMULATOR_URL} first."
            )

        pages = contexts[0].pages
        if not pages:
            raise RuntimeError("No open pages found in the browser context.")

        # Prefer the simulator page; fall back to the first available page.
        self._page = next(
            (p for p in pages if self.SIMULATOR_URL in p.url),
            pages[0],
        )

        if self.SIMULATOR_URL not in self._page.url:
            logger.warning(
                "Simulator page not found; using page: %s", self._page.url
            )
        else:
            logger.info("Connected to simulator page: %s", self._page.url)

    def disconnect(self) -> None:
        """Close the CDP connection and release Playwright resources."""
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self, wait: float = 5.0) -> None:
        """Reload the simulator page to start a new episode.

        Parameters
        ----------
        wait:
            Extra seconds to sleep after the page finishes loading so that
            the simulator's initialisation animation can complete.
        """
        self._require_page()
        self._page.reload(
            wait_until="networkidle",
            timeout=int(self._page_load_timeout * 1_000),
        )
        time.sleep(wait)

    # ------------------------------------------------------------------
    # Actions & observations
    # ------------------------------------------------------------------

    def click_action(self, action_name: str) -> None:
        """Click the control button that corresponds to *action_name*.

        Parameters
        ----------
        action_name:
            One of the keys in :attr:`BUTTON_SELECTORS`.

        Raises
        ------
        ValueError
            If the selector for *action_name* has not been configured.
        """
        self._require_page()
        selector = self.BUTTON_SELECTORS.get(action_name)
        if not selector:
            raise ValueError(
                f"Selector for action '{action_name}' is not configured. "
                "Fill in BUTTON_SELECTORS with the correct CSS selectors."
            )
        self._page.click(selector)

    def read_state(self) -> dict[str, float]:
        """Read the current simulator state from the page DOM.

        Returns
        -------
        dict[str, float]
            Keys: ``x``, ``y``, ``z``, ``roll``, ``range``, ``yaw``,
            ``rate``, ``pitch``.

        Raises
        ------
        ValueError
            If any selector in :attr:`STATE_SELECTORS` has not been configured.
        """
        self._require_page()
        state: dict[str, float] = {}
        for key, selector in self.STATE_SELECTORS.items():
            if not selector:
                raise ValueError(
                    f"Selector for state '{key}' is not configured. "
                    "Fill in STATE_SELECTORS with the correct CSS selectors."
                )
            text = self._page.inner_text(selector).strip()
            try:
                state[key] = float(text)
            except ValueError:
                logger.warning(
                    "Could not parse '%s' for state key '%s'; defaulting to 0.0",
                    text,
                    key,
                )
                state[key] = 0.0
        return state

    # ------------------------------------------------------------------
    # Terminal-state helpers
    # ------------------------------------------------------------------

    def is_docked(self) -> bool:
        """Return ``True`` if the simulator shows a successful docking state.

        .. todo::
            Replace the body with the correct DOM check, e.g. detecting a
            success overlay element becoming visible.
        """
        return False  # TODO: implement DOM check for docking success

    def is_crashed(self) -> bool:
        """Return ``True`` if the simulator shows a collision/failure state.

        .. todo::
            Replace the body with the correct DOM check, e.g. detecting a
            failure banner or a red alert element becoming visible.
        """
        return False  # TODO: implement DOM check for crash/failure

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_page(self) -> None:
        """Raise :class:`RuntimeError` if not yet connected."""
        if self._page is None:
            raise RuntimeError("Not connected. Call connect() first.")
