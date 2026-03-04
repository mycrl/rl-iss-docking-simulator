"""
Browser automation layer for the SpaceX ISS Docking Simulator.

Supports two operating modes:

**Managed mode** (``launch=True``)
    Playwright launches its own Chromium browser, navigates to the simulator
    URL, and manages the full browser lifecycle.  No manual Chrome startup is
    required::

        browser = SimulatorBrowser(launch=True)
        browser.connect()

**CDP mode** (default, ``launch=False``)
    Playwright connects to an already-opened Chrome instance via the Chrome
    DevTools Protocol.  Start Chrome manually first::

        google-chrome --remote-debugging-port=9222 https://iss-sim.spacex.com/

    Then::

        browser = SimulatorBrowser()   # or SimulatorBrowser(cdp_url="http://localhost:9222")
        browser.connect()

.. note::
    When running in managed mode (``launch=True``), the simulator page requires
    a manual START interaction before gameplay begins.  After the page finishes
    loading, :meth:`SimulatorBrowser.connect` (and :meth:`SimulatorBrowser.reset`
    on each episode) will pause and prompt::

        Simulator page loaded. Click START in the browser, then press [y] + Enter:

    Type ``y`` and press Enter to proceed.  This keeps the browser visible so
    the operator can click the simulator's own START button first.
"""

import logging
import re
import time
from typing import Optional

from playwright.sync_api import Browser, Page, sync_playwright

logger = logging.getLogger(__name__)


class SimulatorBrowser:
    """
    Controls the SpaceX ISS Docking Simulator via Playwright.

    Two modes are supported — select via the *launch* constructor parameter:

    * **Managed mode** (``launch=True``): Playwright starts its own Chromium
      browser, navigates to :attr:`SIMULATOR_URL`, and owns the lifecycle.
      The browser is closed when :meth:`disconnect` is called.
    * **CDP mode** (``launch=False``, default): Playwright connects to an
      already-running Chrome instance via the Chrome DevTools Protocol endpoint
      at *cdp_url*.

    Parameters
    ----------
    launch:
        When ``True``, launch a new browser automatically.
        When ``False`` (default), connect to an existing browser via CDP.
    headless:
        Only used when ``launch=True``.  If ``True``, the browser runs without
        a visible window (headless).  Defaults to ``False`` so the simulator
        UI is visible during training.
    cdp_url:
        URL of the Chrome remote-debugging endpoint used in CDP mode
        (default: ``http://localhost:9222``).  Ignored when ``launch=True``.
    page_load_timeout:
        Seconds to wait for the simulator page to load before raising a
        timeout error.
    """

    CDP_URL: str = "http://localhost:9222"
    SIMULATOR_URL: str = "https://iss-sim.spacex.com/"

    # ------------------------------------------------------------------
    # CSS selectors for control buttons.
    # ------------------------------------------------------------------
    BUTTON_SELECTORS: dict[str, str] = {
        "translate_forward":  "#translate-forward-button",
        "translate_backward": "#translate-backward-button",
        "translate_up":       "#translate-up-button",
        "translate_down":     "#translate-down-button",
        "translate_left":     "#translate-left-button",
        "translate_right":    "#translate-right-button",
        "roll_left":          "#roll-left-button",
        "roll_right":         "#roll-right-button",
        "pitch_up":           "#pitch-up-button",
        "pitch_down":         "#pitch-down-button",
        "yaw_left":           "#yaw-left-button",
        "yaw_right":          "#yaw-right-button",
        "toggle_translation": "#toggle-translation",
        "toggle_rotation":    "#toggle-rotation",
    }

    # ------------------------------------------------------------------
    # CSS selectors for state readout elements.
    # The simulator DOM uses a different child-element structure per field:
    #   x, y, z       — single div inside their container (#x-range, etc.)
    #   roll/yaw/pitch — first child div = angle (°); second = angular rate (°/s)
    #   range          — second child div = distance to port (m)
    #   rate           — second child div = approach rate (m/s)
    # ------------------------------------------------------------------
    STATE_SELECTORS: dict[str, str] = {
        "x":          "#x-range div",
        "y":          "#y-range div",
        "z":          "#z-range div",
        "roll":       "#roll div:nth-child(1)",
        "roll_rate":  "#roll div:nth-child(2)",
        "range":      "#range div:nth-child(2)",
        "yaw":        "#yaw div:nth-child(1)",
        "yaw_rate":   "#yaw div:nth-child(2)",
        "rate":       "#rate div:nth-child(2)",
        "pitch":      "#pitch div:nth-child(1)",
        "pitch_rate": "#pitch div:nth-child(2)",
    }

    def __init__(
        self,
        launch: bool = False,
        headless: bool = False,
        cdp_url: str = CDP_URL,
        page_load_timeout: float = 30.0,
    ) -> None:
        self._launch = launch
        self._headless = headless
        self._cdp_url = cdp_url
        self._page_load_timeout = page_load_timeout
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the simulator.

        In **managed mode** (``launch=True``), Playwright launches a new
        Chromium browser and navigates directly to :attr:`SIMULATOR_URL`.

        In **CDP mode** (``launch=False``, default), Playwright connects to an
        already-running Chrome instance via the DevTools Protocol endpoint
        specified by ``cdp_url``.
        """
        self._playwright = sync_playwright().start()

        if self._launch:
            self._browser = self._playwright.chromium.launch(headless=self._headless)
            context = self._browser.new_context()
            self._page = context.new_page()
            self._page.goto(
                self.SIMULATOR_URL,
                wait_until="networkidle",
                timeout=int(self._page_load_timeout * 1_000),
            )
            self._wait_for_user_start()
            logger.info("Launched browser and navigated to %s", self.SIMULATOR_URL)
        else:
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
        self._wait_for_user_start()

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
            Keys: ``x``, ``y``, ``z``,
            ``roll``, ``roll_rate``,
            ``range``,
            ``yaw``, ``yaw_rate``,
            ``rate``,
            ``pitch``, ``pitch_rate``.

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
            # The DOM includes units (e.g. "200.0 m", "15.0°", "0.039 m/s").
            # Extract just the leading numeric token (with optional sign).
            match = re.search(r"-?\d+\.?\d*", text)
            try:
                state[key] = float(match.group()) if match else 0.0
            except (ValueError, AttributeError):
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

    def _wait_for_user_start(self) -> None:
        """Pause and wait for the operator to confirm the simulator has started.

        Prints a prompt to stdout and blocks until the user types ``y`` (or
        ``Y``) and presses Enter.  This gives the operator time to click the
        simulator's own START button in the browser before training proceeds.
        """
        while True:
            answer = input(
                "Simulator page loaded. Click START in the browser, "
                "then press [y] + Enter to continue: "
            ).strip().lower()
            if answer == "y":
                break
            print("Please type 'y' and press Enter to continue.")

    def _require_page(self) -> None:
        """Raise :class:`RuntimeError` if not yet connected."""
        if self._page is None:
            raise RuntimeError("Not connected. Call connect() first.")
