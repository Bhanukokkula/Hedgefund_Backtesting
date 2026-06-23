"""GLASSBOX: a glass-box cross-sectional factor backtester.

Core invariant (quoted verbatim, the spine of the whole codebase):

    No computation may read any datum whose knowable-date is later than the
    simulation's current as-of clock.

Every data access in this codebase goes through an AsOfAccessor bound to a
single monotonic simulation clock. There is no API that hands strategy code
a full, future-inclusive series. See glassbox.engine.asof for the contract
and tests/test_asof_adversarial.py for proof the engine refuses to cheat.
"""

__version__ = "0.1.0"
