import unittest

from console_compat import configure_safe_stdio


class _ConfigurableStream:
    def __init__(self):
        self.errors = None

    def reconfigure(self, **kwargs):
        self.errors = kwargs.get("errors")


class _ImmutableStream:
    def reconfigure(self, **kwargs):
        raise ValueError("immutable")


class ConsoleCompatibilityTests(unittest.TestCase):
    def test_configures_redirected_streams_to_replace_unencodable_status_text(self):
        stream = _ConfigurableStream()

        configure_safe_stdio(stream)

        self.assertEqual("replace", stream.errors)

    def test_ignores_streams_that_cannot_be_reconfigured(self):
        configure_safe_stdio(object(), _ImmutableStream())


if __name__ == "__main__":
    unittest.main()
