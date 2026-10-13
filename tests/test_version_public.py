"""The version the installer and the offline bundle print is the release."""

import unittest

import extract


class VersionString(unittest.TestCase):
    def test_B7_the_version_the_installer_and_the_offline_bundle_print_is_the_release(self):
        self.assertEqual(extract.__version__, "5.0.7")


if __name__ == "__main__":
    unittest.main()
