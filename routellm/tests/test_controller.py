import unittest
from unittest.mock import patch

with patch("openai.OpenAI", return_value=object()):
    from routellm.controller import Controller, RoutingError


class ControllerModelNameTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(
            routers=[], strong_model="strong", weak_model="weak"
        )

    def test_parse_router_model_name(self):
        self.assertEqual(
            self.controller._parse_model_name("router-random-0.7"),
            ("random", 0.7),
        )

    def test_reject_malformed_model_names(self):
        invalid_names = (
            "x",
            "router-random",
            "routerjunk-random-0.5",
            "router--0.5",
        )
        for model in invalid_names:
            with self.subTest(model=model):
                with self.assertRaises(RoutingError):
                    self.controller._parse_model_name(model)

    def test_reject_non_numeric_threshold(self):
        with self.assertRaises(RoutingError):
            self.controller._parse_model_name("router-random-not-a-number")


if __name__ == "__main__":
    unittest.main()
