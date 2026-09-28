import unittest
from routellm import openai_server


class TestOpenAIServer(unittest.TestCase):
    def test_app_and_routes_exist(self):
        self.assertIsNotNone(openai_server.app)
        route_paths = [route.path for route in openai_server.app.routes]
        self.assertIn("/v1/chat/completions", route_paths)
        self.assertIn("/health", route_paths)


if __name__ == "__main__":
    unittest.main()
