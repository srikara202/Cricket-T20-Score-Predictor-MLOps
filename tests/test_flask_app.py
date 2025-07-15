import unittest
import json
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # create a test client for our Flask app
        cls.client = app.test_client()

    def test_home_page(self):
        """Landing page should load and contain our new title."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # make sure you updated your HTML <title> to “T20 Score Predictor”
        self.assertIn(b'<title>T20 Score Predictor</title>', response.data)

    def test_predict_endpoint(self):
        """POST /predict should return a JSON with a numeric 'predicted_score'."""
        # craft a minimal valid payload
        payload = {
            "current_score": 80,
            "balls_left": 60,
            "wickets_left": 6,
            "crr": 8.0,
            "last_five": 30
        }
        response = self.client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

        # parse JSON
        data = response.get_json()
        self.assertIsInstance(data, dict, "Response must be JSON")
        self.assertIn('predicted_score', data)

        # value should be a number (int or float)
        val = data['predicted_score']
        self.assertTrue(isinstance(val, (int, float)),
                        f"predicted_score should be numeric, got {type(val)}")

        # and of a plausible magnitude
        self.assertGreaterEqual(val, 0, "predicted_score should be non-negative")
        self.assertLess(val, 500, "predicted_score seems unreasonably high")

if __name__ == '__main__':
    unittest.main()
