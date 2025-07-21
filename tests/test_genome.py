import unittest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession

from genome import (
    create_user, authorize, get_json_data, get_filtered_data_using_token,
    select_users_post_by_id, get_total_count, generate_http_filters
)
from genome import DummyJsonProps

class DummyJsonTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local[*]").appName("TestApp").getOrCreate()

    @patch("genome.requests.post")
    def test_create_user_success(self, mock_post):
        mock_response = MagicMock(status_code=201)
        mock_post.return_value = mock_response
        create_user("Test", "User", "testuser", "123456")
        mock_post.assert_called_once()

    @patch("genome.requests.post")
    def test_authorize_success(self, mock_post):
        mock_post.return_value.json.return_value = {
            "accessToken": "abc123", "refreshToken": "def456"
        }
        access_token, refresh_token = authorize("testuser", "123456")
        self.assertEqual(access_token, "abc123")
        self.assertEqual(refresh_token, "def456")

    @patch("genome.requests.get")
    @patch("genome.spark.createDataFrame")
    def test_get_json_data(self, mock_create_df, mock_get):
        mock_get.return_value.json.return_value = {"users": [{"id": 1, "gender": "female"}]}
        mock_get.return_value.raise_for_status = lambda: None
        props = DummyJsonProps.FEMALE_USERS
        get_json_data("users", 0, 1, props)
        mock_create_df.assert_called_once()

    def test_generate_http_filters(self):
        filters = {"gender": "female", "status": "active"}
        result = generate_http_filters(filters)
        self.assertIn("key=gender", result)
        self.assertIn("value=female", result)
        self.assertIn("key=status", result)

    @patch("genome.requests.get")
    @patch("genome.spark.createDataFrame")
    def test_get_filtered_data_using_token(self, mock_create_df, mock_get):
        mock_get.return_value.json.return_value = {"users": [{"id": 1, "gender": "female"}]}
        mock_get.return_value.raise_for_status = lambda: None
        props = DummyJsonProps.FEMALE_USERS
        get_filtered_data_using_token("users", {"gender": "female"}, props, "abc123")
        mock_create_df.assert_called_once()

    @patch("genome.requests.get")
    def test_select_users_post_by_id(self, mock_get):
        mock_get.return_value.json.return_value = {"posts": [{"id": 1, "userId": 123}]}
        mock_get.return_value.raise_for_status = lambda: None
        result = select_users_post_by_id(123, DummyJsonProps.POSTS)
        self.assertIn("posts", {"posts": result})

    @patch("genome.requests.get")
    def test_get_total_count(self, mock_get):
        mock_get.return_value.json.return_value = {"total": 150}
        mock_get.return_value.raise_for_status = lambda: None
        count = get_total_count("users")
        self.assertEqual(count, 150)


if __name__ == "__main__":
    unittest.main()