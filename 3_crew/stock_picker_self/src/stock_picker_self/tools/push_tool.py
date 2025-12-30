from crewai.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field, field_validator
import os
import requests


class PushNotification(BaseModel):
    """A message to be sent to the user"""
    message: str = Field(..., description="The message to be sent to the user.")
    
    @field_validator('message', mode='before')
    @classmethod
    def normalize_message(cls, v: Any) -> str:
        """Normalize message input - handle cases where dict is passed instead of string"""
        if isinstance(v, dict):
            # If it's a dict (e.g., Field metadata), try to extract the actual value
            # This can happen if schema info is passed instead of the value
            if 'message' in v:
                v = v['message']
            elif 'description' in v:
                # If only Field metadata is present, convert to string representation
                v = str(v)
            else:
                v = str(v)
        if not isinstance(v, str):
            v = str(v)
        return v

class PushNotificationTool(BaseTool):
    name: str = "Send a Push Notification"
    description: str = (
        "This tool is used to send a push notification to the user."
    )
    args_schema: Type[BaseModel] = PushNotification

    def _run(self, message: str) -> str:
        pushover_user = os.getenv("PUSHOVER_USER")
        pushover_token = os.getenv("PUSHOVER_TOKEN")
        pushover_url = "https://api.pushover.net/1/messages.json"

        print(f"Push: {message}")
        payload = {"user": pushover_user, "token": pushover_token, "message": message}
        requests.post(pushover_url, data=payload)
        return '{"notification": "ok"}'