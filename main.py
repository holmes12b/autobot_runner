from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import time
import requests
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
WEBHOOK_URL = os.getenv("BOOKING_WEBHOOK")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

class BookingRequest(BaseModel):
    message: str

@app.post("/run-booking")
def run_booking(req: BookingRequest):
    try:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=req.message
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            if run_status.status == "completed":
                return {"status": "completed", "note": "No function call needed."}

            elif run_status.status == "requires_action":
                tool_call = run_status.required_action.submit_tool_outputs.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)

                # Post to your Flask webhook on Render
                response = requests.post(
                    WEBHOOK_URL,
                    headers={"Content-Type": "application/json"},
                    json=arguments
                )

                # Submit output back to OpenAI
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=[{
                        "tool_call_id": tool_call.id,
                        "output": "Booking logged successfully"
                    }]
                )

                return {
                    "status": "booking logged",
                    "gpt_args": arguments,
                    "webhook_response": response.json()
                }

            else:
                time.sleep(1)

    except Exception as e:
        return {"status": "error", "message": str(e)}
