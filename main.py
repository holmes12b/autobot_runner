from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import requests
from openai import OpenAI
import os

# Load environment variables
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
        # Create a new GPT thread
        thread = client.beta.threads.create()

        # Add the user's booking message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=req.message
        )

        # Start the assistant run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID
        )

        # Wait for assistant to complete or trigger a function call
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            if run_status.status == "completed":
                print("✅ GPT run completed — no function call made.")
                return {"status": "completed", "note": "No function call needed."}

            elif run_status.status == "requires_action":
                print("🤖 GPT is calling a function...")

                tool_call = run_status.required_action.submit_tool_outputs.tool_calls[0]

                print("🔎 Raw GPT function arguments:")
                print(repr(tool_call.function.arguments))

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except Exception as e:
                    print("❌ JSON parsing failed:", e)
                    return {"status": "error", "message": str(e)}

                # Forward the parsed data to your webhook
                response = requests.post(
                    WEBHOOK_URL,
                    headers={"Content-Type": "application/json"},
                    json=arguments
                )

                try:
                    webhook_output = response.json()
                except Exception as e:
                    print("⚠️  Webhook returned non-JSON response:", e)
                    webhook_output = response.text

                # Confirm the function was handled
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
                    "webhook_status": response.status_code,
                    "webhook_response": webhook_output
                }

            else:
                time.sleep(1)

    except Exception as e:
        print("❌ Error in run_booking:", e)
        return {"status": "error", "message": str(e)}